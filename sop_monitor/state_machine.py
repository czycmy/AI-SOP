"""按区域串行装配监控的 SOP 状态机。

本模块是第一阶段 MVP 的核心业务逻辑。它接收逐帧孔位/零件检测结果，
维护当前区域和当前期望 SOP 步骤，校验装配顺序，通过稳定帧投票确认孔位已装，
记录异常事件，并在当前区域通过后切换到下一区域。本阶段只判断“有没有装”，
不判断零件类型是否正确。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sop_monitor.models import (
    Detection,
    EventType,
    FrameObservation,
    MonitorConfig,
    MonitorEvent,
    RegionSpec,
    StepSpec,
)


@dataclass
class StepRuntime:
    """当前期望 SOP 步骤的临时计数器。"""

    stable_frames: int = 0
    elapsed_frames: int = 0
    emitted_error_keys: set[tuple[str, str, str]] = field(default_factory=set)


class SopStateMachine:
    """根据已配置的 SOP 顺序校验逐帧检测结果。"""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.region_index = 0
        self.step_index = 0
        self.runtime = StepRuntime()
        self.completed = False

    @property
    def active_region(self) -> RegionSpec | None:
        if self.completed:
            return None
        return self.config.regions[self.region_index]

    @property
    def expected_step(self) -> StepSpec | None:
        region = self.active_region
        if region is None:
            return None
        return region.steps[self.step_index]

    def update(self, observation: FrameObservation) -> list[MonitorEvent]:
        """处理一帧检测结果，并返回这一帧产生的业务事件。"""

        if self.completed:
            return []

        region = self.active_region
        expected = self.expected_step
        if region is None or expected is None:
            return []

        self.runtime.elapsed_frames += 1
        events: list[MonitorEvent] = []
        detections = self._active_detections(observation, region.region_id)

        # 顺序规则：等待当前期望孔位时，如果当前区域内其他孔位已经确认有零件，
        # 就说明操作顺序不符合 SOP。此时只报顺序异常，不再重复报漏装异常。
        has_out_of_order_detection = False
        for detection in detections:
            if detection.hole_id == expected.hole_id:
                continue
            if not self._is_high_confidence_present(detection):
                continue
            has_out_of_order_detection = True
            events.extend(self._emit_once(
                EventType.ORDER_ERROR,
                observation.frame_index,
                region,
                expected,
                detection,
                f"当前应装 {expected.hole_id}，但检测到 {detection.hole_id} 已有零件。",
            ))

        # 完整性规则：当前期望孔位必须有高置信度检测结果，才可能判定完成。
        expected_detection = self._best_detection_for_expected(detections, expected)
        if expected_detection is None:
            self.runtime.stable_frames = 0
            if not has_out_of_order_detection:
                events.extend(self._maybe_missing_event(observation.frame_index, region, expected))
            return events

        # 稳定帧投票：避免因为单帧误检就把一个孔位判定为装配完成。
        self.runtime.stable_frames += 1
        if self.runtime.stable_frames < self.config.stable_frames_required:
            return events

        events.append(self._event(
            EventType.STEP_COMPLETED,
            observation.frame_index,
            region,
            expected,
            expected_detection,
            f"{region.name} {expected.hole_id} 装配确认完成。",
        ))
        events.extend(self._advance(observation.frame_index, region, expected))
        return events

    def _active_detections(self, observation: FrameObservation, region_id: str) -> list[Detection]:
        """只保留当前激活物理区域内的检测结果。"""

        return [item for item in observation.detections if item.region_id == region_id]

    def _best_detection_for_expected(
        self,
        detections: list[Detection],
        expected: StepSpec,
    ) -> Detection | None:
        candidates = [
            item
            for item in detections
            if item.hole_id == expected.hole_id and self._is_high_confidence_present(item)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _is_high_confidence_present(self, detection: Detection) -> bool:
        return detection.present and detection.confidence >= self.config.confidence_threshold

    def _advance(
        self,
        frame_index: int,
        region: RegionSpec,
        expected: StepSpec,
    ) -> list[MonitorEvent]:
        """推进到下一个 SOP 步骤、下一个区域或全部完成状态。"""

        self.step_index += 1
        self.runtime = StepRuntime()

        if self.step_index < len(region.steps):
            return []

        events = [
            self._event(
                EventType.REGION_COMPLETED,
                frame_index,
                region,
                expected,
                None,
                f"{region.name} 校验通过。",
            )
        ]

        self.region_index += 1
        self.step_index = 0
        if self.region_index >= len(self.config.regions):
            self.completed = True
            events.append(self._event(
                EventType.ALL_COMPLETED,
                frame_index,
                region,
                expected,
                None,
                "全部区域 SOP 校验完成。",
            ))

        return events

    def _maybe_missing_event(
        self,
        frame_index: int,
        region: RegionSpec,
        expected: StepSpec,
    ) -> list[MonitorEvent]:
        """当前步骤等待过久仍未确认时，产生漏装事件。"""

        if self.runtime.elapsed_frames < self.config.missing_timeout_frames:
            return []
        return self._emit_once(
            EventType.MISSING_PART,
            frame_index,
            region,
            expected,
            None,
            f"等待超时，{expected.hole_id} 未确认装配完成。",
        )

    def _emit_once(
        self,
        event_type: EventType,
        frame_index: int,
        region: RegionSpec,
        expected: StepSpec,
        detection: Detection | None,
        message: str,
    ) -> list[MonitorEvent]:
        """避免同一个当前步骤反复记录完全相同的异常。"""

        observed_hole_id = detection.hole_id if detection else ""
        # 当前阶段不判断零件类型，同一孔位的同类异常只记录一次即可。
        key = (event_type.value, observed_hole_id, "")
        if key in self.runtime.emitted_error_keys:
            return []
        self.runtime.emitted_error_keys.add(key)
        return [self._event(event_type, frame_index, region, expected, detection, message)]

    def _event(
        self,
        event_type: EventType,
        frame_index: int,
        region: RegionSpec,
        expected: StepSpec,
        detection: Detection | None,
        message: str,
    ) -> MonitorEvent:
        """构建标准化事件对象，供日志或界面展示使用。"""

        return MonitorEvent(
            event_type=event_type,
            frame_index=frame_index,
            region_id=region.region_id,
            expected_step=expected.step,
            expected_hole_id=expected.hole_id,
            expected_part_type=expected.part_type,
            observed_hole_id=detection.hole_id if detection else None,
            observed_part_type=detection.part_type if detection else None,
            confidence=detection.confidence if detection else None,
            message=message,
        )
