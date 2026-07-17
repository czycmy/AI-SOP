"""按区域串行装配监控的 SOP 状态机。

本模块是第一阶段 MVP 的核心业务逻辑。它接收逐帧孔位/零件检测结果，
维护当前区域和当前期望 SOP 步骤，校验装配顺序，并通过“零件稳定落位、L 型
工具参与紧固、工具离开且零件重新稳定可见”三个阶段确认孔位完成。ROI 和检测框
只用于后台判断，不属于业务状态本身。
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

    part_stable_frames: int = 0
    final_part_stable_frames: int = 0
    tool_evidence_frames: int = 0
    tool_leave_frames: int = 0
    next_step_stable_frames: int = 0
    elapsed_frames: int = 0
    part_placed: bool = False
    tool_seen: bool = False
    part_candidate_timestamp_ms: int | None = None
    started_timestamp_ms: int | None = None
    tool_last_seen_timestamp_ms: int | None = None
    next_step_candidate_timestamp_ms: int | None = None
    emitted_error_keys: set[tuple[str, str, str]] = field(default_factory=set)


class SopStateMachine:
    """根据已配置的 SOP 顺序校验逐帧检测结果。"""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.region_index = 0
        self.step_index = 0
        self.runtime = StepRuntime()
        self.completed = False
        self.installed_hole_frames: dict[tuple[str, str], int] = {}
        self.confirmed_installed_holes: set[tuple[str, str]] = set()
        self.reported_order_expected_steps: set[tuple[str, str]] = set()
        self.reported_order_observed_holes: set[tuple[str, str]] = set()
        self.forbidden_tool_stable_frames = 0
        self.forbidden_tool_clear_frames = 0
        self.forbidden_tool_last_seen_timestamp_ms: int | None = None
        self.forbidden_alarm_active = False

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

    @property
    def step_phase(self) -> str:
        """返回界面使用的当前步骤阶段。"""

        if self.completed:
            return "完成"
        return "紧固中" if self.runtime.part_placed else "等待零件"

    @property
    def step_started_timestamp_ms(self) -> int | None:
        """当前孔位零件稳定落位后的计时起点。"""

        return self.runtime.started_timestamp_ms

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
        self._track_installed_holes(observation.detections)
        detections = self._active_detections(observation, region.region_id)
        events.extend(self._update_forbidden_tool(observation, region, expected, detections))

        expected_part = self._best_detection_for_class(detections, expected, expected.part_type)
        expected_tool = self._best_tool_detection(detections)

        # 已完成孔位中的零件会一直留在画面中，只检查当前步骤之后的孔位。
        # 当前步骤已经经历工具紧固时，紧邻的下一孔位稳定出现也可证明操作员已转序。
        has_out_of_order_detection = False
        next_step_detection: Detection | None = None
        next_step = region.steps[self.step_index + 1] if self.step_index + 1 < len(region.steps) else None
        future_steps = {step.hole_id: step for step in region.steps[self.step_index + 1:]}
        for detection in detections:
            future_step = future_steps.get(detection.hole_id)
            if future_step is None:
                continue
            if not self._is_high_confidence_present(detection, future_step.part_type):
                continue
            if (
                next_step is not None
                and detection.hole_id == next_step.hole_id
                and self.runtime.part_placed
                and self.runtime.tool_seen
            ):
                next_step_detection = detection
                continue
            has_out_of_order_detection = True
            observed_key = (region.region_id, detection.hole_id)
            if observed_key not in self.confirmed_installed_holes:
                # 顺序异常也要经过稳定帧确认，避免单帧误检改变 SOP 状态。
                continue
            events.extend(self._emit_order_once(
                observation,
                region,
                expected,
                detection,
                f"当前应装 {expected.hole_id}，但检测到 {detection.hole_id} 已有零件。",
            ))

        if next_step_detection is not None:
            if self.runtime.next_step_stable_frames == 0:
                self.runtime.next_step_candidate_timestamp_ms = observation.timestamp_ms
            self.runtime.next_step_stable_frames += 1
            if self.runtime.next_step_stable_frames >= self.config.stable_frames_required:
                next_started_timestamp_ms = self.runtime.next_step_candidate_timestamp_ms
                events.extend(self._complete_step(
                    observation,
                    region,
                    expected,
                    expected_part,
                    f"{region.name} {expected.hole_id} 紧固结束，已进入下一孔位。",
                ))
                events.extend(self._start_next_from_transition(
                    observation,
                    region,
                    next_step,
                    next_step_detection,
                    expected_tool,
                    next_started_timestamp_ms,
                ))
                return events
        else:
            self.runtime.next_step_stable_frames = 0
            self.runtime.next_step_candidate_timestamp_ms = None

        if not self.runtime.part_placed:
            events.extend(self._update_waiting_for_part(
                observation,
                region,
                expected,
                expected_part,
                expected_tool,
            ))
            if expected_part is None and not has_out_of_order_detection:
                events.extend(self._maybe_missing_event(observation, region, expected))
            return events

        events.extend(self._update_tightening(
            observation,
            region,
            expected,
            expected_part,
            expected_tool,
        ))
        return events

    def _track_installed_holes(self, detections: list[Detection]) -> None:
        """独立统计实际已装孔位，不受 SOP 顺序异常和工具异常影响。"""

        detected_keys = {
            (detection.region_id, detection.hole_id)
            for detection in detections
            if self._is_high_confidence_present(detection, "installed_part")
        }
        configured_keys = {
            (region.region_id, step.hole_id)
            for region in self.config.regions
            for step in region.steps
        }
        for key in configured_keys:
            if key in self.confirmed_installed_holes:
                continue
            if key in detected_keys:
                self.installed_hole_frames[key] = self.installed_hole_frames.get(key, 0) + 1
                if self.installed_hole_frames[key] >= self.config.stable_frames_required:
                    self.confirmed_installed_holes.add(key)
            else:
                self.installed_hole_frames[key] = 0

    def _update_forbidden_tool(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        detections: list[Detection],
    ) -> list[MonitorEvent]:
        """稳定检测锉刀后报警一次，清除一段时间后才允许再次报警。"""

        forbidden_detections = [
            item
            for item in detections
            if self._is_high_confidence_present(
                item,
                self.config.forbidden_tool_class,
                self.config.forbidden_tool_confidence_threshold,
            )
        ]
        if forbidden_detections:
            self.forbidden_tool_stable_frames += 1
            self.forbidden_tool_clear_frames = 0
            self.forbidden_tool_last_seen_timestamp_ms = observation.timestamp_ms
            if (
                self.forbidden_tool_stable_frames >= self.config.forbidden_tool_stable_frames_required
                and not self.forbidden_alarm_active
            ):
                self.forbidden_alarm_active = True
                detection = max(forbidden_detections, key=lambda item: item.confidence)
                return [self._event(
                    EventType.FORBIDDEN_TOOL,
                    observation,
                    region,
                    expected,
                    detection,
                    "检测到禁止使用的锉刀，请立即停止作业。",
                )]
            return []

        self.forbidden_tool_stable_frames = 0
        if self.forbidden_alarm_active:
            self.forbidden_tool_clear_frames += 1
            if (
                observation.timestamp_ms is not None
                and self.forbidden_tool_last_seen_timestamp_ms is not None
            ):
                should_clear = (
                    observation.timestamp_ms - self.forbidden_tool_last_seen_timestamp_ms
                    >= self.config.forbidden_tool_clear_timeout_ms
                )
            else:
                should_clear = (
                    self.forbidden_tool_clear_frames
                    >= self.config.forbidden_tool_clear_frames_required
                )
            if should_clear:
                self.forbidden_alarm_active = False
                self.forbidden_tool_clear_frames = 0
                self.forbidden_tool_last_seen_timestamp_ms = None
        return []

    def finish(self, observation: FrameObservation) -> list[MonitorEvent]:
        """离线视频结束时，在最终零件已稳定可见的前提下收口最后一步。"""

        if self.completed or not self.runtime.part_placed or not self.runtime.tool_seen:
            return []
        if self.runtime.final_part_stable_frames < self.config.stable_frames_required:
            return []
        region = self.active_region
        expected = self.expected_step
        if region is None or expected is None:
            return []
        detections = self._active_detections(observation, region.region_id)
        expected_part = self._best_detection_for_class(detections, expected, expected.part_type)
        return self._complete_step(
            observation,
            region,
            expected,
            expected_part,
            f"{region.name} {expected.hole_id} 紧固完成。",
        )

    def _active_detections(self, observation: FrameObservation, region_id: str) -> list[Detection]:
        """只保留当前激活物理区域内的检测结果。"""

        return [item for item in observation.detections if item.region_id == region_id]

    def _best_detection_for_class(
        self,
        detections: list[Detection],
        expected: StepSpec,
        class_name: str,
    ) -> Detection | None:
        candidates = [
            item
            for item in detections
            if item.hole_id == expected.hole_id
            and self._is_high_confidence_present(item, class_name)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _best_tool_detection(self, detections: list[Detection]) -> Detection | None:
        """工具在紧固过程中会移动，只要求它位于当前总监控区域。"""

        candidates = [
            item
            for item in detections
            if self._is_high_confidence_present(
                item,
                self.config.tool_class,
                self.config.tool_confidence_threshold,
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _is_high_confidence_present(
        self,
        detection: Detection,
        expected_part_type: str,
        confidence_threshold: float | None = None,
    ) -> bool:
        """只让当前步骤要求的检测类别参与 SOP 判定。"""

        threshold = self.config.confidence_threshold if confidence_threshold is None else confidence_threshold
        return (
            detection.part_type == expected_part_type
            and detection.present
            and detection.confidence >= threshold
        )

    def _update_waiting_for_part(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        expected_part: Detection | None,
        expected_tool: Detection | None,
    ) -> list[MonitorEvent]:
        """等待零件稳定落位；落位时间才是当前孔位的计时起点。"""

        if expected_part is None:
            self.runtime.part_stable_frames = 0
            self.runtime.part_candidate_timestamp_ms = None
            self.runtime.tool_evidence_frames = 0
            self.runtime.tool_seen = False
            self.runtime.tool_last_seen_timestamp_ms = None
            return []

        if self.runtime.part_stable_frames == 0:
            self.runtime.part_candidate_timestamp_ms = observation.timestamp_ms
        self.runtime.part_stable_frames += 1

        # 工人可能在放入零件时已经握着工具，因此从零件首次出现起就允许记录工具。
        if expected_tool is not None:
            self.runtime.tool_evidence_frames += 1
            self.runtime.tool_last_seen_timestamp_ms = observation.timestamp_ms
            if self.runtime.tool_evidence_frames >= self.config.tool_evidence_frames_required:
                self.runtime.tool_seen = True

        if self.runtime.part_stable_frames < self.config.stable_frames_required:
            return []

        self.runtime.part_placed = True
        self.runtime.started_timestamp_ms = self.runtime.part_candidate_timestamp_ms
        if self.runtime.started_timestamp_ms is None:
            self.runtime.started_timestamp_ms = observation.timestamp_ms
        return [self._event(
            EventType.STEP_STARTED,
            observation,
            region,
            expected,
            expected_part,
            f"{region.name} {expected.hole_id} 零件已落位，开始紧固计时。",
        )]

    def _update_tightening(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        expected_part: Detection | None,
        expected_tool: Detection | None,
    ) -> list[MonitorEvent]:
        """确认工具参与紧固，并在工具离开、零件稳定可见后完成步骤。"""

        if expected_tool is not None:
            self.runtime.tool_evidence_frames += 1
            self.runtime.tool_last_seen_timestamp_ms = observation.timestamp_ms
            if self.runtime.tool_evidence_frames >= self.config.tool_evidence_frames_required:
                self.runtime.tool_seen = True
            self.runtime.tool_leave_frames = 0
            self.runtime.final_part_stable_frames = 0
            return []

        if not self.runtime.tool_seen:
            return []

        self.runtime.tool_leave_frames += 1
        if expected_part is not None:
            self.runtime.final_part_stable_frames += 1
        else:
            self.runtime.final_part_stable_frames = 0

        if (
            observation.timestamp_ms is not None
            and self.runtime.tool_last_seen_timestamp_ms is not None
        ):
            tool_has_left = (
                observation.timestamp_ms - self.runtime.tool_last_seen_timestamp_ms
                >= self.config.tool_leave_timeout_ms
            )
        else:
            tool_has_left = self.runtime.tool_leave_frames >= self.config.tool_leave_frames_required

        if not tool_has_left:
            return []
        if self.runtime.final_part_stable_frames < self.config.stable_frames_required:
            return []

        return self._complete_step(
            observation,
            region,
            expected,
            expected_part,
            f"{region.name} {expected.hole_id} 紧固完成。",
        )

    def _complete_step(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        expected_part: Detection | None,
        message: str,
    ) -> list[MonitorEvent]:
        """记录当前孔位耗时并推进 SOP。"""

        events = [self._event(
            EventType.STEP_COMPLETED,
            observation,
            region,
            expected,
            expected_part,
            message,
            duration_ms=self._step_duration_ms(observation),
        )]
        events.extend(self._advance(observation, region, expected))
        return events

    def _start_next_from_transition(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        next_step: StepSpec | None,
        next_step_detection: Detection,
        expected_tool: Detection | None,
        started_timestamp_ms: int | None,
    ) -> list[MonitorEvent]:
        """下一孔位已稳定出现时，保留它的首次落位时间并立即开始计时。"""

        if self.completed or next_step is None or self.expected_step != next_step:
            return []
        self.runtime.part_placed = True
        self.runtime.part_stable_frames = self.config.stable_frames_required
        self.runtime.part_candidate_timestamp_ms = started_timestamp_ms
        self.runtime.started_timestamp_ms = started_timestamp_ms
        if self.runtime.started_timestamp_ms is None:
            self.runtime.started_timestamp_ms = observation.timestamp_ms
        if expected_tool is not None:
            self.runtime.tool_evidence_frames = 1
            self.runtime.tool_last_seen_timestamp_ms = observation.timestamp_ms
            self.runtime.tool_seen = self.config.tool_evidence_frames_required <= 1
        return [self._event(
            EventType.STEP_STARTED,
            observation,
            region,
            next_step,
            next_step_detection,
            f"{region.name} {next_step.hole_id} 零件已落位，开始紧固计时。",
        )]

    def _advance(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
    ) -> list[MonitorEvent]:
        """推进 SOP，并跳过已经确认提前安装的异常孔位。"""

        self.step_index += 1
        self.runtime = StepRuntime()

        # 提前安装的孔位已经记录过顺序异常，前序步骤完成后不再倒回去重新计时。
        while self.step_index < len(region.steps):
            next_step = region.steps[self.step_index]
            next_key = (region.region_id, next_step.hole_id)
            if (
                next_key not in self.reported_order_observed_holes
                or next_key not in self.confirmed_installed_holes
            ):
                break
            self.step_index += 1

        if self.step_index < len(region.steps):
            return []

        region_has_order_error = any(
            region_id == region.region_id
            for region_id, _ in self.reported_order_observed_holes
        )

        events = [
            self._event(
                EventType.REGION_COMPLETED,
                observation,
                region,
                expected,
                None,
                (
                    f"{region.name} 监控完成，存在顺序异常。"
                    if region_has_order_error
                    else f"{region.name} 校验通过。"
                ),
            )
        ]

        self.region_index += 1
        self.step_index = 0
        if self.region_index >= len(self.config.regions):
            self.completed = True
            events.append(self._event(
                EventType.ALL_COMPLETED,
                observation,
                region,
                expected,
                None,
                (
                    "全部区域 SOP 监控完成，存在异常。"
                    if self.reported_order_observed_holes
                    else "全部区域 SOP 校验完成。"
                ),
            ))

        return events

    def _maybe_missing_event(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
    ) -> list[MonitorEvent]:
        """当前步骤等待过久仍未确认时，产生漏装事件。"""

        if self.config.missing_timeout_frames <= 0:
            return []
        if self.runtime.elapsed_frames < self.config.missing_timeout_frames:
            return []
        return self._emit_once(
            EventType.MISSING_PART,
            observation,
            region,
            expected,
            None,
            f"等待超时，{expected.hole_id} 未确认装配完成。",
        )

    def _emit_once(
        self,
        event_type: EventType,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        detection: Detection | None,
        message: str,
    ) -> list[MonitorEvent]:
        """避免同一个当前步骤反复记录完全相同的异常。"""

        observed_hole_id = detection.hole_id if detection else ""
        # 同一个期望步骤只记录一次顺序/漏装异常，避免后续孔位连续出现时重复累计。
        if event_type in {EventType.ORDER_ERROR, EventType.MISSING_PART}:
            key = (event_type.value, expected.hole_id, "")
        else:
            key = (event_type.value, observed_hole_id, "")
        if key in self.runtime.emitted_error_keys:
            return []
        self.runtime.emitted_error_keys.add(key)
        return [self._event(event_type, observation, region, expected, detection, message)]

    def _emit_order_once(
        self,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        detection: Detection,
        message: str,
    ) -> list[MonitorEvent]:
        """同一期望步骤或同一提前孔位只记录一次顺序异常。"""

        expected_key = (region.region_id, expected.hole_id)
        observed_key = (region.region_id, detection.hole_id)
        if (
            expected_key in self.reported_order_expected_steps
            or observed_key in self.reported_order_observed_holes
        ):
            return []
        self.reported_order_expected_steps.add(expected_key)
        self.reported_order_observed_holes.add(observed_key)
        return [self._event(
            EventType.ORDER_ERROR,
            observation,
            region,
            expected,
            detection,
            message,
        )]

    def _event(
        self,
        event_type: EventType,
        observation: FrameObservation,
        region: RegionSpec,
        expected: StepSpec,
        detection: Detection | None,
        message: str,
        duration_ms: int | None = None,
    ) -> MonitorEvent:
        """构建标准化事件对象，供日志或界面展示使用。"""

        return MonitorEvent(
            event_type=event_type,
            frame_index=observation.frame_index,
            region_id=region.region_id,
            expected_step=expected.step,
            expected_hole_id=expected.hole_id,
            expected_part_type=expected.part_type,
            observed_hole_id=detection.hole_id if detection else None,
            observed_part_type=detection.part_type if detection else None,
            confidence=detection.confidence if detection else None,
            message=message,
            timestamp_ms=observation.timestamp_ms,
            duration_ms=duration_ms,
        )

    def _step_duration_ms(self, observation: FrameObservation) -> int | None:
        """根据视频/摄像头时间戳计算当前步骤耗时。"""

        if observation.timestamp_ms is None or self.runtime.started_timestamp_ms is None:
            return None
        return max(0, observation.timestamp_ms - self.runtime.started_timestamp_ms)
