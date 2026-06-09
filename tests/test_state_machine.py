"""SOP 状态机测试。

这些测试覆盖第一阶段核心业务要求：区域串行完成、严格孔位顺序校验、
只判断孔位是否已装、以及漏装超时检测。
"""

from __future__ import annotations

import unittest

from sop_monitor.models import Detection, EventType, FrameObservation, MonitorConfig, RegionSpec, StepSpec
from sop_monitor.state_machine import SopStateMachine


def make_config() -> MonitorConfig:
    return MonitorConfig(
        confidence_threshold=0.75,
        stable_frames_required=2,
        missing_timeout_frames=4,
        regions=[
            RegionSpec(
                region_id="R1",
                name="Region 1",
                steps=[
                    StepSpec(step=1, hole_id="H1", part_type="screw_A"),
                    StepSpec(step=2, hole_id="H2", part_type="screw_A"),
                ],
            ),
            RegionSpec(
                region_id="R2",
                name="Region 2",
                steps=[StepSpec(step=1, hole_id="H3", part_type="pin_B")],
            ),
        ],
    )


def obs(frame: int, *detections: Detection) -> FrameObservation:
    return FrameObservation(frame_index=frame, detections=list(detections))


def det(region: str, hole: str, part: str, confidence: float = 0.9) -> Detection:
    return Detection(region_id=region, hole_id=hole, part_type=part, present=True, confidence=confidence)


class SopStateMachineTest(unittest.TestCase):
    def test_completes_regions_in_order(self) -> None:
        machine = SopStateMachine(make_config())
        event_types: list[EventType] = []

        frames = [
            obs(1, det("R1", "H1", "screw_A")),
            obs(2, det("R1", "H1", "screw_A")),
            obs(3, det("R1", "H2", "screw_A")),
            obs(4, det("R1", "H2", "screw_A")),
            obs(5, det("R2", "H3", "pin_B")),
            obs(6, det("R2", "H3", "pin_B")),
        ]

        for frame in frames:
            event_types.extend(event.event_type for event in machine.update(frame))

        self.assertEqual(
            event_types,
            [
                EventType.STEP_COMPLETED,
                EventType.STEP_COMPLETED,
                EventType.REGION_COMPLETED,
                EventType.STEP_COMPLETED,
                EventType.REGION_COMPLETED,
                EventType.ALL_COMPLETED,
            ],
        )
        self.assertTrue(machine.completed)

    def test_order_error_when_future_hole_detected_first(self) -> None:
        machine = SopStateMachine(make_config())
        events = machine.update(obs(1, det("R1", "H2", "screw_A")))
        self.assertEqual([event.event_type for event in events], [EventType.ORDER_ERROR])

    def test_expected_hole_completes_without_part_type_check(self) -> None:
        machine = SopStateMachine(make_config())
        event_types: list[EventType] = []

        event_types.extend(event.event_type for event in machine.update(obs(1, det("R1", "H1", "pin_B"))))
        event_types.extend(event.event_type for event in machine.update(obs(2, det("R1", "H1", "pin_B"))))

        self.assertEqual(event_types, [EventType.STEP_COMPLETED])

    def test_missing_part_after_timeout(self) -> None:
        machine = SopStateMachine(make_config())
        events = []
        for frame_index in range(1, 5):
            events.extend(machine.update(obs(frame_index)))
        self.assertEqual([event.event_type for event in events], [EventType.MISSING_PART])

    def test_order_error_does_not_emit_duplicate_missing_part(self) -> None:
        machine = SopStateMachine(make_config())
        events = []
        for frame_index in range(1, 5):
            events.extend(machine.update(obs(frame_index, det("R1", "H2", "screw_A"))))
        self.assertEqual([event.event_type for event in events], [EventType.ORDER_ERROR])


if __name__ == "__main__":
    unittest.main()
