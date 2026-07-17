"""SOP 状态机测试。

这些测试覆盖区域串行完成、严格孔位顺序校验、零件落位后开始计时、
L 型工具紧固后才完成步骤，以及漏装超时检测。
"""

from __future__ import annotations

import unittest

from sop_monitor.models import Detection, EventType, FrameObservation, MonitorConfig, RegionSpec, StepSpec
from sop_monitor.state_machine import SopStateMachine


def make_config() -> MonitorConfig:
    return MonitorConfig(
        confidence_threshold=0.75,
        stable_frames_required=2,
        tool_evidence_frames_required=1,
        tool_leave_frames_required=2,
        tool_leave_timeout_ms=300,
        forbidden_tool_stable_frames_required=2,
        forbidden_tool_clear_frames_required=2,
        missing_timeout_frames=4,
        regions=[
            RegionSpec(
                region_id="R1",
                name="Region 1",
                steps=[
                    StepSpec(step=1, hole_id="H1", part_type="installed_part"),
                    StepSpec(step=2, hole_id="H2", part_type="installed_part"),
                ],
            ),
            RegionSpec(
                region_id="R2",
                name="Region 2",
                steps=[StepSpec(step=1, hole_id="H3", part_type="installed_part")],
            ),
        ],
    )


def obs(frame: int, *detections: Detection, timestamp_ms: int | None = None) -> FrameObservation:
    return FrameObservation(frame_index=frame, detections=list(detections), timestamp_ms=timestamp_ms)


def det(region: str, hole: str, part: str = "installed_part", confidence: float = 0.9) -> Detection:
    return Detection(region_id=region, hole_id=hole, part_type=part, present=True, confidence=confidence)


class SopStateMachineTest(unittest.TestCase):
    def test_completes_regions_in_order(self) -> None:
        machine = SopStateMachine(make_config())
        event_types: list[EventType] = []

        frames = [
            obs(1, det("R1", "H1")),
            obs(2, det("R1", "H1")),
            obs(3, det("R1", "H1"), det("R1", "H1", "l_tool_visible")),
            obs(4, det("R1", "H1")),
            obs(5, det("R1", "H1")),
            obs(6, det("R1", "H1"), det("R1", "H2")),
            obs(7, det("R1", "H1"), det("R1", "H2")),
            obs(8, det("R1", "H2"), det("R1", "H2", "l_tool_visible")),
            obs(9, det("R1", "H2")),
            obs(10, det("R1", "H2")),
            obs(11, det("R2", "H3")),
            obs(12, det("R2", "H3")),
            obs(13, det("R2", "H3"), det("R2", "H3", "l_tool_visible")),
            obs(14, det("R2", "H3")),
            obs(15, det("R2", "H3")),
        ]

        for frame in frames:
            event_types.extend(event.event_type for event in machine.update(frame))

        self.assertEqual(
            event_types,
            [
                EventType.STEP_STARTED,
                EventType.STEP_COMPLETED,
                EventType.STEP_STARTED,
                EventType.STEP_COMPLETED,
                EventType.REGION_COMPLETED,
                EventType.STEP_STARTED,
                EventType.STEP_COMPLETED,
                EventType.REGION_COMPLETED,
                EventType.ALL_COMPLETED,
            ],
        )
        self.assertTrue(machine.completed)

    def test_order_error_when_future_hole_detected_first(self) -> None:
        machine = SopStateMachine(make_config())
        first_events = machine.update(obs(1, det("R1", "H2")))
        events = machine.update(obs(2, det("R1", "H2")))

        self.assertEqual(first_events, [])
        self.assertEqual([event.event_type for event in events], [EventType.ORDER_ERROR])

    def test_skips_confirmed_out_of_order_hole_after_predecessor_completes(self) -> None:
        config = MonitorConfig(
            confidence_threshold=0.75,
            stable_frames_required=2,
            tool_evidence_frames_required=1,
            tool_leave_frames_required=2,
            regions=[RegionSpec(
                region_id="R1",
                name="Region 1",
                steps=[
                    StepSpec(step=1, hole_id="H1", part_type="installed_part"),
                    StepSpec(step=2, hole_id="H2", part_type="installed_part"),
                    StepSpec(step=3, hole_id="H3", part_type="installed_part"),
                ],
            )],
        )
        machine = SopStateMachine(config)
        events = []

        events.extend(machine.update(obs(1, det("R1", "H2"))))
        events.extend(machine.update(obs(2, det("R1", "H2"))))
        events.extend(machine.update(obs(3, det("R1", "H1"))))
        events.extend(machine.update(obs(4, det("R1", "H1"))))
        events.extend(machine.update(obs(5, det("R1", "H1", "l_tool_visible"))))
        events.extend(machine.update(obs(6, det("R1", "H1"), det("R1", "H2"))))
        events.extend(machine.update(obs(7, det("R1", "H1"), det("R1", "H2"))))

        self.assertEqual(machine.expected_step.hole_id, "H3")
        self.assertIn(EventType.ORDER_ERROR, [event.event_type for event in events])
        self.assertNotIn(
            "H2",
            [
                event.expected_hole_id
                for event in events
                if event.event_type == EventType.STEP_STARTED
            ],
        )

    def test_installed_hole_count_is_independent_of_sop_order(self) -> None:
        machine = SopStateMachine(make_config())

        machine.update(obs(1, det("R1", "H2")))
        machine.update(obs(2, det("R1", "H2")))

        self.assertSetEqual(machine.confirmed_installed_holes, {("R1", "H2")})

    def test_different_future_holes_only_count_one_order_error_for_expected_step(self) -> None:
        config = MonitorConfig(
            confidence_threshold=0.75,
            stable_frames_required=1,
            regions=[RegionSpec(
                region_id="R1",
                name="Region 1",
                steps=[
                    StepSpec(step=1, hole_id="H1", part_type="installed_part"),
                    StepSpec(step=2, hole_id="H2", part_type="installed_part"),
                    StepSpec(step=3, hole_id="H3", part_type="installed_part"),
                ],
            )],
        )
        machine = SopStateMachine(config)

        first_events = machine.update(obs(1, det("R1", "H2")))
        second_events = machine.update(obs(2, det("R1", "H3")))

        self.assertEqual([event.event_type for event in first_events], [EventType.ORDER_ERROR])
        self.assertEqual(second_events, [])
        self.assertSetEqual(
            machine.confirmed_installed_holes,
            {("R1", "H2"), ("R1", "H3")},
        )

    def test_tool_before_part_does_not_start_or_complete_step(self) -> None:
        machine = SopStateMachine(make_config())
        event_types: list[EventType] = []

        event_types.extend(event.event_type for event in machine.update(obs(1, det("R1", "H1", "l_tool_visible"))))
        event_types.extend(event.event_type for event in machine.update(obs(2, det("R1", "H1", "l_tool_visible"))))

        self.assertEqual(event_types, [])

    def test_completed_hole_remaining_visible_is_not_order_error(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "H1")))
        machine.update(obs(2, det("R1", "H1")))
        machine.update(obs(3, det("R1", "H1"), det("R1", "H1", "l_tool_visible")))
        machine.update(obs(4, det("R1", "H1")))
        machine.update(obs(5, det("R1", "H1")))

        events = machine.update(obs(6, det("R1", "H1")))

        self.assertEqual(events, [])

    def test_step_completion_records_video_timestamp_duration(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "H1"), timestamp_ms=1000))
        started_events = machine.update(obs(2, det("R1", "H1"), timestamp_ms=1600))
        machine.update(obs(3, det("R1", "H1"), det("R1", "H1", "l_tool_visible"), timestamp_ms=2000))
        machine.update(obs(4, det("R1", "H1"), timestamp_ms=2200))
        events = machine.update(obs(5, det("R1", "H1"), timestamp_ms=2600))

        self.assertEqual([event.event_type for event in started_events], [EventType.STEP_STARTED])
        self.assertEqual(machine.step_started_timestamp_ms, None)
        completed = next(event for event in events if event.event_type == EventType.STEP_COMPLETED)
        self.assertEqual(completed.timestamp_ms, 2600)
        self.assertEqual(completed.duration_ms, 1600)

    def test_part_placement_starts_timer_but_does_not_complete_step(self) -> None:
        machine = SopStateMachine(make_config())

        machine.update(obs(1, det("R1", "H1"), timestamp_ms=1000))
        events = machine.update(obs(2, det("R1", "H1"), timestamp_ms=1400))

        self.assertEqual([event.event_type for event in events], [EventType.STEP_STARTED])
        self.assertEqual(machine.step_phase, "紧固中")
        self.assertEqual(machine.step_started_timestamp_ms, 1000)
        self.assertEqual(machine.expected_step.hole_id, "H1")

    def test_tool_held_during_part_placement_is_counted(self) -> None:
        machine = SopStateMachine(make_config())

        machine.update(obs(1, det("R1", "H1"), det("R1", "H1", "l_tool_visible")))
        machine.update(obs(2, det("R1", "H1"), det("R1", "H1", "l_tool_visible")))
        machine.update(obs(3, det("R1", "H1")))
        events = machine.update(obs(4, det("R1", "H1")))

        self.assertIn(EventType.STEP_COMPLETED, [event.event_type for event in events])

    def test_next_step_keeps_its_first_stable_detection_as_start_time(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "H1"), timestamp_ms=1000))
        machine.update(obs(2, det("R1", "H1"), timestamp_ms=1100))
        machine.update(obs(3, det("R1", "H1", "l_tool_visible"), timestamp_ms=1200))

        machine.update(obs(4, det("R1", "H2"), timestamp_ms=2000))
        events = machine.update(obs(5, det("R1", "H2"), timestamp_ms=2100))

        self.assertEqual(
            [event.event_type for event in events],
            [EventType.STEP_COMPLETED, EventType.STEP_STARTED],
        )
        self.assertEqual(machine.expected_step.hole_id, "H2")
        self.assertEqual(machine.step_started_timestamp_ms, 2000)

    def test_video_end_can_finish_last_step_after_tool_and_final_part_confirmation(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "H1"), timestamp_ms=1000))
        machine.update(obs(2, det("R1", "H1"), timestamp_ms=1100))
        machine.update(obs(3, det("R1", "H1"), det("R1", "H1", "l_tool_visible"), timestamp_ms=1200))
        machine.update(obs(4, det("R1", "H1"), timestamp_ms=1300))
        last_observation = obs(5, det("R1", "H1"), timestamp_ms=1400)
        machine.update(last_observation)

        events = machine.finish(last_observation)

        self.assertIn(EventType.STEP_COMPLETED, [event.event_type for event in events])

    def test_forbidden_tool_emits_one_alarm_while_it_remains_visible(self) -> None:
        machine = SopStateMachine(make_config())

        first = machine.update(obs(1, det("R1", "*", "forbidden_tool")))
        second = machine.update(obs(2, det("R1", "*", "forbidden_tool")))
        third = machine.update(obs(3, det("R1", "*", "forbidden_tool")))

        self.assertEqual(first, [])
        self.assertEqual([event.event_type for event in second], [EventType.FORBIDDEN_TOOL])
        self.assertEqual(third, [])
        self.assertTrue(machine.forbidden_alarm_active)

    def test_forbidden_tool_alarm_rearms_after_clear_frames(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "*", "forbidden_tool")))
        machine.update(obs(2, det("R1", "*", "forbidden_tool")))
        machine.update(obs(3))
        machine.update(obs(4))

        machine.update(obs(5, det("R1", "*", "forbidden_tool")))
        events = machine.update(obs(6, det("R1", "*", "forbidden_tool")))

        self.assertEqual([event.event_type for event in events], [EventType.FORBIDDEN_TOOL])

    def test_forbidden_tool_short_timestamp_gap_does_not_create_duplicate_alarm(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "*", "forbidden_tool"), timestamp_ms=1000))
        machine.update(obs(2, det("R1", "*", "forbidden_tool"), timestamp_ms=1100))
        machine.update(obs(3, timestamp_ms=3000))
        machine.update(obs(4, timestamp_ms=4500))

        first_return = machine.update(
            obs(5, det("R1", "*", "forbidden_tool"), timestamp_ms=4600)
        )
        second_return = machine.update(
            obs(6, det("R1", "*", "forbidden_tool"), timestamp_ms=4700)
        )

        self.assertEqual(first_return, [])
        self.assertEqual(second_return, [])
        self.assertTrue(machine.forbidden_alarm_active)

    def test_forbidden_tool_timestamp_timeout_rearms_alarm(self) -> None:
        machine = SopStateMachine(make_config())
        machine.update(obs(1, det("R1", "*", "forbidden_tool"), timestamp_ms=1000))
        machine.update(obs(2, det("R1", "*", "forbidden_tool"), timestamp_ms=1100))
        machine.update(obs(3, timestamp_ms=6200))

        machine.update(obs(4, det("R1", "*", "forbidden_tool"), timestamp_ms=7000))
        events = machine.update(
            obs(5, det("R1", "*", "forbidden_tool"), timestamp_ms=7100)
        )

        self.assertEqual([event.event_type for event in events], [EventType.FORBIDDEN_TOOL])

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
            events.extend(machine.update(obs(frame_index, det("R1", "H2"))))
        self.assertEqual([event.event_type for event in events], [EventType.ORDER_ERROR])


if __name__ == "__main__":
    unittest.main()
