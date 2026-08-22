"""连续动作识别共享预处理的单元测试。"""

from __future__ import annotations

import unittest
from collections import deque

import cv2
import numpy as np
import torch

from sop_monitor.action_recognition import (
    ActionEventStateMachine,
    MultiRoiActionEventStateMachine,
    directional_flow_frames,
    frames_to_clip,
    fuse_action_probabilities,
    select_active_roi,
)
from sop_monitor.action_runtime import action_rois_match, sample_timed_frames


class ActionRecognitionTest(unittest.TestCase):
    """验证双ROI选择和模型输入转换。"""

    def test_select_active_roi_uses_motion_energy(self) -> None:
        rois = {
            "H3": (0.0, 0.0, 0.5, 1.0),
            "H4": (0.5, 0.0, 1.0, 1.0),
        }
        frames = []
        for offset in (0, 8, 16):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[30:50, 55 + offset:65 + offset] = 255
            frames.append(frame)
        name, selected, energies = select_active_roi(frames, rois, 64)
        self.assertEqual(name, "H4")
        self.assertGreater(energies["H4"], energies["H3"])
        self.assertEqual(selected[0].shape, (64, 64, 3))

    def test_rgb_motion_and_flow_clip_shapes_match(self) -> None:
        frames = [np.full((32, 32, 3), value, dtype=np.uint8) for value in (0, 30, 60)]
        rgb = frames_to_clip(frames, "rgb")
        motion = frames_to_clip(frames, "motion")
        flow = frames_to_clip(frames, "flow")
        self.assertEqual(rgb.shape, torch.Size([3, 3, 32, 32]))
        self.assertEqual(motion.shape, rgb.shape)
        self.assertEqual(flow.shape, rgb.shape)
        self.assertTrue(torch.isfinite(rgb).all())
        self.assertTrue(torch.isfinite(motion).all())
        self.assertTrue(torch.isfinite(flow).all())

    def test_directional_flow_preserves_horizontal_direction(self) -> None:
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        base = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

        def shifted(offset: int):
            matrix = np.float32([[1, 0, offset], [0, 1, 0]])
            return cv2.warpAffine(base, matrix, (64, 64), borderMode=cv2.BORDER_REFLECT)

        right = directional_flow_frames([shifted(0), shifted(2), shifted(4)])
        left = directional_flow_frames([shifted(0), shifted(-2), shifted(-4)])
        self.assertGreater(float(right[-1][8:-8, 8:-8, 0].mean()), 0.05)
        self.assertLess(float(left[-1][8:-8, 8:-8, 0].mean()), -0.05)

    def test_probability_fusion_uses_configured_weight(self) -> None:
        self.assertAlmostEqual(fuse_action_probabilities(0.8, 0.2, 0.75), 0.65)

    def test_event_state_machine_counts_one_continuous_action_once(self) -> None:
        machine = ActionEventStateMachine(
            trigger_threshold=0.6,
            clear_threshold=0.3,
            vote_window=4,
            trigger_votes=3,
            clear_windows=3,
        )
        updates = [
            machine.update(score, timestamp)
            for timestamp, score in enumerate((0.7, 0.2, 0.8, 0.9))
        ]
        self.assertTrue(updates[-1].event_started)
        self.assertEqual(updates[-1].event_count, 1)

        for timestamp, score in enumerate((0.95, 0.7, 0.8), start=4):
            update = machine.update(score, timestamp)
            self.assertFalse(update.event_started)
            self.assertEqual(update.event_count, 1)

        low_updates = [
            machine.update(0.1, timestamp)
            for timestamp in (7.0, 8.0, 9.0)
        ]
        self.assertTrue(low_updates[-1].event_ended)
        self.assertFalse(low_updates[-1].active)
        self.assertEqual(low_updates[-1].event_count, 1)
        self.assertEqual(low_updates[-1].event_end_seconds, 7.0)

        for timestamp in (10.0, 11.0, 12.0):
            update = machine.update(0.9, timestamp)
        self.assertTrue(update.event_started)
        self.assertEqual(update.event_count, 2)

    def test_multi_roi_votes_cannot_be_mixed_into_false_alarm(self) -> None:
        machine = MultiRoiActionEventStateMachine(
            roi_names={"H3", "H4"},
            trigger_threshold=0.5,
            clear_threshold=0.35,
            vote_window=4,
            trigger_votes=3,
            clear_windows=4,
        )

        scores = (
            {"H3": 0.54, "H4": 0.44},
            {"H3": 0.61, "H4": 0.55},
            {"H3": 0.34, "H4": 0.55},
        )
        updates = [machine.update(probabilities, index) for index, probabilities in enumerate(scores)]

        self.assertFalse(updates[-1].active)
        self.assertEqual(updates[-1].event_count, 0)

    def test_multi_roi_keeps_one_global_event_for_overlapping_rois(self) -> None:
        machine = MultiRoiActionEventStateMachine(
            roi_names={"H3", "H4"},
            trigger_threshold=0.5,
            clear_threshold=0.35,
            vote_window=4,
            trigger_votes=3,
            clear_windows=4,
        )

        updates = [
            machine.update({"H3": h3, "H4": h4}, timestamp)
            for timestamp, (h3, h4) in enumerate(
                ((0.66, 0.47), (0.83, 0.90), (0.94, 0.95), (0.98, 0.92))
            )
        ]

        self.assertTrue(updates[2].event_started)
        self.assertTrue(updates[-1].active)
        self.assertEqual(updates[-1].event_count, 1)

    def test_timed_frame_sampling_uses_entire_window(self) -> None:
        items = deque([
            (0, np.full((8, 8, 3), 0, dtype=np.uint8)),
            (100, np.full((8, 8, 3), 100, dtype=np.uint8)),
            (200, np.full((8, 8, 3), 200, dtype=np.uint8)),
        ])
        sampled = sample_timed_frames(items, 200, 200, 3, 8)
        self.assertEqual([int(frame.mean()) for frame in sampled], [0, 100, 200])

    def test_action_roi_compatibility_requires_same_coordinates(self) -> None:
        first = {"H3": (0.1, 0.1, 0.4, 0.4), "H4": (0.5, 0.1, 0.8, 0.4)}
        self.assertTrue(action_rois_match(first, dict(first)))
        second = {**first, "H4": (0.5, 0.1, 0.9, 0.4)}
        self.assertFalse(action_rois_match(first, second))


if __name__ == "__main__":
    unittest.main()
