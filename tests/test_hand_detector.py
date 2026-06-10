"""后端手部检测辅助逻辑测试。"""

from __future__ import annotations

import unittest

from sop_monitor.hand_detector import HandObservation, any_hand_near_roi, bbox_overlaps_roi


class HandDetectorTest(unittest.TestCase):
    def test_bbox_overlaps_roi(self) -> None:
        self.assertTrue(bbox_overlaps_roi(
            bbox=(80, 80, 180, 180),
            roi=(0.1, 0.1, 0.3, 0.3),
            frame_width=640,
            frame_height=480,
        ))

    def test_bbox_outside_roi(self) -> None:
        self.assertFalse(bbox_overlaps_roi(
            bbox=(500, 350, 620, 460),
            roi=(0.1, 0.1, 0.3, 0.3),
            frame_width=640,
            frame_height=480,
        ))

    def test_any_hand_near_roi(self) -> None:
        hands = [HandObservation(landmarks=[], bbox=(80, 80, 180, 180))]
        self.assertTrue(any_hand_near_roi(hands, (0.1, 0.1, 0.3, 0.3), 640, 480))

    def test_none_roi_is_not_near(self) -> None:
        hands = [HandObservation(landmarks=[], bbox=(80, 80, 180, 180))]
        self.assertFalse(any_hand_near_roi(hands, None, 640, 480))


if __name__ == "__main__":
    unittest.main()

