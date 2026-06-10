"""摄像头检测工具测试。"""

from __future__ import annotations

import unittest

from sop_monitor.camera_utils import match_detection_to_hole
from sop_monitor.config import load_config


class CameraUtilsTest(unittest.TestCase):
    def test_config_loads_step_roi(self) -> None:
        config = load_config("configs/sample_sop.json")
        self.assertEqual(config.regions[0].steps[0].roi, (0.12, 0.22, 0.28, 0.46))

    def test_match_detection_to_hole_by_bbox_center(self) -> None:
        config = load_config("configs/sample_sop.json")
        matched = match_detection_to_hole(
            config,
            bbox=(130, 120, 190, 180),
            frame_width=800,
            frame_height=600,
        )
        self.assertIsNotNone(matched)
        region_id, step = matched
        self.assertEqual(region_id, "R1")
        self.assertEqual(step.hole_id, "H1")

    def test_unmatched_detection_returns_none(self) -> None:
        config = load_config("configs/sample_sop.json")
        matched = match_detection_to_hole(
            config,
            bbox=(760, 20, 790, 50),
            frame_width=800,
            frame_height=600,
        )
        self.assertIsNone(matched)


if __name__ == "__main__":
    unittest.main()

