"""ROI 标定工具测试。"""

from __future__ import annotations

import unittest

from scripts.roi_calibrator import denormalize_roi, normalize_roi, resolve_hole_selectors


class RoiCalibratorTest(unittest.TestCase):
    def test_normalize_roi(self) -> None:
        self.assertEqual(
            normalize_roi((10, 20, 30, 40), image_width=100, image_height=200),
            [0.1, 0.1, 0.4, 0.3],
        )

    def test_denormalize_roi(self) -> None:
        self.assertEqual(
            denormalize_roi([0.1, 0.1, 0.4, 0.3], image_width=100, image_height=200),
            (10, 20, 40, 60),
        )

    def test_resolve_single_hole_without_changing_other_holes(self) -> None:
        config = {
            "regions": [
                {
                    "region_id": "R1",
                    "steps": [{"hole_id": "H1"}, {"hole_id": "H2"}],
                }
            ]
        }

        self.assertEqual(resolve_hole_selectors(config, ["H2"]), {("R1", "H2")})
        self.assertEqual(resolve_hole_selectors(config, ["R1-H1"]), {("R1", "H1")})

    def test_resolve_hole_rejects_unknown_name(self) -> None:
        config = {"regions": [{"region_id": "R1", "steps": [{"hole_id": "H1"}]}]}

        with self.assertRaisesRegex(ValueError, "不存在孔位 H2"):
            resolve_hole_selectors(config, ["H2"])


if __name__ == "__main__":
    unittest.main()
