"""ROI 标定工具测试。"""

from __future__ import annotations

import unittest

from scripts.roi_calibrator import denormalize_roi, normalize_roi


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


if __name__ == "__main__":
    unittest.main()
