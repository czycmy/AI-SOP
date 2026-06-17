"""摄像头检测工具测试。"""

from __future__ import annotations

import unittest

from argparse import Namespace

from sop_monitor.camera_utils import (
    build_hikvision_rtsp_url,
    match_detection_to_hole,
    normalize_camera_source,
    resolve_camera_source,
)
from sop_monitor.config import load_config


class CameraUtilsTest(unittest.TestCase):
    def test_numeric_camera_source_converts_to_index(self) -> None:
        self.assertEqual(normalize_camera_source("0"), 0)

    def test_rtsp_camera_source_stays_string(self) -> None:
        source = "rtsp://admin:pwd@192.168.1.10:554/Streaming/Channels/101"
        self.assertEqual(normalize_camera_source(source), source)

    def test_build_hikvision_rtsp_url(self) -> None:
        self.assertEqual(
            build_hikvision_rtsp_url("192.168.114.222", "admin", "pwd"),
            "rtsp://admin:pwd@192.168.114.222:554/Streaming/Channels/101",
        )

    def test_resolve_hikvision_source(self) -> None:
        source = resolve_camera_source(Namespace(
            camera="0",
            hikvision_ip="192.168.114.222",
            hikvision_user="admin",
            hikvision_password="pwd",
            hikvision_channel="102",
        ))
        self.assertEqual(source, "rtsp://admin:pwd@192.168.114.222:554/Streaming/Channels/102")

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
