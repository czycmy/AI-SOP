"""摄像头检测工具测试。"""

from __future__ import annotations

import unittest

from argparse import Namespace

from sop_monitor.camera_utils import (
    build_hikvision_rtsp_url,
    draw_visible_detection_boxes,
    is_rtsp_source,
    match_detection_to_hole,
    match_detection_to_region,
    normalize_camera_source,
    resolve_camera_source,
)
from sop_monitor.config import load_config
from sop_monitor.models import Detection, MonitorConfig, RegionSpec, StepSpec


def make_roi_config() -> MonitorConfig:
    return MonitorConfig(regions=[
        RegionSpec(
            region_id="R1",
            name="区域一",
            roi=(0.1, 0.1, 0.6, 0.6),
            steps=[
                StepSpec(
                    step=1,
                    hole_id="H1",
                    part_type="installed_part",
                    roi=(0.12, 0.22, 0.28, 0.46),
                ),
            ],
        ),
    ])


class CameraUtilsTest(unittest.TestCase):
    def test_numeric_camera_source_converts_to_index(self) -> None:
        self.assertEqual(normalize_camera_source("0"), 0)

    def test_rtsp_camera_source_stays_string(self) -> None:
        source = "rtsp://admin:pwd@192.168.1.10:554/Streaming/Channels/101"
        self.assertEqual(normalize_camera_source(source), source)

    def test_is_rtsp_source(self) -> None:
        self.assertTrue(is_rtsp_source("rtsp://admin:pwd@192.168.1.10/stream"))
        self.assertFalse(is_rtsp_source("0"))

    def test_build_hikvision_rtsp_url(self) -> None:
        self.assertEqual(
            build_hikvision_rtsp_url("192.0.2.10", "admin", "pwd"),
            "rtsp://admin:pwd@192.0.2.10:554/Streaming/Channels/101",
        )

    def test_resolve_hikvision_source(self) -> None:
        source = resolve_camera_source(Namespace(
            camera="0",
            hikvision_ip="192.0.2.10",
            hikvision_user="admin",
            hikvision_password="pwd",
            hikvision_channel="102",
        ))
        self.assertEqual(source, "rtsp://admin:pwd@192.0.2.10:554/Streaming/Channels/102")

    def test_config_loads_six_steps(self) -> None:
        config = load_config("configs/sample_sop.json")
        self.assertEqual(len(config.regions[0].steps), 6)
        self.assertIsNone(config.regions[0].steps[0].roi)

    def test_match_detection_to_hole_by_bbox_center(self) -> None:
        config = make_roi_config()
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
        config = make_roi_config()
        matched = match_detection_to_hole(
            config,
            bbox=(760, 20, 790, 50),
            frame_width=800,
            frame_height=600,
        )
        self.assertIsNone(matched)

    def test_moving_tool_matches_total_region_without_entering_hole_roi(self) -> None:
        config = make_roi_config()
        region_id = match_detection_to_region(
            config,
            bbox=(320, 240, 360, 280),
            frame_width=800,
            frame_height=600,
        )
        self.assertEqual(region_id, "R1")

    def test_visible_boxes_draw_parts_and_forbidden_tool_only(self) -> None:
        import numpy as np

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            Detection("R1", "H1", "installed_part", True, 0.9, (10, 10, 30, 30)),
            Detection("R1", "*", "forbidden_tool", True, 0.9, (40, 40, 60, 60)),
            Detection("R1", "*", "l_tool_visible", True, 0.9, (70, 70, 90, 90)),
        ]

        draw_visible_detection_boxes(frame, detections)

        self.assertTupleEqual(tuple(frame[10, 10]), (32, 220, 96))
        self.assertTupleEqual(tuple(frame[40, 40]), (40, 40, 235))
        self.assertTupleEqual(tuple(frame[70, 70]), (0, 0, 0))

    def test_region_roi_filters_detection_before_hole_match(self) -> None:
        config = MonitorConfig(regions=[
            RegionSpec(
                region_id="R1",
                name="区域一",
                roi=(0.7, 0.7, 0.9, 0.9),
                steps=[StepSpec(
                    step=1,
                    hole_id="H1",
                    part_type="installed_part",
                    roi=(0.12, 0.22, 0.28, 0.46),
                )],
            ),
        ])
        matched = match_detection_to_hole(
            config,
            bbox=(130, 120, 190, 180),
            frame_width=800,
            frame_height=600,
        )
        self.assertIsNone(matched)


if __name__ == "__main__":
    unittest.main()
