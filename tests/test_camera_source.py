"""摄像头后端抽象测试。"""

from __future__ import annotations

import unittest

from sop_monitor.camera_source import CameraSourceSpec, HikvisionSdkFrameSource, create_frame_source


class CameraSourceFactoryTest(unittest.TestCase):
    def test_hikvision_sdk_backend_requires_windows_runtime(self) -> None:
        with self.assertRaises(RuntimeError):
            create_frame_source(CameraSourceSpec(
                backend="hikvision-sdk",
                source="rtsp://admin:pwd@192.168.1.10:554/Streaming/Channels/101",
                hikvision_ip="192.168.1.10",
                hikvision_password="pwd",
            ))

    def test_unknown_backend_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            create_frame_source(CameraSourceSpec(backend="unknown", source="0"))

    def test_hikvision_sdk_class_stays_out_of_ui_layer(self) -> None:
        self.assertTrue(hasattr(HikvisionSdkFrameSource, "read"))
        self.assertTrue(hasattr(HikvisionSdkFrameSource, "release"))

    def test_hikvision_channel_parse(self) -> None:
        self.assertEqual(HikvisionSdkFrameSource._parse_channel("101"), (1, 0))
        self.assertEqual(HikvisionSdkFrameSource._parse_channel("102"), (1, 1))
        self.assertEqual(HikvisionSdkFrameSource._parse_channel("2"), (2, 0))


if __name__ == "__main__":
    unittest.main()
