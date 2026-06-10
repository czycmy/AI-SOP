"""前端摄像头流服务测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from sop_monitor.web_server import CameraStream, make_handler


class WebServerTest(unittest.TestCase):
    def test_make_handler_binds_camera_stream(self) -> None:
        stream = CameraStream(camera_index=0)
        handler = make_handler(Path("."), stream)
        self.assertIs(handler.camera_stream, stream)


if __name__ == "__main__":
    unittest.main()

