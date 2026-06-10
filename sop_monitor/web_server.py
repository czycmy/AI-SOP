"""前端页面和后端摄像头流服务。

本模块启动一个轻量 HTTP 服务：
- `/web/` 访问现有前端页面
- `/camera.mjpg` 输出后端 OpenCV 摄像头 MJPEG 流

它用于让前端看到后端摄像头画面，避免浏览器和后端各自打开不同摄像头导致画面不同步。
"""

from __future__ import annotations

import argparse
import contextlib
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sop_monitor.camera_utils import open_camera


class CameraStream:
    """按请求打开摄像头并持续产出 JPEG 帧。"""

    def __init__(
        self,
        camera_index: int,
        width: int | None = None,
        height: int | None = None,
        jpeg_quality: int = 80,
        fps: float = 20.0,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.frame_interval = 1.0 / fps if fps > 0 else 0

    def frames(self):
        """生成 MJPEG 流需要的 JPEG 字节。"""

        import cv2

        capture = open_camera(self.camera_index, self.width, self.height)
        try:
            while True:
                started_at = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    yield encoded.tobytes()
                elapsed = time.monotonic() - started_at
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
        finally:
            capture.release()


class CameraRequestHandler(SimpleHTTPRequestHandler):
    """同时处理静态文件和摄像头流的 HTTP handler。"""

    camera_stream: CameraStream

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/web/")
            self.end_headers()
            return
        if self.path.startswith("/camera.mjpg"):
            self._handle_camera_stream()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.path.startswith("/camera.mjpg"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            return
        super().do_HEAD()

    def _handle_camera_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            for frame in self.camera_stream.frames():
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args) -> None:
        """减少 MJPEG 请求的终端噪声。"""

        if self.path.startswith("/camera.mjpg"):
            return
        super().log_message(format, *args)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="AI SOP 前端和摄像头流服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址。")
    parser.add_argument("--port", type=int, default=8000, help="监听端口。")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，内置摄像头通常是 0。")
    parser.add_argument("--width", type=int, default=None, help="采集宽度。")
    parser.add_argument("--height", type=int, default=None, help="采集高度。")
    parser.add_argument("--fps", type=float, default=20.0, help="MJPEG 输出帧率。")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG 质量，范围 1-100。")
    return parser


def make_handler(root: Path, camera_stream: CameraStream):
    """绑定静态文件根目录和摄像头流，生成 HTTP handler 类。"""

    class BoundCameraRequestHandler(CameraRequestHandler):
        pass

    BoundCameraRequestHandler.camera_stream = camera_stream

    def init(self, *handler_args, **handler_kwargs):
        CameraRequestHandler.__init__(self, *handler_args, directory=str(root), **handler_kwargs)

    BoundCameraRequestHandler.__init__ = init
    return BoundCameraRequestHandler


def main() -> int:
    """启动前端和摄像头流服务。"""

    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    camera_stream = CameraStream(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        fps=args.fps,
    )

    handler_class = make_handler(root, camera_stream)

    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    print(f"前端地址：http://{args.host}:{args.port}/web/")
    print(f"摄像头流：http://{args.host}:{args.port}/camera.mjpg")
    print("按 Ctrl+C 停止服务。")

    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
