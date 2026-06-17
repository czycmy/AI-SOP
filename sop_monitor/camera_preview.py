"""摄像头预览命令。

用于确认本地摄像头或 RTSP 网络摄像头是否能打开。
按 q 或 Esc 退出预览窗口。
"""

from __future__ import annotations

import argparse

from sop_monitor.camera_utils import add_camera_source_arguments, open_camera, resolve_camera_source


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="摄像头预览")
    add_camera_source_arguments(parser)
    parser.add_argument("--width", type=int, default=None, help="采集宽度。")
    parser.add_argument("--height", type=int, default=None, help="采集高度。")
    return parser


def main() -> int:
    """打开摄像头实时预览。"""

    import cv2

    args = build_parser().parse_args()
    camera_source = resolve_camera_source(args)
    capture = open_camera(camera_source, args.width, args.height)
    print("摄像头已打开，按 q 或 Esc 退出。")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("读取摄像头画面失败。")
            cv2.imshow("AI SOP Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
