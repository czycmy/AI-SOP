"""H3/H4 动作识别 ROI 标定工具。

本脚本从固定相机图片或视频指定时间读取一帧，依次框选 H3、H4 附近较大的
操作区域，并将归一化坐标保存为 JSON。动作ROI应覆盖孔位、工具往复轨迹和
主要手部活动范围，但不要包含无关的大面积模具背景。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    """创建动作 ROI 标定参数。"""

    parser = argparse.ArgumentParser(description="标定 H3/H4 动作识别 ROI")
    parser.add_argument("--source", required=True, help="标定图片或视频路径。")
    parser.add_argument("--timestamp", type=float, default=0.0, help="视频取帧时间，单位秒。")
    parser.add_argument("--output", default="configs/action_rois.json", help="动作 ROI 输出配置。")
    parser.add_argument("--scale", type=float, default=1.0, help="标定窗口显示缩放比例。")
    return parser


def read_reference_frame(path: Path, timestamp_seconds: float):
    """读取图片；若输入为视频，则读取指定时间的画面。"""

    image = cv2.imread(str(path))
    if image is not None:
        return image
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取标定图片或视频：{path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频 {timestamp_seconds:.2f} 秒画面：{path}")
    return frame


def normalize_roi(roi, width: int, height: int, scale: float) -> list[float]:
    """把显示窗口中的 x/y/w/h 转成原图归一化 xyxy。"""

    x, y, roi_width, roi_height = roi
    x, y = x / scale, y / scale
    roi_width, roi_height = roi_width / scale, roi_height / scale
    return [
        round(max(0.0, x) / width, 6),
        round(max(0.0, y) / height, 6),
        round(min(width, x + roi_width) / width, 6),
        round(min(height, y + roi_height) / height, 6),
    ]


def select_valid_roi(image, name: str, scale: float):
    """要求用户为当前孔位选择一个有效动作区域。"""

    display = image if abs(scale - 1.0) < 1e-6 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    while True:
        preview = display.copy()
        cv2.putText(
            preview,
            f"Select large action ROI: {name}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        roi = cv2.selectROI(f"Action ROI: {name}", preview, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(f"Action ROI: {name}")
        if roi[2] > 0 and roi[3] > 0:
            return roi
        print(f"{name} 未选择有效 ROI，请重新框选。")


def main() -> int:
    """依次标定H3/H4并保存动作ROI配置。"""

    args = build_parser().parse_args()
    if args.scale <= 0:
        raise ValueError("--scale 必须大于0。")
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"找不到标定素材：{source}")
    frame = read_reference_frame(source, args.timestamp)
    height, width = frame.shape[:2]

    print("依次框选 H3、H4 的较大动作区域；按 Enter 或 Space 确认。")
    print("框内需要包含孔位、手部紧固范围，以及锉刀可能出现的往复轨迹。")
    action_rois = {}
    for hole_id in ("H3", "H4"):
        selected = select_valid_roi(frame, hole_id, args.scale)
        action_rois[hole_id] = normalize_roi(selected, width, height, args.scale)
        print(f"{hole_id}: {action_rois[hole_id]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source),
        "timestamp_seconds": args.timestamp,
        "action_rois": action_rois,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已输出：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
