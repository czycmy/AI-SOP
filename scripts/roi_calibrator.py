"""孔位 ROI 标定工具。

打开一张固定相机位图片，按 SOP 配置中的步骤逐个框选孔位判断区域，
并输出带归一化 ROI 坐标的新配置文件。默认不显示旧 ROI，避免重新标定时
被旧框干扰；按 c 取消当前框后会继续要求重新框选当前孔位。
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数。"""

    parser = argparse.ArgumentParser(description="孔位 ROI 标定工具")
    parser.add_argument("--image", required=True, help="用于标定的固定相机图片。")
    parser.add_argument("--config", default="configs/sample_sop.json", help="原始 SOP 配置。")
    parser.add_argument("--output", default="configs/calibrated_sop.json", help="输出的新 SOP 配置。")
    parser.add_argument("--scale", type=float, default=1.0, help="显示缩放比例，图片太大时可用 0.75。")
    parser.add_argument("--show-existing", action="store_true", help="显示配置中已有 ROI，仅用于对照检查。")
    parser.add_argument("--allow-skip", action="store_true", help="允许按 c 跳过当前孔位并保留原 ROI。")
    parser.add_argument("--region-only", action="store_true", help="只标定区域总 ROI，保留已有孔位 ROI。")
    return parser


def normalize_roi(
    roi: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[float]:
    """把 OpenCV 像素 ROI x/y/w/h 转成归一化 xyxy。"""

    x, y, width, height = roi
    x1 = max(0, x) / image_width
    y1 = max(0, y) / image_height
    x2 = min(image_width, x + width) / image_width
    y2 = min(image_height, y + height) / image_height
    return [round(value, 6) for value in (x1, y1, x2, y2)]


def denormalize_roi(
    roi: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """把归一化 xyxy ROI 转成像素 xyxy。"""

    x1, y1, x2, y2 = roi
    return (
        int(round(x1 * image_width)),
        int(round(y1 * image_height)),
        int(round(x2 * image_width)),
        int(round(y2 * image_height)),
    )


def draw_existing_rois(image, config: dict) -> None:
    """在图片上绘制配置中已有的 ROI，方便对照。"""

    height, width = image.shape[:2]
    for region in config.get("regions", []):
        region_id = region.get("region_id", "")
        roi = region.get("roi")
        if roi:
            x1, y1, x2, y2 = denormalize_roi(roi, width, height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(image, f"{region_id} AREA", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
        for step in region.get("steps", []):
            roi = step.get("roi")
            if not roi:
                continue
            x1, y1, x2, y2 = denormalize_roi(roi, width, height)
            label = f"{region_id}-{step.get('hole_id', '')}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 180, 255), 2)
            cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 2)


def maybe_scale_image(image, scale: float):
    """按比例缩放显示图片，并返回缩放后的图片和实际比例。"""

    if scale <= 0:
        raise ValueError("--scale 必须大于 0。")
    if abs(scale - 1.0) < 1e-6:
        return image, 1.0
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def main() -> int:
    """执行 ROI 标定。"""

    args = build_parser().parse_args()
    image_path = Path(args.image)
    config_path = Path(args.config)
    output_path = Path(args.output)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置：{config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    updated = deepcopy(config)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")
    image_height, image_width = image.shape[:2]

    print("ROI 标定说明：")
    print("- 鼠标拖拽框选当前提示的孔位 ROI。")
    print("- 按 Enter/Space 确认当前框。")
    print("- 按 c 取消当前框后，会重新选择当前孔位。")
    print("- 如果确实想跳过当前孔位，启动时加 --allow-skip。")
    print("- 框选窗口需要有焦点。")
    print()

    for region in updated.get("regions", []):
        region_id = region.get("region_id", "")
        while True:
            preview = image.copy()
            if args.show_existing:
                draw_existing_rois(preview, updated)
            cv2.putText(
                preview,
                f"Select monitor area: {region_id}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
            display, scale = maybe_scale_image(preview, args.scale)
            roi = cv2.selectROI(f"Monitor area: {region_id}", display, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(f"Monitor area: {region_id}")
            x, y, width, height = [int(value) for value in roi]
            if width > 0 and height > 0:
                original_roi = (
                    int(round(x / scale)),
                    int(round(y / scale)),
                    int(round(width / scale)),
                    int(round(height / scale)),
                )
                region["roi"] = normalize_roi(original_roi, image_width, image_height)
                print(f"{region_id} area: {region['roi']}")
                break
            if args.allow_skip:
                print(f"跳过 {region_id} 区域总 ROI，保留原 ROI：{region.get('roi')}")
                break
            print(f"{region_id} 未选择有效区域总 ROI，请重新框选。")

        if args.region_only:
            continue
        for step in region.get("steps", []):
            hole_id = step.get("hole_id", "")
            title = f"ROI: {region_id}-{hole_id}"
            while True:
                preview = image.copy()
                if args.show_existing:
                    draw_existing_rois(preview, updated)
                cv2.putText(
                    preview,
                    f"Select {region_id}-{hole_id}",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                display, scale = maybe_scale_image(preview, args.scale)
                roi = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow(title)
                x, y, width, height = [int(value) for value in roi]
                if width > 0 and height > 0:
                    break
                if args.allow_skip:
                    print(f"跳过 {region_id}-{hole_id}，保留原 ROI：{step.get('roi')}")
                    break
                print(f"{region_id}-{hole_id} 未选择有效 ROI，请重新框选。")
            if width <= 0 or height <= 0:
                continue
            original_roi = (
                int(round(x / scale)),
                int(round(y / scale)),
                int(round(width / scale)),
                int(round(height / scale)),
            )
            step["roi"] = normalize_roi(original_roi, image_width, image_height)
            print(f"{region_id}-{hole_id}: {step['roi']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"已输出：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
