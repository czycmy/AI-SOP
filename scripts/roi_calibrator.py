"""孔位 ROI 标定工具。

从固定相机图片或视频指定时刻读取画面，按 SOP 配置框选区域总 ROI 或孔位 ROI，
并输出带归一化坐标的新配置文件。支持只重画指定孔位，其余 ROI 保持不变，
适合为存在轻微相机偏移的视频建立独立配置。
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
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="用于标定的固定相机图片。")
    source_group.add_argument("--video", help="用于标定的视频文件。")
    parser.add_argument("--timestamp", type=float, default=0.0, help="视频取帧时间，单位秒。")
    parser.add_argument("--config", default="configs/sample_sop.json", help="原始 SOP 配置。")
    parser.add_argument("--output", default="configs/calibrated_sop.json", help="输出的新 SOP 配置。")
    parser.add_argument("--scale", type=float, default=1.0, help="显示缩放比例，图片太大时可用 0.75。")
    parser.add_argument("--show-existing", action="store_true", help="显示配置中已有 ROI，仅用于对照检查。")
    parser.add_argument("--allow-skip", action="store_true", help="允许按 c 跳过当前孔位并保留原 ROI。")
    parser.add_argument("--region-only", action="store_true", help="只标定区域总 ROI，保留已有孔位 ROI。")
    parser.add_argument(
        "--only-hole",
        nargs="+",
        metavar="孔位",
        help="只重画指定孔位，例如 --only-hole H2；多个孔位可写 H2 H3。",
    )
    return parser


def read_reference_frame(
    image_path: str | None,
    video_path: str | None,
    timestamp_seconds: float,
):
    """读取标定图片，或读取视频指定时间点的一帧。"""

    if timestamp_seconds < 0:
        raise ValueError("--timestamp 不能小于 0。")

    if image_path:
        source = Path(image_path)
        if not source.exists():
            raise FileNotFoundError(f"找不到图片：{source}")
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"无法读取图片：{source}")
        return image

    source = Path(video_path or "")
    if not source.exists():
        raise FileNotFoundError(f"找不到视频：{source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开视频：{source}")
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频 {timestamp_seconds:.2f} 秒画面：{source}")
    return frame


def resolve_hole_selectors(config: dict, selectors: list[str] | None) -> set[tuple[str, str]] | None:
    """把 H2 或 R1-H2 形式的参数解析为配置中的具体孔位。"""

    if not selectors:
        return None

    known: list[tuple[str, str]] = []
    for region in config.get("regions", []):
        region_id = str(region.get("region_id", ""))
        for step in region.get("steps", []):
            known.append((region_id, str(step.get("hole_id", ""))))

    selected: set[tuple[str, str]] = set()
    for raw_selector in selectors:
        for selector in (part.strip() for part in raw_selector.split(",")):
            if not selector:
                continue
            selector_upper = selector.upper()
            qualified_matches = [
                item for item in known if f"{item[0]}-{item[1]}".upper() == selector_upper
            ]
            matches = qualified_matches or [item for item in known if item[1].upper() == selector_upper]
            if not matches:
                available = ", ".join(f"{region_id}-{hole_id}" for region_id, hole_id in known)
                raise ValueError(f"配置中不存在孔位 {selector}；可选孔位：{available}")
            if len(matches) > 1:
                available = ", ".join(f"{region_id}-{hole_id}" for region_id, hole_id in matches)
                raise ValueError(f"孔位 {selector} 不唯一，请使用完整名称：{available}")
            selected.add(matches[0])

    if not selected:
        raise ValueError("--only-hole 至少需要一个有效孔位。")
    return selected


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

    parser = build_parser()
    args = parser.parse_args()
    if args.region_only and args.only_hole:
        parser.error("--region-only 与 --only-hole 不能同时使用。")

    config_path = Path(args.config)
    output_path = Path(args.output)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置：{config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    updated = deepcopy(config)
    selected_holes = resolve_hole_selectors(updated, args.only_hole)
    image = read_reference_frame(args.image, args.video, args.timestamp)
    image_height, image_width = image.shape[:2]

    print("ROI 标定说明：")
    print("- 鼠标拖拽框选当前提示的孔位 ROI。")
    print("- 按 Enter/Space 确认当前框。")
    print("- 按 c 取消当前框后，会重新选择当前孔位。")
    print("- 如果确实想跳过当前孔位，启动时加 --allow-skip。")
    print("- 框选窗口需要有焦点。")
    if selected_holes:
        selected_names = ", ".join(f"{region_id}-{hole_id}" for region_id, hole_id in sorted(selected_holes))
        print(f"- 本次只重画：{selected_names}；区域总 ROI 和其他孔位保持不变。")
    print()

    for region in updated.get("regions", []):
        region_id = region.get("region_id", "")
        if selected_holes is None:
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
            if selected_holes is not None and (region_id, hole_id) not in selected_holes:
                continue
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
