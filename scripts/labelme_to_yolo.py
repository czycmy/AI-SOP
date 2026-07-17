"""把 LabelMe 标注转换为 YOLO 检测数据集。

输入目录中可以同时包含图片、LabelMe JSON 和未标注图片。已标注图片会转换为
YOLO txt 标签；未标注图片会生成空 txt，作为负样本参与训练。

当前项目使用三类目标：
installed_part 表示孔位内已放入/已安装的零件；
l_tool_visible 表示可见的 L 型紧固工具部分，不包含手；
forbidden_tool 表示锉刀等禁止出现的工具，不包含手。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["installed_part", "l_tool_visible", "forbidden_tool"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数。"""

    parser = argparse.ArgumentParser(description="LabelMe 转 YOLO 数据集")
    parser.add_argument("--input", default="dataset/frames_data", help="包含图片和 LabelMe JSON 的目录。")
    parser.add_argument("--output", default="dataset/yolo_sop_objects", help="输出 YOLO 数据集目录。")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子。")
    return parser


def list_images(input_dir: Path) -> list[Path]:
    """列出输入目录下的图片。"""

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("_")
    )


def convert_labelme_json(json_path: Path) -> list[str]:
    """把单个 LabelMe JSON 转成 YOLO 标签行。"""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_width = float(data["imageWidth"])
    image_height = float(data["imageHeight"])
    lines: list[str] = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in CLASS_TO_ID:
            raise ValueError(f"{json_path} 出现未知标签：{label}")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"{json_path} 只支持 rectangle，当前是：{shape.get('shape_type')}")
        points = shape.get("points", [])
        if len(points) != 2:
            raise ValueError(f"{json_path} rectangle 点数量异常：{len(points)}")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = max(0.0, min(xs)), min(image_width, max(xs))
        y1, y2 = max(0.0, min(ys)), min(image_height, max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        x_center = ((x1 + x2) / 2.0) / image_width
        y_center = ((y1 + y2) / 2.0) / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height
        lines.append(
            f"{CLASS_TO_ID[label]} "
            f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return lines


def split_images(images: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """按比例划分训练集和验证集。"""

    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围内。")
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 and val_ratio > 0 else 0
    val_images = sorted(shuffled[:val_count])
    train_images = sorted(shuffled[val_count:])
    return train_images, val_images


def write_subset(images: list[Path], subset: str, input_dir: Path, output_dir: Path) -> tuple[int, int]:
    """写入某个子集的图片和标签。"""

    image_out = output_dir / "images" / subset
    label_out = output_dir / "labels" / subset
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    labeled_count = 0
    box_count = 0
    for image_path in images:
        target_image = image_out / image_path.name
        shutil.copy2(image_path, target_image)
        json_path = input_dir / f"{image_path.stem}.json"
        lines = convert_labelme_json(json_path) if json_path.exists() else []
        if lines:
            labeled_count += 1
            box_count += len(lines)
        (label_out / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return labeled_count, box_count


def write_data_yaml(output_dir: Path) -> None:
    """写入 YOLO data.yaml。"""

    content = "\n".join([
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
        "  0: installed_part",
        "  1: l_tool_visible",
        "  2: forbidden_tool",
        "",
    ])
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    """执行转换。"""

    args = build_parser().parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入目录：{input_dir}")

    images = list_images(input_dir)
    if not images:
        raise ValueError(f"{input_dir} 中没有图片。")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    train_images, val_images = split_images(images, args.val_ratio, args.seed)

    train_labeled, train_boxes = write_subset(train_images, "train", input_dir, output_dir)
    val_labeled, val_boxes = write_subset(val_images, "val", input_dir, output_dir)
    write_data_yaml(output_dir)

    json_count = len(list(input_dir.glob("*.json")))
    print(f"input_images: {len(images)}")
    print(f"labelme_json: {json_count}")
    print(f"train_images: {len(train_images)}, labeled: {train_labeled}, boxes: {train_boxes}")
    print(f"val_images: {len(val_images)}, labeled: {val_labeled}, boxes: {val_boxes}")
    print(f"output: {output_dir}")
    print(f"data_yaml: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
