"""训练裸锉刀锉削动作的 RGB 或方向光流分类模型。

本脚本读取“文件夹名称即类别标签”的短视频，按每个类别分别随机划分
80%训练集、10%验证集和10%测试集，并使用 torchvision 预训练 R3D-18 进行
迁移学习。输入会先裁剪 H3/H4 动作 ROI，再按连续时间采样。
训练结果可作为现有 YOLO 禁止工具外观检测的补充：YOLO 负责识别明显锉刀，
本模型负责识别黄色包装被撕掉后仍然存在的连续锉削动作。

数据目录示例：
dataset/action_videos/filing_action/*.mp4
dataset/action_videos/normal_tightening/*.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sop_monitor.action_recognition import (
    build_action_model,
    frames_to_clip,
    load_action_rois,
    motion_energy,
    resize_roi_frames,
    select_active_roi,
)


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    """创建训练参数。"""

    parser = argparse.ArgumentParser(description="训练锉削动作视频分类模型")
    parser.add_argument("--data", default="dataset/action_videos", help="类别文件夹所在目录。")
    parser.add_argument("--output", default="runs/filing_action_r3d18", help="训练输出目录。")
    parser.add_argument("--epochs", type=int, default=30, help="最大训练轮数。")
    parser.add_argument("--batch-size", type=int, default=2, help="批大小，显存充足时可改为 4。")
    parser.add_argument("--frames", type=int, default=24, help="每个样本连续采样的帧数。")
    parser.add_argument("--sample-fps", type=float, default=10.0, help="动作模型的目标采样帧率。")
    parser.add_argument("--size", type=int, default=160, help="ROI 裁剪后的输入画面边长。")
    parser.add_argument("--action-rois", default="configs/action_rois.json", help="H3/H4 动作 ROI 配置。")
    parser.add_argument("--no-roi", action="store_true", help="禁用动作 ROI，仅用于对照实验。")
    parser.add_argument(
        "--input-mode",
        choices=("rgb", "flow", "motion"),
        default="rgb",
        help="RGB外观、方向光流或兼容旧权重的绝对帧差输入。",
    )
    parser.add_argument(
        "--label-mode",
        choices=("three-class", "binary"),
        default="three-class",
        help="默认分别学习锉削、正常紧固和其他动作；binary兼容旧训练方式。",
    )
    parser.add_argument("--min-duration", type=float, default=1.8, help="短于该秒数的视频不参与训练。")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="前几轮只训练分类头。")
    parser.add_argument("--head-lr", type=float, default=3e-4, help="冻结骨干阶段的分类头学习率。")
    parser.add_argument("--lr", type=float, default=3e-5, help="解冻骨干后的 AdamW 学习率。")
    parser.add_argument("--dropout", type=float, default=0.4, help="分类头 Dropout 比例。")
    parser.add_argument("--workers", type=int, default=0, help="Windows 首次运行建议保持 0。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--patience", type=int, default=8, help="验证集连续无提升后提前停止。")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="auto 会依次选择 CUDA、MPS、CPU。",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 Kinetics-400 预训练权重。")
    return parser


def choose_device(requested: str) -> torch.device:
    """优先使用 Windows NVIDIA CUDA，其次使用 Apple MPS。"""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_videos(
    root: Path,
    min_duration: float,
    label_mode: str,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """扫描动作目录，并按三分类或兼容的二分类方式生成标签。"""

    if not root.is_dir():
        raise FileNotFoundError(f"找不到数据目录：{root}")
    source_classes = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and path.name != "uncertain"
    )
    if "filing_action" not in source_classes or len(source_classes) < 2:
        raise ValueError("至少需要 filing_action 和一个非锉削动作目录。")

    samples: list[tuple[Path, str]] = []
    source_counts = {}
    for source_class in source_classes:
        label = (
            "filing_action"
            if source_class == "filing_action"
            else "non_filing_action"
            if label_mode == "binary"
            else source_class
        )
        accepted_count = 0
        for path in sorted((root / source_class).rglob("*")):
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            capture = cv2.VideoCapture(str(path))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            ok, _ = capture.read()
            capture.release()
            duration = frame_count / fps if fps > 0 else 0.0
            if ok and frame_count > 0 and duration >= min_duration:
                samples.append((path, label))
                accepted_count += 1
            elif ok and duration < min_duration:
                print(f"跳过过短视频（{duration:.2f}s < {min_duration:.2f}s）：{path}")
            else:
                print(f"跳过损坏或空视频：{path}")
        source_counts[source_class] = accepted_count
    if not samples:
        raise ValueError("没有找到可用视频。")
    print("来源目录：" + "，".join(f"{name}={count}" for name, count in source_counts.items()))
    labels = {label for _, label in samples}
    class_names = ["filing_action", *sorted(labels - {"filing_action"})]
    return samples, class_names


def stratified_split(
    samples: list[tuple[Path, str]], seed: int
) -> dict[str, list[tuple[Path, str]]]:
    """每个类别独立随机划分8:1:1，保证三个集合类别比例均衡。"""

    grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name, class_samples in grouped.items():
        rng.shuffle(class_samples)
        count = len(class_samples)
        val_count = max(1, round(count * 0.1))
        test_count = max(1, round(count * 0.1))
        train_count = count - val_count - test_count
        if train_count < 1:
            raise ValueError(f"类别 {class_name} 至少需要4段有效视频。")
        splits["train"].extend(class_samples[:train_count])
        splits["val"].extend(class_samples[train_count:train_count + val_count])
        splits["test"].extend(class_samples[train_count + val_count:])
    for split_samples in splits.values():
        rng.shuffle(split_samples)
    return splits


def save_manifest(output_dir: Path, splits: dict[str, list[tuple[Path, str]]]) -> None:
    """保存本次固定划分，便于复现实验并检查数据泄漏。"""

    with (output_dir / "dataset_split.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "label", "video"])
        for split_name, samples in splits.items():
            for path, label in samples:
                writer.writerow([split_name, label, str(path.resolve())])


class ActionVideoDataset(Dataset):
    """读取连续动作窗口，自动选择运动更明显的H3或H4。"""

    def __init__(
        self,
        samples: list[tuple[Path, str]],
        class_to_index: dict[str, int],
        frame_count: int,
        image_size: int,
        sample_fps: float,
        action_rois: dict[str, tuple[float, float, float, float]],
        input_mode: str,
        training: bool,
    ):
        self.samples = samples
        self.class_to_index = class_to_index
        self.frame_count = frame_count
        self.image_size = image_size
        self.sample_fps = sample_fps
        self.action_rois = action_rois
        self.input_mode = input_mode
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, class_name = self.samples[index]
        frames = self._read_action_frames(path)
        if self.training:
            # 同一窗口使用一致的轻量光照增强，适配现场亮度小幅波动。
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-8.0, 8.0)
            frames = [cv2.convertScaleAbs(frame, alpha=alpha, beta=beta) for frame in frames]
            # 锉削和紧固标签在时间倒放后不变，可避免模型死记单一转动方向。
            if random.random() < 0.5:
                frames.reverse()
        clip = frames_to_clip(frames, self.input_mode)
        return clip, torch.tensor(self.class_to_index[class_name], dtype=torch.long)

    def _read_action_frames(self, path: Path) -> list[torch.Tensor]:
        """顺序解码后，从一段连续时间范围采样动作帧。

        H.264 视频通常只能从关键帧可靠解码。逐帧调用 CAP_PROP_POS_FRAMES
        会频繁跳到非关键帧，尤其是直接裁切生成的短视频容易产生大量解码警告。
        这里先顺序读取全部有效帧，既减少警告，也避免损坏帧污染训练样本。
        """

        capture = cv2.VideoCapture(str(path))
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        decoded_frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames.append(frame)
        capture.release()
        if not decoded_frames:
            raise RuntimeError(f"无法读取视频帧：{path}")

        effective_sample_fps = self.sample_fps
        if self.training:
            # 在不破坏动作结构的范围内模拟约±15%的操作速度差异。
            effective_sample_fps *= random.uniform(0.85, 1.15)
        required_span = max(
            self.frame_count,
            round((self.frame_count - 1) * source_fps / effective_sample_fps) + 1,
        )
        span = min(len(decoded_frames), required_span)
        max_start = len(decoded_frames) - span
        if self.training and max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = self._highest_motion_start(decoded_frames, span, max_start)
        positions = torch.linspace(start, start + span - 1, self.frame_count).round().int().tolist()
        sampled_frames = [decoded_frames[position] for position in positions]
        _, active_frames, _ = select_active_roi(
            sampled_frames,
            self.action_rois,
            self.image_size,
        )
        return active_frames

    def _highest_motion_start(self, frames: list, span: int, max_start: int) -> int:
        """验证/测试时选择运动最明显的时间窗口，避免长视频抽到等待阶段。"""

        if max_start <= 0:
            return 0
        stride = max(1, span // 3)
        starts = list(range(0, max_start + 1, stride))
        if starts[-1] != max_start:
            starts.append(max_start)
        best_start = 0
        best_energy = -1.0
        for start in starts:
            positions = torch.linspace(start, start + span - 1, min(8, self.frame_count)).round().int().tolist()
            candidate = [frames[position] for position in positions]
            energy = max(
                motion_energy(resize_roi_frames(candidate, roi, 64))
                for roi in self.action_rois.values()
            )
            if energy > best_energy:
                best_energy = energy
                best_start = start
        return best_start


def classification_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
    filing_index: int,
) -> dict[str, float | int]:
    """计算报警类别的准确率、精确率、召回率、F1和混淆矩阵。"""

    predictions = [value >= threshold for value in probabilities]
    targets = [label == filing_index for label in labels]
    tp = sum(pred and target for pred, target in zip(predictions, targets))
    fp = sum(pred and not target for pred, target in zip(predictions, targets))
    fn = sum(not pred and target for pred, target in zip(predictions, targets))
    tn = sum(not pred and not target for pred, target in zip(predictions, targets))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrate_threshold(
    labels: list[int],
    probabilities: list[float],
    filing_index: int,
) -> tuple[float, dict[str, float | int]]:
    """在验证集上选择F1最高的锉削报警阈值。"""

    best_threshold = 0.5
    best_metrics = classification_metrics(labels, probabilities, best_threshold, filing_index)
    for step in range(20, 81):
        threshold = step / 100
        metrics = classification_metrics(labels, probabilities, threshold, filing_index)
        rank = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            -abs(threshold - 0.5),
        )
        best_rank = (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
            -abs(best_threshold - 0.5),
        )
        if rank > best_rank:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def run_epoch(model, loader, criterion, device, filing_index: int, optimizer=None):
    """执行一轮训练或评估，并返回损失、标签和锉削概率。"""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    all_labels: list[int] = []
    filing_probabilities: list[float] = []
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(clips)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        filing_probabilities.extend(logits.softmax(dim=1)[:, filing_index].detach().cpu().tolist())
    metrics = classification_metrics(all_labels, filing_probabilities, 0.5, filing_index)
    return total_loss / total_count, metrics, all_labels, filing_probabilities


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """冻结或解冻R3D骨干，分类头始终参与训练。"""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("fc.")


def main() -> int:
    """完成数据划分、迁移学习和最终测试。"""

    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, class_names = discover_videos(
        Path(args.data),
        args.min_duration,
        args.label_mode,
    )
    splits = stratified_split(samples, args.seed)
    action_rois, mask_top = load_action_rois(Path(args.action_rois), args.no_roi)
    save_manifest(output_dir, splits)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    filing_index = class_to_index["filing_action"]
    device = choose_device(args.device)
    print(
        f"设备：{device}；类别：{class_to_index}；动作ROI：{action_rois}；"
        f"排除顶部：{mask_top:.3f}；输入：{args.input_mode}；标签：{args.label_mode}"
    )
    print("数据量：" + "，".join(f"{name}={len(items)}" for name, items in splits.items()))

    loaders = {}
    for split_name, split_samples in splits.items():
        dataset = ActionVideoDataset(
            split_samples,
            class_to_index,
            args.frames,
            args.size,
            args.sample_fps,
            action_rois,
            args.input_mode,
            training=split_name == "train",
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split_name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

    model = build_action_model(
        len(class_names),
        pretrained=not args.no_pretrained,
        head_dropout=args.dropout,
    )
    model.to(device)
    train_counts = {
        name: sum(label == name for _, label in splits["train"])
        for name in class_names
    }
    class_weights = torch.tensor(
        [len(splits["train"]) / (len(class_names) * train_counts[name]) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    set_backbone_trainable(model, trainable=args.freeze_epochs <= 0)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr if args.freeze_epochs <= 0 else args.head_lr,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_threshold = 0.5
    stale_epochs = 0
    history = []
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            print(f"Epoch {epoch:03d}：已解冻R3D骨干，学习率切换为 {args.lr:g}")
        train_loss, train_metrics, _, _ = run_epoch(
            model, loaders["train"], criterion, device, filing_index, optimizer
        )
        val_loss, _, val_labels, val_probabilities = run_epoch(
            model, loaders["val"], criterion, device, filing_index
        )
        recommended_threshold, val_metrics = calibrate_threshold(
            val_labels, val_probabilities, filing_index
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_filing_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_filing_precision": val_metrics["precision"],
            "val_filing_recall": val_metrics["recall"],
            "val_filing_f1": val_metrics["f1"],
            "recommended_threshold": recommended_threshold,
        })
        common_checkpoint = {
            "model_name": "r3d_18",
            "model_state_dict": model.state_dict(),
            "class_to_index": class_to_index,
            "frames": args.frames,
            "sample_fps": args.sample_fps,
            "image_size": args.size,
            "action_rois": action_rois,
            "mask_top": mask_top,
            "input_mode": args.input_mode,
            "label_mode": args.label_mode,
            "head_dropout": args.dropout,
            "epoch": epoch,
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_loss,
            "val_filing_precision": val_metrics["precision"],
            "val_filing_recall": val_metrics["recall"],
            "val_filing_f1": val_metrics["f1"],
            "recommended_threshold": recommended_threshold,
        }
        torch.save(common_checkpoint, last_checkpoint_path)
        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.4f} acc={train_metrics['accuracy']:.3f} "
            f"| val loss={val_loss:.4f} acc={val_metrics['accuracy']:.3f} "
            f"filing P/R/F1={val_metrics['precision']:.3f}/{val_metrics['recall']:.3f}/"
            f"{val_metrics['f1']:.3f} threshold={recommended_threshold:.2f}"
        )
        # 禁止工具更关注锉削F1；F1相同时继续比较验证损失。
        is_better = (
            val_metrics["f1"] > best_f1
            or (
                abs(val_metrics["f1"] - best_f1) < 1e-12
                and val_loss < best_val_loss
            )
        )
        if is_better:
            best_f1 = float(val_metrics["f1"])
            best_val_loss = val_loss
            best_epoch = epoch
            best_threshold = recommended_threshold
            stale_epochs = 0
            torch.save({
                **common_checkpoint,
                "best_epoch": best_epoch,
                "best_val_filing_f1": best_f1,
                "best_val_loss": best_val_loss,
                "recommended_threshold": best_threshold,
            }, checkpoint_path)
            print(
                f"  已更新 best.pt：epoch={best_epoch}，"
                f"filing_f1={best_f1:.3f}，val_loss={best_val_loss:.6f}"
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集连续 {args.patience} 轮无提升，提前停止。")
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, _, test_labels, test_probabilities = run_epoch(
        model, loaders["test"], criterion, device, filing_index
    )
    test_metrics = classification_metrics(
        test_labels,
        test_probabilities,
        best_threshold,
        filing_index,
    )
    print(
        f"最佳轮次：{best_epoch}；验证锉削F1：{best_f1:.3f}；"
        f"验证损失：{best_val_loss:.6f}；报警阈值：{best_threshold:.2f}"
    )
    print(
        f"随机测试集：loss={test_loss:.4f}，accuracy={test_metrics['accuracy']:.3f}，"
        f"filing P/R/F1={test_metrics['precision']:.3f}/{test_metrics['recall']:.3f}/"
        f"{test_metrics['f1']:.3f}，TP/FP/FN/TN="
        f"{test_metrics['tp']}/{test_metrics['fp']}/{test_metrics['fn']}/{test_metrics['tn']}"
    )
    print(f"最佳权重：{checkpoint_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
