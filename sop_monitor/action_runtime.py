"""客户端实时连续动作双模型推理器。

本模块加载 RGB 外观模型和方向光流模型，按时间戳缓存 H3/H4 动作区域，
每隔固定时间执行一次双路推理并融合概率。连续窗口由事件状态机去重，确保
一次持续锉削只触发一次客户端异常记录。动作 ROI 只在后台裁剪，不绘制到画面。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from sop_monitor.action_recognition import (
    MultiRoiActionEventStateMachine,
    build_action_model,
    crop_normalized_roi,
    frames_to_clip,
    fuse_action_probabilities,
)


@dataclass(frozen=True)
class LoadedActionModel:
    """动作权重及训练阶段保存的预处理参数。"""

    model: torch.nn.Module
    class_to_index: dict[str, int]
    sample_count: int
    image_size: int
    sample_fps: float
    action_rois: dict[str, tuple[float, float, float, float]]
    input_mode: str
    recommended_threshold: float

    @property
    def window_ms(self) -> int:
        """返回一次推理覆盖的毫秒数。"""

        return max(1, round(self.sample_count / self.sample_fps * 1000))


@dataclass(frozen=True)
class ActionFusionResult:
    """客户端一次双路动作推理结果。"""

    timestamp_ms: int
    rgb_probability: float
    flow_probability: float
    fused_probability: float
    active_roi: str
    alarm_active: bool
    event_started: bool
    event_ended: bool
    event_count: int


def choose_action_device(requested: str) -> torch.device:
    """优先选择 CUDA，其次选择 Apple MPS，最后使用 CPU。"""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_action_model(path: str | Path, device: torch.device) -> LoadedActionModel:
    """加载动作权重，并兼容早期缺少输入配置的检查点。"""

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到动作模型：{model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_to_index = checkpoint.get("class_to_index")
    if not isinstance(class_to_index, dict) or "filing_action" not in class_to_index:
        raise ValueError(f"动作模型类别中缺少 filing_action：{model_path}")

    model = build_action_model(
        len(class_to_index),
        pretrained=False,
        head_dropout=float(checkpoint.get("head_dropout") or 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    raw_rois = checkpoint.get("action_rois")
    if not raw_rois:
        raise ValueError(f"动作模型没有保存H3/H4 ROI，请重新训练：{model_path}")
    action_rois = {
        str(name): tuple(float(value) for value in roi)
        for name, roi in raw_rois.items()
    }
    if set(action_rois) != {"H3", "H4"}:
        raise ValueError(f"动作模型必须同时包含H3和H4 ROI：{model_path}")

    sample_count = int(checkpoint["frames"])
    sample_fps = float(checkpoint.get("sample_fps") or sample_count / 2.5)
    return LoadedActionModel(
        model=model,
        class_to_index={str(name): int(index) for name, index in class_to_index.items()},
        sample_count=sample_count,
        image_size=int(checkpoint["image_size"]),
        sample_fps=sample_fps,
        action_rois=action_rois,
        input_mode=str(checkpoint.get("input_mode") or "rgb"),
        recommended_threshold=float(checkpoint.get("recommended_threshold") or 0.5),
    )


def action_rois_match(
    first: dict[str, tuple[float, float, float, float]],
    second: dict[str, tuple[float, float, float, float]],
) -> bool:
    """检查两个动作模型是否使用完全相同的ROI。"""

    if first.keys() != second.keys():
        return False
    return all(
        all(abs(left - right) < 1e-6 for left, right in zip(first[name], second[name]))
        for name in first
    )


def sample_timed_frames(
    items: deque[tuple[int, np.ndarray]],
    end_timestamp_ms: int,
    window_ms: int,
    sample_count: int,
    image_size: int,
) -> list[np.ndarray]:
    """按时间均匀采样最近一个动作窗口，适配视频和实时摄像头的不规则帧率。"""

    if not items:
        raise ValueError("动作帧缓存为空。")
    timestamps = np.asarray([timestamp for timestamp, _ in items], dtype=np.int64)
    targets = np.linspace(
        end_timestamp_ms - window_ms,
        end_timestamp_ms,
        sample_count,
    )
    positions = np.searchsorted(timestamps, targets, side="left")
    selected: list[np.ndarray] = []
    for target, position in zip(targets, positions):
        right = min(int(position), len(items) - 1)
        left = max(0, right - 1)
        index = (
            left
            if abs(int(timestamps[left]) - target) <= abs(int(timestamps[right]) - target)
            else right
        )
        selected.append(
            cv2.resize(
                items[index][1],
                (image_size, image_size),
                interpolation=cv2.INTER_AREA,
            )
        )
    return selected


class ActionFusionMonitor:
    """维护实时帧缓存并执行 RGB + 方向光流融合判定。"""

    def __init__(
        self,
        rgb_model_path: str | Path,
        flow_model_path: str | Path,
        device: str = "auto",
        rgb_weight: float = 0.7,
        threshold: float = 0.5,
        clear_threshold: float = 0.35,
        interval_seconds: float = 0.2,
        vote_window: int = 4,
        trigger_votes: int = 3,
        clear_windows: int = 4,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("动作推理间隔必须大于0秒。")
        self.device = choose_action_device(device)
        self.rgb_model = load_action_model(rgb_model_path, self.device)
        self.flow_model = load_action_model(flow_model_path, self.device)
        if self.rgb_model.input_mode != "rgb":
            raise ValueError(
                f"RGB动作权重的输入模式应为rgb，实际为{self.rgb_model.input_mode}。"
            )
        if self.flow_model.input_mode != "flow":
            raise ValueError(
                f"光流动作权重的输入模式应为flow，实际为{self.flow_model.input_mode}。"
            )
        if not action_rois_match(
            self.rgb_model.action_rois,
            self.flow_model.action_rois,
        ):
            raise ValueError("RGB与方向光流模型的H3/H4动作ROI不一致。")

        self.action_rois = self.rgb_model.action_rois
        self.rgb_weight = rgb_weight
        # 提前校验融合权重，避免运行到第一段动作时才报错。
        fuse_action_probabilities(0.0, 0.0, rgb_weight)
        self.interval_ms = max(1, round(interval_seconds * 1000))
        self.max_window_ms = max(
            self.rgb_model.window_ms,
            self.flow_model.window_ms,
        )
        self.frame_buffers: dict[str, deque[tuple[int, np.ndarray]]] = {
            name: deque()
            for name in self.action_rois
        }
        self.event_state = MultiRoiActionEventStateMachine(
            roi_names=set(self.action_rois),
            trigger_threshold=threshold,
            clear_threshold=clear_threshold,
            vote_window=vote_window,
            trigger_votes=trigger_votes,
            clear_windows=clear_windows,
        )
        self._last_frame_timestamp_ms: int | None = None
        self._last_prediction_timestamp_ms: int | None = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> ActionFusionResult | None:
        """接收最新BGR帧；达到窗口和间隔要求时返回一次融合结果。"""

        timestamp_ms = int(timestamp_ms)
        if self._last_frame_timestamp_ms == timestamp_ms:
            return None
        self._last_frame_timestamp_ms = timestamp_ms

        cutoff_ms = timestamp_ms - self.max_window_ms - self.interval_ms
        for name, roi in self.action_rois.items():
            buffer = self.frame_buffers[name]
            buffer.append((timestamp_ms, crop_normalized_roi(frame, roi).copy()))
            while len(buffer) > 2 and buffer[1][0] < cutoff_ms:
                buffer.popleft()

        earliest_timestamp = max(buffer[0][0] for buffer in self.frame_buffers.values())
        if timestamp_ms - earliest_timestamp < self.max_window_ms:
            return None
        if (
            self._last_prediction_timestamp_ms is not None
            and timestamp_ms - self._last_prediction_timestamp_ms < self.interval_ms
        ):
            return None
        self._last_prediction_timestamp_ms = timestamp_ms

        rgb_probabilities = self._predict(self.rgb_model, timestamp_ms)
        flow_probabilities = self._predict(self.flow_model, timestamp_ms)
        fused_probabilities = {
            name: fuse_action_probabilities(
                rgb_probabilities[name],
                flow_probabilities[name],
                self.rgb_weight,
            )
            for name in self.action_rois
        }
        update = self.event_state.update(fused_probabilities, timestamp_ms / 1000)
        candidate_rois = self.event_state.active_rois or set(fused_probabilities)
        active_roi = max(candidate_rois, key=fused_probabilities.get)
        fused_probability = fused_probabilities[active_roi]
        return ActionFusionResult(
            timestamp_ms=timestamp_ms,
            rgb_probability=rgb_probabilities[active_roi],
            flow_probability=flow_probabilities[active_roi],
            fused_probability=fused_probability,
            active_roi=active_roi,
            alarm_active=update.active,
            event_started=update.event_started,
            event_ended=update.event_ended,
            event_count=update.event_count,
        )

    def _predict(
        self,
        action_model: LoadedActionModel,
        timestamp_ms: int,
    ) -> dict[str, float]:
        """批量计算一个模型在H3/H4区域上的锉削概率。"""

        clips = []
        roi_names = list(self.action_rois)
        for name in roi_names:
            frames = sample_timed_frames(
                self.frame_buffers[name],
                timestamp_ms,
                action_model.window_ms,
                action_model.sample_count,
                action_model.image_size,
            )
            clips.append(frames_to_clip(frames, action_model.input_mode))
        batch = torch.stack(clips).to(self.device)
        filing_index = action_model.class_to_index["filing_action"]
        with torch.inference_mode():
            probabilities = action_model.model(batch).softmax(dim=1)[:, filing_index]
        return {
            name: float(probabilities[index].item())
            for index, name in enumerate(roi_names)
        }
