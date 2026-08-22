"""连续动作识别的共享预处理、双路融合和事件判定工具。

本模块统一训练与推理阶段的 H3/H4 动作 ROI 裁剪、时间戳区域排除、
RGB/帧差/方向光流输入转换、R3D-18 分类头结构，以及连续锉削事件状态机。
方向光流保留运动方向，用于区分锉刀往复运动与正常 L 型工具紧固动作。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
MOTION_MEAN = torch.tensor([0.12, 0.12, 0.12]).view(1, 3, 1, 1)
MOTION_STD = torch.tensor([0.18, 0.18, 0.18]).view(1, 3, 1, 1)
FLOW_COMPONENT_LIMIT = 12.0


@dataclass(frozen=True)
class ActionAlarmUpdate:
    """一次动作概率更新后的事件状态。"""

    active: bool
    event_started: bool
    event_ended: bool
    event_count: int
    event_start_seconds: float | None
    event_end_seconds: float | None


class ActionEventStateMachine:
    """把滑动窗口概率合并为稳定事件，避免同一动作被重复计数。"""

    def __init__(
        self,
        trigger_threshold: float,
        clear_threshold: float,
        vote_window: int = 4,
        trigger_votes: int = 3,
        clear_windows: int = 4,
    ) -> None:
        if not 0.0 <= clear_threshold < trigger_threshold <= 1.0:
            raise ValueError("解除阈值必须小于触发阈值，且两个阈值都应位于0～1。")
        if vote_window < 1 or not 1 <= trigger_votes <= vote_window:
            raise ValueError("触发票数必须位于1和投票窗口长度之间。")
        if clear_windows < 1:
            raise ValueError("解除报警所需窗口数必须大于0。")
        self.trigger_threshold = trigger_threshold
        self.clear_threshold = clear_threshold
        self.vote_window = vote_window
        self.trigger_votes = trigger_votes
        self.clear_windows = clear_windows
        self.active = False
        self.event_count = 0
        self.event_start_seconds: float | None = None
        self._recent_votes: deque[tuple[float, bool]] = deque(maxlen=vote_window)
        self._clear_count = 0
        self._clear_started_at: float | None = None

    def update(self, probability: float, timestamp_seconds: float) -> ActionAlarmUpdate:
        """输入一个窗口概率，并返回报警及事件起止变化。"""

        probability = float(min(1.0, max(0.0, probability)))
        event_started = False
        event_ended = False
        ended_start: float | None = None
        event_end: float | None = None

        if not self.active:
            self._recent_votes.append(
                (timestamp_seconds, probability >= self.trigger_threshold)
            )
            positive_votes = [
                timestamp for timestamp, positive in self._recent_votes if positive
            ]
            if len(positive_votes) >= self.trigger_votes:
                self.active = True
                self.event_count += 1
                self.event_start_seconds = positive_votes[0]
                self._clear_count = 0
                self._clear_started_at = None
                event_started = True
        else:
            if probability < self.clear_threshold:
                if self._clear_count == 0:
                    self._clear_started_at = timestamp_seconds
                self._clear_count += 1
            else:
                self._clear_count = 0
                self._clear_started_at = None
            if self._clear_count >= self.clear_windows:
                ended_start = self.event_start_seconds
                event_end = (
                    self._clear_started_at
                    if self._clear_started_at is not None
                    else timestamp_seconds
                )
                self.active = False
                self.event_start_seconds = None
                self._recent_votes.clear()
                self._clear_count = 0
                self._clear_started_at = None
                event_ended = True

        return ActionAlarmUpdate(
            active=self.active,
            event_started=event_started,
            event_ended=event_ended,
            event_count=self.event_count,
            event_start_seconds=(
                ended_start if event_ended else self.event_start_seconds
            ),
            event_end_seconds=event_end,
        )

    def finish(self, timestamp_seconds: float) -> ActionAlarmUpdate:
        """在视频结束时关闭尚未结束的事件。"""

        if not self.active:
            return ActionAlarmUpdate(
                active=False,
                event_started=False,
                event_ended=False,
                event_count=self.event_count,
                event_start_seconds=None,
                event_end_seconds=None,
            )
        event_start = self.event_start_seconds
        self.active = False
        self.event_start_seconds = None
        self._recent_votes.clear()
        self._clear_count = 0
        self._clear_started_at = None
        return ActionAlarmUpdate(
            active=False,
            event_started=False,
            event_ended=True,
            event_count=self.event_count,
            event_start_seconds=event_start,
            event_end_seconds=timestamp_seconds,
        )


class MultiRoiActionEventStateMachine:
    """按 ROI 独立投票，再合并为一次全局动作报警。

    H3、H4 的零散高分不能互相凑票，只有同一 ROI 内达到连续证据要求时
    才触发报警；多个相邻 ROI 同时成立时仍只记录一次连续动作事件。
    """

    def __init__(
        self,
        roi_names: list[str] | tuple[str, ...] | set[str],
        trigger_threshold: float,
        clear_threshold: float,
        vote_window: int = 4,
        trigger_votes: int = 3,
        clear_windows: int = 4,
    ) -> None:
        names = tuple(str(name) for name in roi_names)
        if not names:
            raise ValueError("至少需要一个动作 ROI。")
        if len(set(names)) != len(names):
            raise ValueError("动作 ROI 名称不能重复。")
        self._states = {
            name: ActionEventStateMachine(
                trigger_threshold=trigger_threshold,
                clear_threshold=clear_threshold,
                vote_window=vote_window,
                trigger_votes=trigger_votes,
                clear_windows=clear_windows,
            )
            for name in names
        }
        self.active = False
        self.event_count = 0
        self.event_start_seconds: float | None = None

    @property
    def active_rois(self) -> set[str]:
        """返回当前已经分别通过投票的 ROI。"""

        return {name for name, state in self._states.items() if state.active}

    def update(
        self,
        probabilities: dict[str, float],
        timestamp_seconds: float,
    ) -> ActionAlarmUpdate:
        """分别更新各 ROI，并返回合并后的全局报警状态。"""

        missing = self._states.keys() - probabilities.keys()
        if missing:
            raise ValueError(f"缺少动作 ROI 概率：{', '.join(sorted(missing))}")

        was_active = self.active
        updates = {
            name: state.update(probabilities[name], timestamp_seconds)
            for name, state in self._states.items()
        }
        self.active = bool(self.active_rois)
        event_started = self.active and not was_active
        event_ended = was_active and not self.active
        ended_start: float | None = None
        event_end: float | None = None

        if event_started:
            self.event_count += 1
            starts = [
                update.event_start_seconds
                for update in updates.values()
                if update.active and update.event_start_seconds is not None
            ]
            self.event_start_seconds = min(starts) if starts else timestamp_seconds
        elif event_ended:
            ended_start = self.event_start_seconds
            ends = [
                update.event_end_seconds
                for update in updates.values()
                if update.event_ended and update.event_end_seconds is not None
            ]
            event_end = max(ends) if ends else timestamp_seconds
            self.event_start_seconds = None

        return ActionAlarmUpdate(
            active=self.active,
            event_started=event_started,
            event_ended=event_ended,
            event_count=self.event_count,
            event_start_seconds=ended_start if event_ended else self.event_start_seconds,
            event_end_seconds=event_end,
        )

    def finish(self, timestamp_seconds: float) -> ActionAlarmUpdate:
        """在视频结束时关闭所有 ROI 中尚未结束的同一次事件。"""

        for state in self._states.values():
            state.finish(timestamp_seconds)
        if not self.active:
            return ActionAlarmUpdate(
                active=False,
                event_started=False,
                event_ended=False,
                event_count=self.event_count,
                event_start_seconds=None,
                event_end_seconds=None,
            )

        event_start = self.event_start_seconds
        self.active = False
        self.event_start_seconds = None
        return ActionAlarmUpdate(
            active=False,
            event_started=False,
            event_ended=True,
            event_count=self.event_count,
            event_start_seconds=event_start,
            event_end_seconds=timestamp_seconds,
        )


def load_action_rois(
    config_path: Path,
    disabled: bool = False,
) -> tuple[dict[str, tuple[float, float, float, float]], float]:
    """读取动作 ROI，并把顶部边界限制在时间戳区域下方。"""

    if disabled:
        return {"FULL": (0.0, 0.0, 1.0, 1.0)}, 0.0
    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_rois = data.get("action_rois", {})
    if not raw_rois.get("H3") or not raw_rois.get("H4"):
        raise ValueError(f"动作配置必须同时包含 H3 和 H4 ROI：{config_path}")
    mask_top = float(data.get("mask_top", 0.115))
    rois = {}
    for name in ("H3", "H4"):
        values = tuple(float(value) for value in raw_rois[name])
        if len(values) != 4 or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{name} ROI 必须是4个0～1归一化坐标：{values}")
        x1, y1, x2, y2 = values
        roi = (x1, max(y1, mask_top), x2, y2)
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            raise ValueError(f"{name} ROI 在排除时间戳后为空：{roi}")
        rois[name] = roi
    return rois, mask_top


def crop_normalized_roi(frame, roi: tuple[float, float, float, float]):
    """按归一化 xyxy 坐标裁剪图像。"""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    left, top = max(0, round(x1 * width)), max(0, round(y1 * height))
    right, bottom = min(width, round(x2 * width)), min(height, round(y2 * height))
    if right <= left or bottom <= top:
        raise ValueError(f"ROI 裁剪结果为空：{roi}")
    return frame[top:bottom, left:right]


def resize_roi_frames(frames, roi, image_size: int) -> list:
    """将同一时间窗口裁剪为固定尺寸的动作画面。"""

    return [
        cv2.resize(
            crop_normalized_roi(frame, roi),
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        for frame in frames
    ]


def motion_energy(frames: list) -> float:
    """计算一个动作窗口的平均帧差能量。"""

    if len(frames) < 2:
        return 0.0
    previous = cv2.cvtColor(cv2.resize(frames[0], (64, 64)), cv2.COLOR_BGR2GRAY)
    values = []
    for frame in frames[1:]:
        current = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        values.append(float(cv2.mean(cv2.absdiff(current, previous))[0]))
        previous = current
    return sum(values) / len(values)


def select_active_roi(
    frames: list,
    rois: dict[str, tuple[float, float, float, float]],
    image_size: int,
) -> tuple[str, list, dict[str, float]]:
    """选择窗口内运动更明显的 H3 或 H4，并返回对应裁剪帧。"""

    candidates = {
        name: resize_roi_frames(frames, roi, image_size)
        for name, roi in rois.items()
    }
    energies = {name: motion_energy(items) for name, items in candidates.items()}
    active_name = max(energies, key=energies.get)
    return active_name, candidates[active_name], energies


def directional_flow_frames(frames: list) -> list[np.ndarray]:
    """计算方向光流，三个通道依次为水平、垂直速度和运动强度。

    水平和垂直速度裁剪到固定像素范围后映射到[-1, 1]；运动强度映射到
    [0, 1]。与绝对帧差不同，正反方向不会被合并成同一种输入。
    """

    if not frames:
        raise ValueError("方向光流至少需要一帧图像。")
    if len(frames) == 1:
        height, width = frames[0].shape[:2]
        return [np.zeros((height, width, 3), dtype=np.float32)]

    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    encoded_frames: list[np.ndarray] = []
    for frame in frames[1:]:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        horizontal = np.clip(
            flow[..., 0] / FLOW_COMPONENT_LIMIT,
            -1.0,
            1.0,
        )
        vertical = np.clip(
            flow[..., 1] / FLOW_COMPONENT_LIMIT,
            -1.0,
            1.0,
        )
        magnitude = np.clip(
            cv2.magnitude(flow[..., 0], flow[..., 1]) / FLOW_COMPONENT_LIMIT,
            0.0,
            1.0,
        )
        encoded_frames.append(
            np.stack((horizontal, vertical, magnitude), axis=-1).astype(np.float32)
        )
        previous = current
    # 首帧没有前序图像，复制第一组光流以保持时间维度不变。
    return [encoded_frames[0].copy(), *encoded_frames]


def frames_to_clip(frames: list, input_mode: str = "rgb") -> torch.Tensor:
    """把 BGR 帧转换为标准化的 C,T,H,W 模型输入。"""

    if input_mode == "flow":
        flow_frames = directional_flow_frames(frames)
        return torch.from_numpy(np.stack(flow_frames)).permute(3, 0, 1, 2).contiguous()

    tensors = [
        torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        for frame in frames
    ]
    clip = torch.stack(tensors).float() / 255.0  # T,C,H,W
    if input_mode == "motion":
        differences = torch.abs(clip[1:] - clip[:-1])
        first = differences[:1] if len(differences) else torch.zeros_like(clip[:1])
        clip = torch.clamp(torch.cat([first, differences], dim=0) * 4.0, 0.0, 1.0)
        clip = (clip - MOTION_MEAN) / MOTION_STD
    elif input_mode == "rgb":
        clip = (clip - KINETICS_MEAN) / KINETICS_STD
    else:
        raise ValueError(f"不支持的动作输入模式：{input_mode}")
    return clip.permute(1, 0, 2, 3).contiguous()


def fuse_action_probabilities(
    rgb_probability: float,
    flow_probability: float,
    rgb_weight: float = 0.7,
) -> float:
    """按可配置权重融合外观概率和方向光流概率。"""

    if not 0.0 <= rgb_weight <= 1.0:
        raise ValueError("RGB融合权重必须位于0～1。")
    rgb_probability = min(1.0, max(0.0, float(rgb_probability)))
    flow_probability = min(1.0, max(0.0, float(flow_probability)))
    return rgb_probability * rgb_weight + flow_probability * (1.0 - rgb_weight)


def build_action_model(
    class_count: int,
    pretrained: bool,
    head_dropout: float,
) -> nn.Module:
    """创建带可选 Dropout 分类头的 R3D-18。"""

    model = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
    input_features = model.fc.in_features
    if head_dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(input_features, class_count))
    else:
        model.fc = nn.Linear(input_features, class_count)
    return model
