"""AI SOP AOI 监控系统的核心数据模型。

本文件定义配置加载、检测输入、SOP 状态校验、事件日志都会用到的共享结构。
这些模型保持简单明确，后续把第一阶段 JSONL 检测输入替换成真实 YOLO/ONNX
检测器时，只需要输出同样的数据结构即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """SOP 监控过程中产生的业务事件类型。"""

    STEP_COMPLETED = "step_completed"
    REGION_COMPLETED = "region_completed"
    ALL_COMPLETED = "all_completed"
    ORDER_ERROR = "order_error"
    MISSING_PART = "missing_part"


@dataclass(frozen=True)
class StepSpec:
    """一个 SOP 步骤：指定孔位需要确认已装。"""

    step: int
    hole_id: str
    part_type: str


@dataclass(frozen=True)
class RegionSpec:
    """一个物理监控区域，内部包含一组有顺序的 SOP 步骤。"""

    region_id: str
    name: str
    steps: list[StepSpec]


@dataclass(frozen=True)
class MonitorConfig:
    """一次检测任务的运行参数和区域顺序配置。"""

    regions: list[RegionSpec]
    confidence_threshold: float = 0.75
    stable_frames_required: int = 8
    missing_timeout_frames: int = 120


@dataclass(frozen=True)
class Detection:
    """单帧中模型对某个孔位/零件状态的一条检测结果。"""

    region_id: str
    hole_id: str
    part_type: str
    present: bool
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameObservation:
    """某一帧视频对应的全部检测结果。"""

    frame_index: int
    detections: list[Detection]
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class MonitorEvent:
    """可直接写入日志或展示到界面的 SOP 结果/异常事件。"""

    event_type: EventType
    frame_index: int
    region_id: str
    expected_step: int | None
    expected_hole_id: str | None
    expected_part_type: str | None
    observed_hole_id: str | None = None
    observed_part_type: str | None = None
    confidence: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "frame_index": self.frame_index,
            "region_id": self.region_id,
            "expected_step": self.expected_step,
            "expected_hole_id": self.expected_hole_id,
            "expected_part_type": self.expected_part_type,
            "observed_hole_id": self.observed_hole_id,
            "observed_part_type": self.observed_part_type,
            "confidence": self.confidence,
            "message": self.message,
        }
