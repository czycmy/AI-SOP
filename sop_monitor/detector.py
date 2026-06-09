"""检测输入适配器。

第一阶段 MVP 先从 JSONL 读取预计算检测结果，这样在相机和模型链路接好之前，
也可以先验证 SOP 业务逻辑。后续真实 YOLO 检测器只需要实现同样的 Detector
协议，并持续产出 FrameObservation 对象。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from sop_monitor.models import Detection, FrameObservation


class Detector(Protocol):
    """监控运行时使用的统一检测器接口。"""

    def observations(self) -> Iterable[FrameObservation]:
        """逐帧产出检测结果。"""


class JsonlDetectionReader:
    """从 JSONL 文件读取预计算的逐帧检测结果。

    这是第一阶段的接入点。后续 YOLO 检测器可以替换本类，只要继续产出同样的
    FrameObservation 对象即可。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def observations(self) -> Iterable[FrameObservation]:
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

                # 将原始 JSON 字典归一化为统一结构，后续 YOLO/TensorRT 检测器
                # 也应该返回这个结构。
                yield FrameObservation(
                    frame_index=int(payload["frame_index"]),
                    timestamp_ms=payload.get("timestamp_ms"),
                    detections=[
                        Detection(
                            region_id=item["region_id"],
                            hole_id=item["hole_id"],
                            part_type=item.get("part_type", "installed_part"),
                            present=bool(item.get("present", True)),
                            confidence=float(item.get("confidence", 1.0)),
                            bbox=tuple(item["bbox"]) if "bbox" in item else None,
                            extra=item.get("extra", {}),
                        )
                        for item in payload.get("detections", [])
                    ],
                )
