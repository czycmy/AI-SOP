"""SOP 配置加载器。

本模块读取 JSON SOP 配置文件，并转换成带类型的 MonitorConfig 对象。
配置内容包括物理区域、每个区域内的孔位装配顺序，以及运行时校验阈值。
"""

from __future__ import annotations

import json
from pathlib import Path

from sop_monitor.models import MonitorConfig, RegionSpec, StepSpec


def load_config(path: str | Path) -> MonitorConfig:
    """加载并校验 JSON SOP 配置文件。"""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    regions = [
        RegionSpec(
            region_id=region["region_id"],
            name=region.get("name", region["region_id"]),
            roi=tuple(region["roi"]) if "roi" in region else None,
            steps=[
                StepSpec(
                    step=int(step["step"]),
                    hole_id=step["hole_id"],
                    part_type=step.get("part_type", "installed_part"),
                    roi=tuple(step["roi"]) if "roi" in step else None,
                )
                for step in region["steps"]
            ],
        )
        for region in data["regions"]
    ]

    # 状态机至少需要一个区域，并且每个区域至少要有一个 SOP 步骤。
    if not regions:
        raise ValueError("SOP config must contain at least one region.")
    for region in regions:
        if not region.steps:
            raise ValueError(f"Region {region.region_id} must contain at least one step.")

    return MonitorConfig(
        regions=regions,
        confidence_threshold=float(data.get("confidence_threshold", 0.75)),
        stable_frames_required=int(data.get("stable_frames_required", 8)),
        tool_class=str(data.get("tool_class", "l_tool_visible")),
        tool_confidence_threshold=float(data.get("tool_confidence_threshold", 0.5)),
        tool_evidence_frames_required=int(data.get("tool_evidence_frames_required", 5)),
        tool_leave_frames_required=int(data.get("tool_leave_frames_required", 5)),
        tool_leave_timeout_ms=int(data.get("tool_leave_timeout_ms", 4500)),
        forbidden_tool_class=str(data.get("forbidden_tool_class", "forbidden_tool")),
        forbidden_tool_confidence_threshold=float(data.get("forbidden_tool_confidence_threshold", 0.5)),
        display_forbidden_tool_confidence_threshold=float(
            data.get("display_forbidden_tool_confidence_threshold", 0.4)
        ),
        forbidden_tool_stable_frames_required=int(data.get("forbidden_tool_stable_frames_required", 3)),
        forbidden_tool_clear_frames_required=int(data.get("forbidden_tool_clear_frames_required", 15)),
        forbidden_tool_clear_timeout_ms=int(data.get("forbidden_tool_clear_timeout_ms", 5000)),
        missing_timeout_frames=int(data.get("missing_timeout_frames", 120)),
    )
