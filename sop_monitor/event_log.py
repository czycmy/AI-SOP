"""SOP 监控结果的事件日志。

本日志器每行写入一个 JSON 对象。JSONL 适合生产场景，因为它可以实时追加，
后续也方便导入数据库、看板或异常复核工具。
"""

from __future__ import annotations

import json
from pathlib import Path

from sop_monitor.models import MonitorEvent


class JsonlEventLogger:
    """将监控事件追加写入 JSONL 文件。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_many(self, events: list[MonitorEvent]) -> None:
        """批量写入事件；没有事件时不触碰文件。"""

        if not events:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
