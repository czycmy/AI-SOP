"""AI SOP AOI 监控 MVP。

本包实现第一阶段监控流程：读取检测结果，按区域校验 SOP 顺序和零件是否存在，
最后输出结构化事件。
"""

__all__ = [
    "config",
    "detector",
    "event_log",
    "models",
    "state_machine",
]
