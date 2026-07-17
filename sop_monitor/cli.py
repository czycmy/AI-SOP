"""第一阶段 SOP 监控的命令行入口。

本 CLI 负责串联 SOP 配置加载器、JSONL 检测结果读取器、状态机和事件日志器。
主要用于本地验证，以及在真实相机/模型链路接入前回放检测结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sop_monitor.config import load_config
from sop_monitor.detector import JsonlDetectionReader
from sop_monitor.event_log import JsonlEventLogger
from sop_monitor.state_machine import SopStateMachine


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="AI SOP monitor state-machine replay")
    parser.add_argument("--config", required=True, help="Path to SOP config JSON.")
    parser.add_argument("--detections", required=True, help="Path to detection JSONL.")
    parser.add_argument("--events", default="runs/events.jsonl", help="Output event JSONL path.")
    return parser


def main() -> int:
    """基于 JSONL 检测流运行 SOP 监控。"""

    args = build_parser().parse_args()
    config = load_config(args.config)
    detector = JsonlDetectionReader(args.detections)
    state_machine = SopStateMachine(config)
    logger = JsonlEventLogger(args.events)

    events_path = Path(args.events)
    # 每次命令行回放都从干净事件文件开始，避免新旧结果混在一起。
    if events_path.exists():
        events_path.unlink()

    total_events = 0
    for observation in detector.observations():
        # 状态机负责全部业务逻辑：区域切换、顺序校验、稳定帧投票和异常事件生成。
        events = state_machine.update(observation)
        logger.write_many(events)
        total_events += len(events)
        for event in events:
            print(f"[{event.frame_index}] {event.event_type.value}: {event.message}")

    print(f"done: {total_events} events written to {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
