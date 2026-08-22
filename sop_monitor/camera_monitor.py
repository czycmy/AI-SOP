"""摄像头实时 SOP 监控命令。

本命令从摄像头读取实时画面，调用 YOLO 模型检测已装零件和 L 型工具；零件按
孔位 ROI 映射为 H1/H2 等状态，工具按总监控区域映射，最后交给 SOP 状态机判断
落位、紧固、顺序和完整性。
可选使用 MediaPipe 在后端检测手部关键点，手部结果只用于画面展示和状态提示，
不参与 SOP 完成/异常判定。部分 macOS/无头环境下 MediaPipe 图形后端可能不稳定，
因此需要通过 --hands 显式开启。

本文件保留为命令行监控/调试入口；现场主入口是 ``desktop_app.py``。离线视频
联调、海康 SDK 和双动作模型也应优先通过桌面客户端启动。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sop_monitor.camera_utils import (
    add_camera_source_arguments,
    draw_monitor_overlay,
    has_any_roi,
    match_detection_to_hole,
    match_detection_to_region,
    open_camera,
    resolve_camera_source,
)
from sop_monitor.config import load_config
from sop_monitor.event_log import JsonlEventLogger
from sop_monitor.hand_detector import MediaPipeHandDetector, any_hand_near_roi, draw_hand_overlay
from sop_monitor.models import Detection, FrameObservation
from sop_monitor.state_machine import SopStateMachine


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="摄像头实时 SOP 监控")
    add_camera_source_arguments(parser)
    parser.add_argument("--config", required=True, help="SOP 配置 JSON。")
    parser.add_argument("--model", required=True, help="YOLO 模型路径，例如 weights/best.pt。")
    parser.add_argument("--events", default="runs/camera_events.jsonl", help="事件日志输出路径。")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 检测置信度阈值。")
    parser.add_argument("--width", type=int, default=None, help="采集宽度。")
    parser.add_argument("--height", type=int, default=None, help="采集高度。")
    parser.add_argument("--display", action="store_true", help="显示实时画面窗口。")
    parser.add_argument("--hands", action="store_true", help="开启后端 MediaPipe 手部监控。")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe 手部模型路径。")
    return parser


def main() -> int:
    """启动摄像头实时监控。"""

    import cv2
    from ultralytics import YOLO

    args = build_parser().parse_args()
    config = load_config(args.config)
    if not has_any_roi(config):
        raise ValueError("SOP 配置缺少孔位 roi，无法把模型检测结果映射到 H1/H2 等孔位。")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 YOLO 模型文件：{model_path}")

    events_path = Path(args.events)
    if events_path.exists():
        events_path.unlink()

    camera_source = resolve_camera_source(args)
    capture = open_camera(camera_source, args.width, args.height)
    model = YOLO(str(model_path))
    state_machine = SopStateMachine(config)
    logger = JsonlEventLogger(events_path)
    hand_detector = MediaPipeHandDetector(model_path=args.hand_model) if args.hands else None

    frame_index = 0
    print("摄像头实时监控已启动，按 q 或 Esc 退出窗口。")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("读取摄像头画面失败。")

            frame_index += 1
            detections = predict_frame(model, config, frame, args.conf)
            observation = FrameObservation(frame_index=frame_index, detections=detections)
            events = state_machine.update(observation)
            logger.write_many(events)

            hands = hand_detector.detect(frame) if hand_detector else []
            active_region = state_machine.active_region
            expected = state_machine.expected_step
            near_active_roi = any_hand_near_roi(
                hands,
                expected.roi if expected else None,
                frame.shape[1],
                frame.shape[0],
            )

            for event in events:
                print(f"[{event.frame_index}] {event.event_type.value}: {event.message}")

            if args.display:
                draw_monitor_overlay(
                    frame,
                    config,
                    detections,
                    active_region.region_id if active_region else None,
                    expected.hole_id if expected else None,
                )
                if hand_detector:
                    draw_hand_overlay(frame, hands, near_active_roi)
                cv2.imshow("AI SOP Camera Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            elif state_machine.completed:
                break
    finally:
        capture.release()
        if hand_detector:
            hand_detector.close()
        if args.display:
            cv2.destroyAllWindows()

    print(f"done: events written to {events_path}")
    return 0


def predict_frame(
    model,
    config,
    frame,
    conf: float,
    target_classes: set[str] | None = None,
) -> list[Detection]:
    """用 YOLO 推理一帧，并按需过滤类别后转换成孔位检测结果。"""

    height, width = frame.shape[:2]
    results = model.predict(frame, conf=conf, verbose=False)
    detections: list[Detection] = []
    seen_hole_classes: set[tuple[str, str, str]] = set()

    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, "installed_part")
            if target_classes is not None and class_name not in target_classes:
                continue
            xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            # 零件属于某个具体孔位；移动工具可能跨出孔位小框，因此只要求它的
            # 中心位于区域总 ROI。二者不能共用同一种 ROI 匹配方式。
            if class_name in {config.tool_class, config.forbidden_tool_class}:
                region_id = match_detection_to_region(config, xyxy, width, height)
                if region_id is None:
                    continue
                hole_id = "*"
            else:
                matched = match_detection_to_hole(config, xyxy, width, height)
                if matched is None:
                    continue
                region_id, step = matched
                hole_id = step.hole_id
            key = (region_id, hole_id, class_name)
            # 同一帧、同一孔位、同一类别只保留一个结果，避免重复检测框让状态机
            # 在一帧内累计多次证据。
            if key in seen_hole_classes:
                continue
            seen_hole_classes.add(key)
            detections.append(Detection(
                region_id=region_id,
                hole_id=hole_id,
                part_type=class_name,
                present=True,
                confidence=float(box.conf[0].item()),
                bbox=xyxy,
            ))

    return detections


if __name__ == "__main__":
    raise SystemExit(main())
