"""使用 RGB 与方向光流双模型测试完整装配视频。

脚本支持旧的单模型测试，也支持同时加载 RGB 外观模型和方向光流模型。
双模型模式会融合 H3/H4 各自的概率，并通过连续事件状态机输出稳定报警，
保证一次连续锉削只生成一条异常事件。输出包括结果视频、逐窗口预测 CSV 和
锉削事件 CSV；该脚本只验证动作识别，不修改现有 YOLO/SOP 状态机。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sop_monitor.action_recognition import (
    MultiRoiActionEventStateMachine,
    build_action_model,
    crop_normalized_roi,
    frames_to_clip,
    fuse_action_probabilities,
    motion_energy,
)


@dataclass
class LoadedActionModel:
    """动作权重及其训练时保存的输入配置。"""

    model: torch.nn.Module
    class_to_index: dict[str, int]
    sample_count: int
    image_size: int
    sample_fps: float
    action_rois: dict[str, tuple[float, float, float, float]]
    input_mode: str
    recommended_threshold: float

    @property
    def window_seconds(self) -> float:
        """返回该模型一次推理覆盖的时间长度。"""

        return self.sample_count / self.sample_fps


def build_parser() -> argparse.ArgumentParser:
    """创建完整视频测试参数。"""

    parser = argparse.ArgumentParser(description="测试裸锉刀锉削连续动作")
    parser.add_argument("--model", help="兼容旧流程的单个动作模型 best.pt。")
    parser.add_argument("--rgb-model", help="RGB 外观模型 best.pt。")
    parser.add_argument("--flow-model", help="方向光流模型 best.pt。")
    parser.add_argument("--source", required=True, help="待测试的完整视频。")
    parser.add_argument("--output", default="runs/filing_action_test", help="结果输出目录。")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.0,
        help="覆盖权重内的动作窗口长度；0表示分别使用训练参数。",
    )
    parser.add_argument("--stride-seconds", type=float, default=0.2, help="相邻推理窗口间隔。")
    parser.add_argument("--threshold", type=float, default=None, help="融合报警阈值；默认融合两个权重的推荐值。")
    parser.add_argument("--clear-threshold", type=float, default=None, help="解除报警阈值；默认为触发阈值的70%。")
    parser.add_argument("--rgb-weight", type=float, default=0.7, help="RGB概率融合权重，方向光流权重为1减去该值。")
    parser.add_argument("--vote-window", type=int, default=4, help="触发报警的滑动投票窗口数。")
    parser.add_argument("--alarm-windows", type=int, default=3, help="投票窗口内至少多少个阳性才触发报警。")
    parser.add_argument("--clear-windows", type=int, default=4, help="连续多少个低分窗口解除报警。")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser


def choose_device(requested: str) -> torch.device:
    """选择可用推理设备。"""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, device: torch.device) -> LoadedActionModel:
    """从训练检查点恢复模型和输入配置。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    if "filing_action" not in class_to_index:
        raise ValueError(f"动作模型类别中缺少 filing_action：{path}")
    head_dropout = float(checkpoint.get("head_dropout", 0.0))
    model = build_action_model(
        len(class_to_index),
        pretrained=False,
        head_dropout=head_dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    raw_action_rois = checkpoint.get("action_rois")
    if raw_action_rois:
        action_rois = {
            name: tuple(float(value) for value in roi)
            for name, roi in raw_action_rois.items()
        }
    else:
        action_rois = {
            "MAIN": tuple(checkpoint.get("roi") or (0.0, 0.0, 1.0, 1.0))
        }
    return LoadedActionModel(
        model=model,
        class_to_index=class_to_index,
        sample_count=int(checkpoint["frames"]),
        image_size=int(checkpoint["image_size"]),
        sample_fps=float(
            checkpoint.get("sample_fps") or checkpoint["frames"] / 2.5
        ),
        action_rois=action_rois,
        input_mode=str(checkpoint.get("input_mode") or "rgb"),
        recommended_threshold=float(checkpoint.get("recommended_threshold") or 0.5),
    )


def validate_model_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    """校验单模型与双模型参数，避免混用产生含糊结果。"""

    single_path = Path(args.model) if args.model else None
    rgb_path = Path(args.rgb_model) if args.rgb_model else None
    flow_path = Path(args.flow_model) if args.flow_model else None
    if single_path and (rgb_path or flow_path):
        raise ValueError("--model 不能与 --rgb-model/--flow-model 同时使用。")
    if not single_path and not (rgb_path and flow_path):
        raise ValueError("请使用 --model，或同时提供 --rgb-model 与 --flow-model。")
    for path in (single_path, rgb_path, flow_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"找不到动作模型：{path}")
    return single_path, rgb_path, flow_path


def rois_match(
    first: dict[str, tuple[float, float, float, float]],
    second: dict[str, tuple[float, float, float, float]],
) -> bool:
    """判断两个模型保存的动作 ROI 是否一致。"""

    if first.keys() != second.keys():
        return False
    return all(
        all(abs(left - right) < 1e-6 for left, right in zip(first[name], second[name]))
        for name in first
    )


def prepare_clip(
    frames: list,
    model: LoadedActionModel,
    device: torch.device,
) -> torch.Tensor:
    """按单个模型的训练参数采样、缩放并转换输入。"""

    positions = torch.linspace(0, len(frames) - 1, model.sample_count).round().int().tolist()
    sampled = [
        cv2.resize(
            frames[position],
            (model.image_size, model.image_size),
            interpolation=cv2.INTER_AREA,
        )
        for position in positions
    ]
    return frames_to_clip(sampled, model.input_mode).to(device)


def predict_rois(
    model: LoadedActionModel,
    frame_buffers: dict[str, deque],
    window_frames: int,
    device: torch.device,
) -> dict[str, float]:
    """批量计算一个模型在所有动作 ROI 上的锉削概率。"""

    roi_names = list(frame_buffers)
    clips = torch.stack([
        prepare_clip(list(frame_buffers[name])[-window_frames:], model, device)
        for name in roi_names
    ])
    filing_index = model.class_to_index["filing_action"]
    with torch.inference_mode():
        probabilities = model.model(clips).softmax(dim=1)[:, filing_index]
    return {
        name: float(probabilities[index].item())
        for index, name in enumerate(roi_names)
    }


def write_prediction_csv(path: Path, rows: list[dict]) -> None:
    """保存逐窗口双路概率和事件状态。"""

    fieldnames = (
        "timestamp_seconds",
        "predicted_label",
        "rgb_probability",
        "flow_probability",
        "filing_probability",
        "active_roi",
        "active_roi_motion",
        "h3_rgb_probability",
        "h3_flow_probability",
        "h3_fused_probability",
        "h4_rgb_probability",
        "h4_flow_probability",
        "h4_fused_probability",
        "event_started",
        "event_ended",
        "event_count",
        "alarm_active",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_event_csv(path: Path, events: list[dict]) -> None:
    """保存去重后的锉削事件清单。"""

    fieldnames = (
        "event_id",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "peak_probability",
        "peak_roi",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def main() -> int:
    """执行滑动窗口双路推理并保存视频、明细和事件。"""

    args = build_parser().parse_args()
    single_path, rgb_path, flow_path = validate_model_paths(args)
    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到测试视频：{source_path}")
    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds 必须大于0。")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "result.mp4"
    output_csv = output_dir / "predictions.csv"
    output_events = output_dir / "events.csv"
    device = choose_device(args.device)

    single_model = load_model(single_path, device) if single_path else None
    rgb_model = load_model(rgb_path, device) if rgb_path else None
    flow_model = load_model(flow_path, device) if flow_path else None
    fusion_enabled = rgb_model is not None and flow_model is not None

    if fusion_enabled:
        assert rgb_model is not None and flow_model is not None
        if rgb_model.input_mode != "rgb":
            raise ValueError(f"--rgb-model 的输入模式应为 rgb，实际为 {rgb_model.input_mode}。")
        if flow_model.input_mode != "flow":
            raise ValueError(f"--flow-model 的输入模式应为 flow，实际为 {flow_model.input_mode}。")
        if not rois_match(rgb_model.action_rois, flow_model.action_rois):
            raise ValueError("RGB与方向光流模型保存的H3/H4动作ROI不一致，不能直接融合。")
        action_rois = rgb_model.action_rois
        default_threshold = fuse_action_probabilities(
            rgb_model.recommended_threshold,
            flow_model.recommended_threshold,
            args.rgb_weight,
        )
        models = (rgb_model, flow_model)
    else:
        assert single_model is not None
        action_rois = single_model.action_rois
        default_threshold = single_model.recommended_threshold
        models = (single_model,)

    threshold = float(args.threshold if args.threshold is not None else default_threshold)
    clear_threshold = float(
        args.clear_threshold
        if args.clear_threshold is not None
        else threshold * 0.7
    )
    event_state = MultiRoiActionEventStateMachine(
        roi_names=set(action_rois),
        trigger_threshold=threshold,
        clear_threshold=clear_threshold,
        vote_window=args.vote_window,
        trigger_votes=args.alarm_windows,
        clear_windows=args.clear_windows,
    )

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开测试视频：{source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建结果视频：{output_video}")

    model_windows = {
        id(model): max(
            model.sample_count,
            round((args.window_seconds or model.window_seconds) * fps),
        )
        for model in models
    }
    max_window_frames = max(model_windows.values())
    stride_frames = max(1, round(args.stride_seconds * fps))
    frame_buffers = {
        name: deque(maxlen=max_window_frames)
        for name in action_rois
    }
    frame_index = 0
    filing_probability = 0.0
    rgb_probability = 0.0
    flow_probability = 0.0
    predicted_label = "warming_up"
    active_roi_name = "-"
    roi_motion = {name: 0.0 for name in action_rois}
    alarm_active = False
    rows: list[dict] = []
    events: list[dict] = []
    current_event: dict | None = None
    last_timestamp = 0.0
    zero_probabilities = {name: 0.0 for name in action_rois}
    rgb_probabilities = zero_probabilities.copy()
    flow_probabilities = zero_probabilities.copy()
    fused_probabilities = zero_probabilities.copy()
    mode_text = "rgb+flow" if fusion_enabled else models[0].input_mode
    print(
        f"设备：{device}；视频：{total_frames}帧，{fps:.2f} FPS；"
        f"输入：{mode_text}；动作ROI：{action_rois}；触发/解除阈值："
        f"{threshold:.2f}/{clear_threshold:.2f}；投票："
        f"{args.alarm_windows}/{args.vote_window}"
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            last_timestamp = frame_index / fps
            for name, roi in action_rois.items():
                frame_buffers[name].append(crop_normalized_roi(frame, roi).copy())

            should_predict = (
                all(len(items) == max_window_frames for items in frame_buffers.values())
                and (frame_index - max_window_frames) % stride_frames == 0
            )
            event_started = False
            event_ended = False
            if should_predict:
                roi_motion = {
                    name: motion_energy(list(frame_buffers[name]))
                    for name in frame_buffers
                }
                if fusion_enabled:
                    assert rgb_model is not None and flow_model is not None
                    rgb_probabilities = predict_rois(
                        rgb_model,
                        frame_buffers,
                        model_windows[id(rgb_model)],
                        device,
                    )
                    flow_probabilities = predict_rois(
                        flow_model,
                        frame_buffers,
                        model_windows[id(flow_model)],
                        device,
                    )
                    fused_probabilities = {
                        name: fuse_action_probabilities(
                            rgb_probabilities[name],
                            flow_probabilities[name],
                            args.rgb_weight,
                        )
                        for name in action_rois
                    }
                else:
                    assert single_model is not None
                    single_probabilities = predict_rois(
                        single_model,
                        frame_buffers,
                        model_windows[id(single_model)],
                        device,
                    )
                    fused_probabilities = single_probabilities
                    if single_model.input_mode == "rgb":
                        rgb_probabilities = single_probabilities
                        flow_probabilities = zero_probabilities.copy()
                    elif single_model.input_mode == "flow":
                        rgb_probabilities = zero_probabilities.copy()
                        flow_probabilities = single_probabilities
                    else:
                        rgb_probabilities = single_probabilities
                        flow_probabilities = zero_probabilities.copy()

                update = event_state.update(fused_probabilities, last_timestamp)
                candidate_rois = event_state.active_rois or set(fused_probabilities)
                active_roi_name = max(candidate_rois, key=fused_probabilities.get)
                filing_probability = fused_probabilities[active_roi_name]
                rgb_probability = rgb_probabilities[active_roi_name]
                flow_probability = flow_probabilities[active_roi_name]
                predicted_label = (
                    "filing_action"
                    if filing_probability >= threshold
                    else "non_filing_action"
                )
                alarm_active = update.active
                event_started = update.event_started
                event_ended = update.event_ended

                if event_started:
                    current_event = {
                        "event_id": update.event_count,
                        "start_seconds": update.event_start_seconds or last_timestamp,
                        "end_seconds": "",
                        "duration_seconds": "",
                        "peak_probability": filing_probability,
                        "peak_roi": active_roi_name,
                    }
                if current_event is not None and filing_probability > float(
                    current_event["peak_probability"]
                ):
                    current_event["peak_probability"] = filing_probability
                    current_event["peak_roi"] = active_roi_name
                if event_ended and current_event is not None:
                    end_seconds = update.event_end_seconds or last_timestamp
                    current_event["end_seconds"] = end_seconds
                    current_event["duration_seconds"] = end_seconds - float(
                        current_event["start_seconds"]
                    )
                    events.append(current_event)
                    current_event = None

                rows.append({
                    "timestamp_seconds": f"{last_timestamp:.3f}",
                    "predicted_label": predicted_label,
                    "rgb_probability": f"{rgb_probability:.6f}",
                    "flow_probability": f"{flow_probability:.6f}",
                    "filing_probability": f"{filing_probability:.6f}",
                    "active_roi": active_roi_name,
                    "active_roi_motion": f"{roi_motion[active_roi_name]:.6f}",
                    "h3_rgb_probability": f"{rgb_probabilities.get('H3', 0.0):.6f}",
                    "h3_flow_probability": f"{flow_probabilities.get('H3', 0.0):.6f}",
                    "h3_fused_probability": f"{fused_probabilities.get('H3', 0.0):.6f}",
                    "h4_rgb_probability": f"{rgb_probabilities.get('H4', 0.0):.6f}",
                    "h4_flow_probability": f"{flow_probabilities.get('H4', 0.0):.6f}",
                    "h4_fused_probability": f"{fused_probabilities.get('H4', 0.0):.6f}",
                    "event_started": int(event_started),
                    "event_ended": int(event_ended),
                    "event_count": update.event_count,
                    "alarm_active": int(alarm_active),
                })
                print(
                    f"{last_timestamp:7.2f}s | rgb={rgb_probability:.3f} "
                    f"flow={flow_probability:.3f} fused={filing_probability:.3f} "
                    f"| roi={active_roi_name} | alarm={alarm_active} "
                    f"events={update.event_count}"
                )

            color = (0, 0, 255) if alarm_active else (30, 200, 30)
            cv2.rectangle(frame, (12, 12), (505, 135), (20, 20, 20), -1)
            cv2.putText(frame, f"Action: {predicted_label}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"RGB: {rgb_probability:.3f}  Flow: {flow_probability:.3f}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            cv2.putText(frame, f"Fused: {filing_probability:.3f}  ROI: {active_roi_name}", (25, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            cv2.putText(frame, f"{'ALARM' if alarm_active else 'NORMAL'}  Events: {event_state.event_count}", (25, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    finish_update = event_state.finish(last_timestamp)
    if finish_update.event_ended and current_event is not None:
        end_seconds = finish_update.event_end_seconds or last_timestamp
        current_event["end_seconds"] = end_seconds
        current_event["duration_seconds"] = end_seconds - float(
            current_event["start_seconds"]
        )
        events.append(current_event)

    for event in events:
        event["start_seconds"] = f"{float(event['start_seconds']):.3f}"
        event["end_seconds"] = f"{float(event['end_seconds']):.3f}"
        event["duration_seconds"] = f"{float(event['duration_seconds']):.3f}"
        event["peak_probability"] = f"{float(event['peak_probability']):.6f}"
    write_prediction_csv(output_csv, rows)
    write_event_csv(output_events, events)
    print(f"结果视频：{output_video.resolve()}")
    print(f"预测明细：{output_csv.resolve()}")
    print(f"事件清单：{output_events.resolve()}（共{len(events)}次）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
