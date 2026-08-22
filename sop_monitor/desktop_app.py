"""AI SOP PySide6 桌面客户端。

本文件提供现场部署用的原生桌面界面：自动打开摄像头预览，展示区域 SOP、
装配画面、状态概览和异常记录。传入 YOLO 模型后，后台根据孔位 ROI 判断零件
落位和 L 型工具紧固过程；可选的 RGB + 方向光流双模型用于补充识别裸锉刀
连续动作。动作 ROI 只参与后台推理，不绘制到客户端画面。
监控控制按钮只做界面预留，后续再接入真实控制逻辑。

运行结构：Qt 主线程只负责绘制界面；CameraWorker 独立线程完成取流、YOLO、
SOP 状态机、可选动作模型和手部检测，再通过 Signal 把一帧画面和状态字典送回
主线程。维护时不要在 Qt 主线程直接做模型推理，否则窗口会明显卡顿。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sop_monitor.camera_source import CameraSourceSpec, create_frame_source
from sop_monitor.camera_utils import add_camera_source_arguments, resolve_camera_source
from sop_monitor.config import load_config
from sop_monitor.models import MonitorConfig


@dataclass(frozen=True)
class ActionMonitorOptions:
    """客户端连续动作双模型的启动与判定参数。"""

    rgb_model: str | None = None
    flow_model: str | None = None
    device: str = "auto"
    interval_seconds: float = 0.2
    rgb_weight: float = 0.7
    threshold: float = 0.5
    clear_threshold: float = 0.35
    vote_window: int = 4
    trigger_votes: int = 3
    clear_windows: int = 4

    @property
    def enabled(self) -> bool:
        """两个权重都提供时启用连续动作监控。"""

        return bool(self.rgb_model and self.flow_model)


@dataclass(frozen=True)
class ClientExportOptions:
    """无窗口导出客户端演示视频的参数。"""

    output_path: str | None = None
    fps: float = 10.0
    width: int = 1440
    height: int = 900

    @property
    def enabled(self) -> bool:
        """提供输出路径时启用离屏导出。"""

        return bool(self.output_path)


def build_parser() -> argparse.ArgumentParser:
    """创建桌面客户端命令行参数。"""

    parser = argparse.ArgumentParser(description="AI SOP PySide6 桌面客户端")
    parser.add_argument("--config", default="configs/sample_sop.json", help="SOP 配置 JSON。")
    add_camera_source_arguments(parser)
    parser.add_argument("--width", type=int, default=1280, help="摄像头采集宽度。")
    parser.add_argument("--height", type=int, default=720, help="摄像头采集高度。")
    parser.add_argument("--model", default=None, help="YOLO 模型路径；提供后客户端会启用 SOP 实时检测。")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO 检测置信度阈值。")
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=0,
        help="YOLO 推理间隔帧数；0 表示本地视频自动用 3，实时摄像头用 1。",
    )
    parser.add_argument("--hands", action="store_true", help="开启 MediaPipe 手部监控展示。")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe 手部模型路径。")
    parser.add_argument("--hand-interval", type=int, default=5, help="手部检测间隔帧数，值越大延迟越低但手部刷新越慢。")
    parser.add_argument("--action-rgb-model", default=None, help="连续动作RGB模型best.pt。")
    parser.add_argument("--action-flow-model", default=None, help="连续动作方向光流模型best.pt。")
    parser.add_argument("--action-device", choices=("auto", "cuda", "mps", "cpu"), default="auto", help="连续动作模型推理设备。")
    parser.add_argument("--action-interval", type=float, default=0.2, help="连续动作推理间隔秒数。")
    parser.add_argument("--action-rgb-weight", type=float, default=0.7, help="连续动作RGB融合权重。")
    parser.add_argument("--action-threshold", type=float, default=0.5, help="连续动作报警阈值。")
    parser.add_argument("--action-clear-threshold", type=float, default=0.35, help="连续动作报警解除阈值。")
    parser.add_argument("--action-vote-window", type=int, default=4, help="连续动作触发投票窗口数。")
    parser.add_argument("--action-trigger-votes", type=int, default=3, help="投票窗口内触发报警所需阳性数。")
    parser.add_argument("--action-clear-windows", type=int, default=4, help="解除连续动作报警所需低分窗口数。")
    parser.add_argument("--export-client-video", default=None, help="将完整客户端界面离屏导出为MP4，不打开窗口。")
    parser.add_argument("--export-fps", type=float, default=10.0, help="客户端演示视频输出帧率。")
    parser.add_argument("--export-width", type=int, default=1440, help="客户端演示视频宽度。")
    parser.add_argument("--export-height", type=int, default=900, help="客户端演示视频高度。")
    return parser


def main() -> int:
    """启动 PySide6 桌面客户端。"""

    args = build_parser().parse_args()
    if bool(args.action_rgb_model) != bool(args.action_flow_model):
        raise ValueError("连续动作监控必须同时提供 --action-rgb-model 和 --action-flow-model。")
    if args.export_client_video:
        if args.camera_backend != "opencv" or not Path(str(args.camera)).is_file():
            raise ValueError("客户端视频导出仅支持通过 --camera 指定本地视频文件。")
        if args.export_fps <= 0 or args.export_width <= 0 or args.export_height <= 0:
            raise ValueError("客户端视频导出的帧率、宽度和高度必须大于0。")
        # PySide6尚未导入，此时切换平台插件可确保整个过程不弹出真实窗口。
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    config = load_config(args.config)
    camera_source = resolve_camera_source(args)
    camera_spec = CameraSourceSpec(
        backend=args.camera_backend,
        source=camera_source,
        width=args.width,
        height=args.height,
        hikvision_ip=args.hikvision_ip,
        hikvision_user=args.hikvision_user,
        hikvision_password=args.hikvision_password,
        hikvision_port=args.hikvision_port,
        hikvision_channel=args.hikvision_channel,
        hikvision_sdk_dir=args.hikvision_sdk_dir,
    )
    return run_qt_app(
        config,
        camera_spec,
        args.model,
        args.conf,
        args.detect_interval,
        args.hands,
        args.hand_model,
        args.hand_interval,
        ActionMonitorOptions(
            rgb_model=args.action_rgb_model,
            flow_model=args.action_flow_model,
            device=args.action_device,
            interval_seconds=args.action_interval,
            rgb_weight=args.action_rgb_weight,
            threshold=args.action_threshold,
            clear_threshold=args.action_clear_threshold,
            vote_window=args.action_vote_window,
            trigger_votes=args.action_trigger_votes,
            clear_windows=args.action_clear_windows,
        ),
        ClientExportOptions(
            output_path=args.export_client_video,
            fps=args.export_fps,
            width=args.export_width,
            height=args.export_height,
        ),
    )


def run_qt_app(
    config: MonitorConfig,
    camera_spec: CameraSourceSpec,
    model_path: str | None,
    conf: float,
    detect_interval: int,
    enable_hands: bool,
    hand_model: str,
    hand_interval: int,
    action_options: ActionMonitorOptions,
    export_options: ClientExportOptions,
) -> int:
    """延迟导入 PySide6 并启动应用，避免测试环境没有 Qt 时影响核心模块。"""

    try:
        from PySide6.QtCore import QRectF, Qt, QThread, Signal
        from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QListWidget,
            QMainWindow,
            QPushButton,
            QSizePolicy,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 PySide6，请先安装依赖：.venv/bin/python -m pip install -r requirements.txt") from exc

    import cv2
    import numpy as np
    from sop_monitor.camera_monitor import predict_frame
    from sop_monitor.camera_utils import draw_visible_detection_boxes, has_any_roi
    from sop_monitor.hand_detector import MediaPipeHandDetector, any_hand_near_roi, draw_hand_overlay
    from sop_monitor.models import EventType, FrameObservation
    from sop_monitor.state_machine import SopStateMachine

    class PieChartWidget(QWidget):
        """右侧显示已加工孔位与计划孔位的进度环。"""

        def __init__(self, completed_count: int = 0, planned_count: int = 0):
            super().__init__()
            self.completed_count = completed_count
            self.planned_count = planned_count
            self.setMinimumSize(86, 86)
            self.setMaximumHeight(112)

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            side = min(self.width(), self.height()) - 16
            rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
            if self.planned_count <= 0:
                painter.setPen(QPen(QColor("#d9e1ea"), 10))
                painter.drawEllipse(rect)
                painter.setPen(QColor("#52606d"))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "0")
                return

            completed = min(self.completed_count, self.planned_count)
            completed_span = int(360 * 16 * completed / self.planned_count)
            painter.setPen(QPen(QColor("#d9e1ea"), 10))
            painter.drawEllipse(rect)
            painter.setPen(QPen(QColor("#16a34a"), 10))
            painter.drawArc(rect, 90 * 16, -completed_span)
            painter.setPen(QColor("#1f2933"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{completed}/{self.planned_count}")

    class CameraWorker(QThread):
        """在独立线程中读取摄像头并执行所有视觉推理。

        ``frame_ready`` 只传显示画面，``monitor_state_ready`` 传业务状态。
        Qt 控件只能在主线程修改，所以本线程不直接访问界面组件。
        """

        frame_ready = Signal(QImage, int)
        error_ready = Signal(str)
        hand_status_ready = Signal(str)
        monitor_state_ready = Signal(object)

        def __init__(
            self,
            camera_spec: CameraSourceSpec,
            config: MonitorConfig,
            model_path: str | None,
            conf: float,
            detect_interval: int,
            enable_hands: bool,
            hand_model: str,
            hand_interval: int,
            action_options: ActionMonitorOptions,
        ):
            super().__init__()
            self.camera_spec = camera_spec
            self.config = config
            self.model_path = model_path
            self.conf = conf
            self.detect_interval = max(0, detect_interval)
            self.enable_hands = enable_hands
            self.hand_model = hand_model
            self.hand_interval = max(1, hand_interval)
            self.action_options = action_options
            self._running = True

        def stop(self):
            """请求摄像头线程停止。"""

            self._running = False

        def run(self):
            try:
                frame_source = create_frame_source(self.camera_spec)
            except (RuntimeError, NotImplementedError) as exc:
                self.error_ready.emit(str(exc))
                return

            is_video_file = (
                self.camera_spec.backend == "opencv"
                and Path(self.camera_spec.source).is_file()
            )
            # 离线视频要按源时间戳控制播放速度；实时流由取流后端自行提供最新帧。
            # 视频默认隔 3 帧跑一次 YOLO，现场摄像头默认每个新帧都检测。
            detection_interval = self.detect_interval or (3 if is_video_file else 1)
            playback_wall_started: float | None = None
            playback_video_started_ms: int | None = None
            last_inference_timestamp_ms: int | None = None

            yolo_model = None
            state_machine = None
            if self.model_path:
                try:
                    from ultralytics import YOLO

                    if not Path(self.model_path).exists():
                        raise FileNotFoundError(f"找不到 YOLO 模型文件：{self.model_path}")
                    if not has_any_roi(self.config):
                        raise ValueError("SOP 配置缺少孔位 ROI，无法启用实时检测。")
                    yolo_model = YOLO(str(self.model_path))
                    state_machine = SopStateMachine(self.config)
                    self.monitor_state_ready.emit({"status": "监控中"})
                except Exception as exc:  # noqa: BLE001 - 模型异常时仍保留摄像头预览。
                    self.error_ready.emit(f"YOLO 检测启动失败：{exc}")

            hand_detector = None
            if self.enable_hands:
                try:
                    hand_detector = MediaPipeHandDetector(model_path=self.hand_model)
                    self.hand_status_ready.emit("未检测")
                except Exception as exc:  # noqa: BLE001 - 现场客户端需要把启动错误展示到界面。
                    self.hand_status_ready.emit("手部模型异常")
                    self.error_ready.emit(f"手部监控启动失败：{exc}")

            action_monitor = None
            action_alarm_active = False
            latest_action_result = None
            # 连续动作模型和 YOLO/SOP 是两条独立判定链。动作模型启动失败时，
            # 孔位装配监控仍可继续，反之亦然。
            if self.action_options.enabled:
                try:
                    from sop_monitor.action_runtime import ActionFusionMonitor

                    action_monitor = ActionFusionMonitor(
                        rgb_model_path=self.action_options.rgb_model or "",
                        flow_model_path=self.action_options.flow_model or "",
                        device=self.action_options.device,
                        rgb_weight=self.action_options.rgb_weight,
                        threshold=self.action_options.threshold,
                        clear_threshold=self.action_options.clear_threshold,
                        interval_seconds=self.action_options.interval_seconds,
                        vote_window=self.action_options.vote_window,
                        trigger_votes=self.action_options.trigger_votes,
                        clear_windows=self.action_options.clear_windows,
                    )
                    self.monitor_state_ready.emit({"status": "监控中", "action_ready": True})
                except Exception as exc:  # noqa: BLE001 - 动作模型失败时仍保留YOLO和画面。
                    self.error_ready.emit(f"连续动作模型启动失败：{exc}")

            try:
                frame_index = 0
                last_observation = None
                part_box_cache = {}
                latest_forbidden_detection = None
                while self._running:
                    ok, frame = frame_source.read()
                    if not ok:
                        if frame_source.is_finished():
                            finish_events = []
                            if state_machine and last_observation is not None:
                                finish_events = state_machine.finish(last_observation)
                            final_status = "完成" if state_machine and state_machine.completed else "视频结束"
                            # 跨线程只发送普通字典和事件对象，界面层据此刷新表格、
                            # 计数和异常列表，不在这里直接操作任何 Qt 控件。
                            self.monitor_state_ready.emit({
                                "status": final_status,
                                "active_region_id": (
                                    state_machine.active_region.region_id
                                    if state_machine and state_machine.active_region else None
                                ),
                                "active_hole_id": (
                                    state_machine.expected_step.hole_id
                                    if state_machine and state_machine.expected_step else None
                                ),
                                "step_phase": state_machine.step_phase if state_machine else "等待零件",
                                "step_started_timestamp_ms": (
                                    state_machine.step_started_timestamp_ms if state_machine else None
                                ),
                                "events": finish_events,
                                "installed_hole_keys": (
                                    sorted(state_machine.confirmed_installed_holes)
                                    if state_machine else []
                                ),
                                "timestamp_ms": last_observation.timestamp_ms if last_observation else 0,
                            })
                            break
                        self.error_ready.emit("读取摄像头画面失败，等待新帧")
                        self.msleep(30)
                        continue
                    if frame is None:
                        self.msleep(30)
                        continue
                    frame_index += 1
                    frame_timestamp_ms = frame_source.timestamp_ms()
                    detections = []
                    events = []
                    active_region_id = self.config.regions[0].region_id if self.config.regions else None
                    active_hole_id = None
                    active_roi = None

                    if yolo_model and state_machine:
                        try:
                            source_frame_changed = frame_timestamp_ms != last_inference_timestamp_ms
                            interval_due = (frame_index - 1) % detection_interval == 0
                            should_detect = (
                                not state_machine.completed
                                and source_frame_changed
                                and (interval_due if is_video_file else True)
                            )
                            if should_detect:
                                # 零件用于确认落位，L 型工具用于确认紧固过程。
                                detections = predict_frame(
                                    yolo_model,
                                    self.config,
                                    frame,
                                    min(
                                        self.conf,
                                        self.config.tool_confidence_threshold,
                                        self.config.display_forbidden_tool_confidence_threshold,
                                    ),
                                    target_classes={
                                        "installed_part",
                                        self.config.tool_class,
                                        self.config.forbidden_tool_class,
                                    },
                                )
                                last_observation = FrameObservation(
                                    frame_index=frame_index,
                                    detections=detections,
                                    timestamp_ms=frame_timestamp_ms,
                                )
                                events = state_machine.update(last_observation)
                                last_inference_timestamp_ms = frame_timestamp_ms
                                # 红框与报警状态保持一致，避免弹簧等短暂误检只画框却不报警。
                                if state_machine.forbidden_alarm_active:
                                    forbidden_detections = [
                                        detection
                                        for detection in detections
                                        if detection.part_type == self.config.forbidden_tool_class
                                        and detection.confidence
                                        >= self.config.display_forbidden_tool_confidence_threshold
                                    ]
                                    latest_forbidden_detection = max(
                                        forbidden_detections,
                                        key=lambda item: item.confidence,
                                        default=None,
                                    )
                                else:
                                    latest_forbidden_detection = None

                                confirmed_holes = state_machine.confirmed_installed_holes
                                for detection in detections:
                                    key = (detection.region_id, detection.hole_id)
                                    if (
                                        detection.part_type == "installed_part"
                                        and detection.bbox is not None
                                        and detection.confidence >= self.config.confidence_threshold
                                        and key in confirmed_holes
                                    ):
                                        # 绿框只展示稳定确认后的零件，短时空孔误检不进入画面。
                                        part_box_cache[key] = (detection, frame_timestamp_ms)
                                stale_keys = [
                                    key
                                    for key, (_, detected_at_ms) in part_box_cache.items()
                                    if frame_timestamp_ms - detected_at_ms > 800
                                ]
                                for key in stale_keys:
                                    del part_box_cache[key]
                            active_region = state_machine.active_region
                            expected = state_machine.expected_step
                            active_region_id = active_region.region_id if active_region else None
                            active_hole_id = expected.hole_id if expected else None
                            active_roi = expected.roi if expected else None
                            self.monitor_state_ready.emit({
                                "status": (
                                    "完成" if state_machine.completed
                                    else "异常" if state_machine.forbidden_alarm_active or action_alarm_active
                                    else "监控中"
                                ),
                                "active_region_id": active_region_id,
                                "active_hole_id": active_hole_id,
                                "step_phase": state_machine.step_phase,
                                "step_started_timestamp_ms": state_machine.step_started_timestamp_ms,
                                "events": events,
                                "installed_hole_keys": sorted(state_machine.confirmed_installed_holes),
                                "frame_index": frame_index,
                                "timestamp_ms": frame_timestamp_ms,
                            })
                        except Exception as exc:  # noqa: BLE001 - 单帧推理异常不应直接关闭客户端。
                            self.error_ready.emit(f"YOLO 检测异常：{exc}")
                            yolo_model = None
                            state_machine = None

                    if action_monitor:
                        try:
                            action_result = action_monitor.process_frame(
                                frame,
                                frame_timestamp_ms,
                            )
                            if action_result is not None:
                                latest_action_result = action_result
                                action_alarm_active = action_result.alarm_active
                                self.monitor_state_ready.emit({
                                    "status": "异常" if action_alarm_active else (
                                        "完成" if state_machine and state_machine.completed else "监控中"
                                    ),
                                    "active_region_id": active_region_id,
                                    "active_hole_id": active_hole_id,
                                    "step_phase": (
                                        state_machine.step_phase if state_machine else "等待零件"
                                    ),
                                    "step_started_timestamp_ms": (
                                        state_machine.step_started_timestamp_ms
                                        if state_machine else None
                                    ),
                                    "events": [],
                                    "installed_hole_keys": (
                                        sorted(state_machine.confirmed_installed_holes)
                                        if state_machine else []
                                    ),
                                    "timestamp_ms": frame_timestamp_ms,
                                    "action_alarm_active": action_alarm_active,
                                    "action_event_started": action_result.event_started,
                                    "action_event_ended": action_result.event_ended,
                                    "action_event_count": action_result.event_count,
                                    "action_probability": action_result.fused_probability,
                                    "action_rgb_probability": action_result.rgb_probability,
                                    "action_flow_probability": action_result.flow_probability,
                                    "action_roi": action_result.active_roi,
                                })
                        except Exception as exc:  # noqa: BLE001 - 单次动作异常不关闭客户端。
                            self.error_ready.emit(f"连续动作检测异常：{exc}")
                            action_monitor = None
                            action_alarm_active = False

                    if active_roi is None:
                        active_roi = self._default_active_roi()
                    draw_visible_detection_boxes(
                        frame,
                        [detection for detection, _ in part_box_cache.values()]
                        + ([latest_forbidden_detection] if latest_forbidden_detection else []),
                        forbidden_tool_class=self.config.forbidden_tool_class,
                    )
                    if latest_action_result and latest_action_result.alarm_active:
                        banner_bottom = max(82, frame.shape[0] - 12)
                        banner_top = banner_bottom - 70
                        banner_right = min(510, frame.shape[1] - 12)
                        cv2.rectangle(
                            frame,
                            (12, banner_top),
                            (banner_right, banner_bottom),
                            (0, 0, 190),
                            -1,
                        )
                        cv2.putText(
                            frame,
                            "FILING ACTION ALARM",
                            (26, banner_top + 31),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.78,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"ROI {latest_action_result.active_roi}  "
                            f"score {latest_action_result.fused_probability:.2f}",
                            (26, banner_top + 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (255, 255, 255),
                            2,
                        )
                    if hand_detector and frame_index % self.hand_interval == 0:
                        try:
                            hands = hand_detector.detect(frame, timestamp_ms=frame_timestamp_ms)
                            near_active_roi = any_hand_near_roi(
                                hands,
                                active_roi,
                                frame.shape[1],
                                frame.shape[0],
                            )
                            draw_hand_overlay(frame, hands, near_active_roi)
                            if near_active_roi:
                                self.hand_status_ready.emit("靠近区域")
                            elif hands:
                                self.hand_status_ready.emit("手部跟踪")
                            else:
                                self.hand_status_ready.emit("未检测")
                        except Exception as exc:  # noqa: BLE001 - 避免手部展示异常导致客户端退出。
                            self.hand_status_ready.emit("检测异常")
                            self.error_ready.emit(f"手部检测异常：{exc}")
                            hand_detector.close()
                            hand_detector = None
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_height, frame_width, channels = rgb.shape
                    bytes_per_line = channels * frame_width
                    image = QImage(
                        rgb.data,
                        frame_width,
                        frame_height,
                        bytes_per_line,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(image, frame_timestamp_ms)
                    if is_video_file:
                        if playback_wall_started is None:
                            playback_wall_started = time.monotonic()
                            playback_video_started_ms = frame_timestamp_ms
                        target_elapsed = (frame_timestamp_ms - (playback_video_started_ms or 0)) / 1000
                        sleep_seconds = target_elapsed - (time.monotonic() - playback_wall_started)
                        if sleep_seconds > 0:
                            self.msleep(max(1, int(round(sleep_seconds * 1000))))
                    else:
                        self.msleep(5)
            finally:
                if hand_detector:
                    hand_detector.close()
                frame_source.release()

        def _default_active_roi(self) -> tuple[float, float, float, float] | None:
            if not self.config.regions or not self.config.regions[0].steps:
                return None
            return self.config.regions[0].steps[0].roi

    class SopDesktopWindow(QMainWindow):
        """现场工位使用的 AI SOP 桌面主窗口。"""

        def __init__(self, config: MonitorConfig, export_options: ClientExportOptions):
            super().__init__()
            self.config = config
            self.export_options = export_options
            self.camera_worker: CameraWorker | None = None
            self.latest_image: QImage | None = None
            self.camera_active = False
            self.step_rows: list[tuple[str, str]] = []
            self.completed_keys: set[tuple[str, str]] = set()
            self.abnormal_keys: set[tuple[str, str]] = set()
            self.installed_keys: set[tuple[str, str]] = set()
            self.abnormal_count = 0
            self.action_event_count_seen = 0
            self.stat_values: dict[str, QLabel] = {}
            self._export_writer = None
            self._next_export_timestamp_ms: float | None = None
            self._last_export_progress_bucket = -1

            self.setWindowTitle("AI SOP 监控台")
            self.resize(1440, 900)
            self.setMinimumSize(1180, 720)
            self._build_ui()
            self._apply_style()
            if self.export_options.enabled:
                self._prepare_client_export()
            self.start_camera()

        def _build_ui(self):
            root = QWidget()
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(14, 10, 14, 10)
            root_layout.setSpacing(8)
            self.setCentralWidget(root)

            topbar = QHBoxLayout()
            topbar.setSpacing(8)
            title = QLabel("AI SOP 监控台")
            title.setObjectName("title")
            self.region_label = self._summary_card("当前区域", self.config.regions[0].region_id if self.config.regions else "-")
            self.hole_label = self._summary_card(
                "当前孔位",
                self.config.regions[0].steps[0].hole_id if self.config.regions and self.config.regions[0].steps else "-",
            )
            self.status_label = self._summary_card("状态", "待机")

            topbar.addWidget(title, 1)
            topbar.addWidget(self.region_label)
            topbar.addWidget(self.hole_label)
            topbar.addWidget(self.status_label)
            root_layout.addLayout(topbar)

            body = QHBoxLayout()
            body.setSpacing(10)
            root_layout.addLayout(body, 1)

            workbench = self._panel("装配画面")
            workbench_layout = workbench.layout()
            workbench_layout.setContentsMargins(10, 8, 10, 10)
            workbench_head = QHBoxLayout()
            workbench_head.setSpacing(8)
            toolbar = QHBoxLayout()
            toolbar.setSpacing(6)
            self.start_btn = self._toolbar_button("开始监控")
            self.pause_btn = self._toolbar_button("暂停监控")
            self.resume_btn = self._toolbar_button("恢复监控")
            self.finish_btn = self._toolbar_button("结束/复位")
            self.ack_btn = self._toolbar_button("异常确认")
            self.camera_btn = self._toolbar_button("关闭摄像头")
            for button in [self.start_btn, self.pause_btn, self.resume_btn, self.finish_btn, self.ack_btn]:
                button.clicked.connect(self.mark_control_pending)
                toolbar.addWidget(button)
            self.camera_btn.clicked.connect(self.toggle_camera)
            toolbar.addWidget(self.camera_btn)
            toolbar.addStretch(1)
            workbench_head.addLayout(toolbar, 1)
            workbench_layout.addLayout(workbench_head)

            self.video_label = QLabel("摄像头连接中")
            self.video_label.setObjectName("videoLabel")
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setMinimumHeight(610)
            self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            workbench_layout.addWidget(self.video_label, 1)
            body.addWidget(workbench, 7)

            side_panel = QFrame()
            side_panel.setObjectName("sidePanel")
            side_layout = QVBoxLayout(side_panel)
            side_layout.setContentsMargins(0, 0, 0, 0)
            side_layout.setSpacing(8)
            body.addWidget(side_panel, 3)

            sop_panel = self._panel("区域 SOP")
            sop_layout = sop_panel.layout()
            sop_panel.setMinimumWidth(420)
            self.sop_table = QTableWidget()
            self.sop_table.setObjectName("sopTable")
            self.sop_table.setColumnCount(6)
            self.sop_table.setHorizontalHeaderLabels(["序号", "步骤名称", "当前孔位", "下一步动作", "结果", "耗时"])
            self.sop_table.verticalHeader().setVisible(False)
            self.sop_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.sop_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.sop_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.sop_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.sop_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.sop_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.sop_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.sop_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self.sop_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self.sop_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            self.sop_table.horizontalHeader().setMinimumSectionSize(44)
            self.sop_table.setAlternatingRowColors(True)
            self._populate_sop_table()
            sop_layout.addWidget(self.sop_table)
            side_layout.addWidget(sop_panel, 1)

            stats_panel = self._production_stats_panel()
            side_layout.addWidget(stats_panel)
            self.hand_metric = self._metric_card("手部状态", "画面稳定")
            side_layout.addWidget(self.hand_metric)

            event_panel = self._panel("异常记录")
            event_panel.setMaximumHeight(118)
            event_layout = event_panel.layout()
            self.event_list = QListWidget()
            self.event_list.setObjectName("eventList")
            self.event_list.addItem("暂无异常")
            event_layout.addWidget(self.event_list)
            root_layout.addWidget(event_panel)

        def _summary_card(self, label: str, value: str) -> QFrame:
            card = QFrame()
            card.setObjectName("summaryCard")
            layout = QHBoxLayout(card)
            layout.setContentsMargins(12, 6, 12, 6)
            layout.setSpacing(8)
            label_widget = QLabel(label)
            label_widget.setObjectName("summaryLabel")
            value_widget = QLabel(value)
            value_widget.setObjectName("summaryValue")
            layout.addWidget(label_widget)
            layout.addWidget(value_widget)
            return card

        def _metric_card(self, label: str, value: str) -> QFrame:
            card = QFrame()
            card.setObjectName("metricCard")
            layout = QHBoxLayout(card)
            layout.setContentsMargins(10, 7, 10, 7)
            layout.setSpacing(8)
            label_widget = QLabel(label)
            label_widget.setObjectName("metricLabel")
            value_widget = QLabel(value)
            value_widget.setObjectName("metricValue")
            layout.addWidget(label_widget)
            layout.addStretch(1)
            layout.addWidget(value_widget)
            return card

        def _production_stats_panel(self) -> QFrame:
            panel = QFrame()
            panel.setObjectName("statsPanel")
            layout = QHBoxLayout(panel)
            layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(10)

            count_layout = QGridLayout()
            count_layout.setSpacing(6)
            count_layout.addWidget(self._stat_cell("加工量", "0", "total"), 0, 0)
            count_layout.addWidget(self._stat_cell("正常", "0", "normal"), 1, 0)
            count_layout.addWidget(self._stat_cell("异常", "0", "abnormal"), 2, 0)
            layout.addLayout(count_layout, 1)

            planned_count = sum(len(region.steps) for region in self.config.regions)
            self.stats_pie = PieChartWidget(completed_count=0, planned_count=planned_count)
            layout.addWidget(self.stats_pie)
            return panel

        def _stat_cell(self, label: str, value: str, key: str) -> QFrame:
            cell = QFrame()
            cell.setObjectName("statCell")
            layout = QHBoxLayout(cell)
            layout.setContentsMargins(8, 5, 8, 5)
            label_widget = QLabel(label)
            label_widget.setObjectName("metricLabel")
            value_widget = QLabel(value)
            value_widget.setObjectName("metricValue")
            self.stat_values[key] = value_widget
            layout.addWidget(label_widget)
            layout.addStretch(1)
            layout.addWidget(value_widget)
            return cell

        def _panel(self, title: str) -> QFrame:
            panel = QFrame()
            panel.setObjectName("panel")
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(10, 8, 10, 10)
            layout.setSpacing(6)
            title_label = QLabel(title)
            title_label.setObjectName("panelTitle")
            layout.addWidget(title_label)
            return panel

        def _toolbar_button(self, text: str) -> QPushButton:
            button = QPushButton(text)
            button.setObjectName("toolbarButton")
            button.setMinimumHeight(30)
            return button

        def _populate_sop_table(self):
            steps = [
                (region.region_id, step.hole_id)
                for region in self.config.regions
                for step in region.steps
            ]
            self.step_rows = steps
            self.sop_table.setRowCount(len(steps))
            for row, (region_id, hole_id) in enumerate(steps):
                status = "等待零件" if row == 0 else "等待"
                next_action = "放入零件" if row == 0 else "等待前序步骤"
                self._set_sop_row(
                    row,
                    str(row + 1),
                    f"步骤 {row + 1}",
                    f"{region_id}-{hole_id}",
                    next_action,
                    status,
                    "-",
                )

        def _set_sop_row(
            self,
            row: int,
            index: str,
            step_name: str,
            hole_id: str,
            next_action: str,
            result: str,
            duration: str,
        ):
            """更新 SOP 表格单行，后续真实监控逻辑会复用这里刷新状态和耗时。"""

            self.sop_table.setItem(row, 0, self._table_item(index, result))
            self.sop_table.setItem(row, 1, self._table_item(step_name, result))
            self.sop_table.setItem(row, 2, self._table_item(hole_id, result))
            self.sop_table.setItem(row, 3, self._table_item(next_action, result))
            self.sop_table.setItem(row, 4, self._table_item(result, result))
            self.sop_table.setItem(row, 5, self._table_item(duration, result))
            self.sop_table.setRowHeight(row, 34)

        def _table_item(self, text: str, status: str) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            colors = {
                "当前": (QColor("#fff7d6"), QColor("#8a5a00")),
                "等待零件": (QColor("#fff7d6"), QColor("#8a5a00")),
                "紧固中": (QColor("#dbeafe"), QColor("#1d4ed8")),
                "完成": (QColor("#dcfce7"), QColor("#166534")),
                "异常": (QColor("#fee2e2"), QColor("#991b1b")),
                "等待": (QColor("#ffffff"), QColor("#52606d")),
            }
            background, foreground = colors.get(status, colors["等待"])
            if text == status:
                item.setBackground(background)
                item.setForeground(foreground)
            else:
                item.setForeground(QColor("#1f2933"))
            return item

        def mark_control_pending(self):
            self._set_summary_value(self.status_label, "待接入")

        def _prepare_client_export(self):
            """初始化固定尺寸的客户端界面视频编码器。"""

            output_path = Path(self.export_options.output_path or "").resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.setFixedSize(self.export_options.width, self.export_options.height)
            self._export_writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.export_options.fps,
                (self.export_options.width, self.export_options.height),
            )
            if not self._export_writer.isOpened():
                self._export_writer = None
                raise RuntimeError(f"无法创建客户端演示视频：{output_path}")

        def _capture_client_frame(self, timestamp_ms: int):
            """按源视频时间采样当前完整界面并写入演示视频。"""

            if self._export_writer is None:
                return
            frame_period_ms = 1000.0 / self.export_options.fps
            if self._next_export_timestamp_ms is None:
                self._next_export_timestamp_ms = float(timestamp_ms)
            if timestamp_ms + 0.5 < self._next_export_timestamp_ms:
                return

            canvas = QImage(
                self.export_options.width,
                self.export_options.height,
                QImage.Format.Format_RGB888,
            )
            canvas.fill(QColor("#f4f6f8"))
            self.render(canvas)
            raw = np.frombuffer(
                canvas.bits(),
                dtype=np.uint8,
                count=canvas.sizeInBytes(),
            )
            rgb = raw.reshape(canvas.height(), canvas.bytesPerLine())[
                :, : canvas.width() * 3
            ].reshape(canvas.height(), canvas.width(), 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            while self._next_export_timestamp_ms <= timestamp_ms + 0.5:
                self._export_writer.write(bgr)
                self._next_export_timestamp_ms += frame_period_ms
            progress_bucket = timestamp_ms // 10000
            if progress_bucket > self._last_export_progress_bucket:
                self._last_export_progress_bucket = progress_bucket
                minutes, seconds = divmod(timestamp_ms // 1000, 60)
                print(f"客户端视频导出进度：{minutes:02d}:{seconds:02d}")

        def _release_client_export(self):
            """完成并关闭客户端演示视频。"""

            if self._export_writer is None:
                return
            self._export_writer.release()
            self._export_writer = None
            print(
                "客户端演示视频："
                f"{Path(self.export_options.output_path or '').resolve()}"
            )

        def start_camera(self):
            if self.camera_active:
                return
            self.action_event_count_seen = 0
            self.camera_worker = CameraWorker(
                camera_spec,
                self.config,
                model_path,
                conf,
                detect_interval,
                enable_hands,
                hand_model,
                hand_interval,
                action_options,
            )
            self.camera_worker.frame_ready.connect(self.update_frame)
            self.camera_worker.error_ready.connect(self.show_camera_error)
            self.camera_worker.hand_status_ready.connect(self.update_hand_status)
            self.camera_worker.monitor_state_ready.connect(self.update_monitor_state)
            self.camera_worker.finished.connect(self.on_camera_finished)
            self.camera_active = True
            self.camera_btn.setText("关闭摄像头")
            self.camera_worker.start()

        def stop_camera(self):
            if not self.camera_worker:
                return
            self.camera_worker.stop()
            self.camera_worker.wait(1500)
            self.camera_worker = None
            self.camera_active = False
            self.camera_btn.setText("打开摄像头")
            self.video_label.setText("摄像头已关闭")
            self.video_label.setPixmap(QPixmap())

        def toggle_camera(self):
            if self.camera_active:
                self.stop_camera()
                return
            self.start_camera()

        def update_frame(self, image: QImage, timestamp_ms: int):
            self.latest_image = image
            self._display_frame(image)
            self._capture_client_frame(timestamp_ms)

        def _display_frame(self, image: QImage):
            """按画面区域等比例显示最新帧。"""

            pixmap = QPixmap.fromImage(image)
            scaled = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.video_label.setPixmap(scaled)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self.latest_image:
                self._display_frame(self.latest_image)

        def show_camera_error(self, message: str):
            self.video_label.setText(message)
            self._set_summary_value(self.status_label, "摄像头异常")

        def update_monitor_state(self, payload: object):
            """根据后台检测线程返回的状态刷新顶部、SOP 表格和异常记录。"""

            if not isinstance(payload, dict):
                return

            status = str(payload.get("status", "监控中"))
            active_region_id = payload.get("active_region_id")
            active_hole_id = payload.get("active_hole_id")
            timestamp_ms = int(payload.get("timestamp_ms", 0) or 0)
            step_phase = str(payload.get("step_phase", "等待零件"))
            started_timestamp = payload.get("step_started_timestamp_ms")
            started_timestamp_ms = int(started_timestamp) if started_timestamp is not None else None
            if "installed_hole_keys" in payload:
                self.installed_keys = {
                    (str(region_id), str(hole_id))
                    for region_id, hole_id in payload.get("installed_hole_keys", [])
                }

            if active_region_id:
                self._set_summary_value(self.region_label, str(active_region_id))
            if active_hole_id:
                self._set_summary_value(self.hole_label, str(active_hole_id))
            self._set_summary_value(self.status_label, status)

            for event in payload.get("events", []):
                if event.event_type == EventType.STEP_COMPLETED:
                    key = (event.region_id, event.expected_hole_id or "")
                    self.completed_keys.add(key)
                    if key not in self.abnormal_keys:
                        self._update_step_result(key, "完成", event.duration_ms)
                elif event.event_type in {EventType.ORDER_ERROR, EventType.MISSING_PART}:
                    self.abnormal_count += 1
                    if event.event_type == EventType.ORDER_ERROR and event.observed_hole_id:
                        key = (event.region_id, event.observed_hole_id)
                        next_action = "顺序错误"
                    else:
                        key = (event.region_id, event.expected_hole_id or "")
                        next_action = "漏装检查"
                    self.abnormal_keys.add(key)
                    self._update_step_result(key, "异常", event.duration_ms, next_action)
                    self._append_abnormal_event(event.message)
                elif event.event_type == EventType.FORBIDDEN_TOOL:
                    self.abnormal_count += 1
                    self._append_abnormal_event(event.message)

            action_event_count = int(payload.get("action_event_count", 0) or 0)
            if action_event_count > self.action_event_count_seen:
                new_event_count = action_event_count - self.action_event_count_seen
                self.abnormal_count += new_event_count
                probability = float(payload.get("action_probability", 0.0) or 0.0)
                self._append_abnormal_event(
                    f"检测到疑似违规锉削连续动作（融合置信度{probability:.2f}）。"
                )
                self.action_event_count_seen = action_event_count

            if active_region_id and active_hole_id:
                self._mark_active_step(
                    (str(active_region_id), str(active_hole_id)),
                    timestamp_ms,
                    step_phase,
                    started_timestamp_ms,
                )

            self._refresh_stats()

        def _mark_active_step(
            self,
            active_key: tuple[str, str],
            timestamp_ms: int,
            step_phase: str,
            started_timestamp_ms: int | None,
        ):
            for row, key in enumerate(self.step_rows):
                if key in self.completed_keys or key in self.abnormal_keys:
                    continue
                if key == active_key:
                    is_tightening = step_phase == "紧固中" and started_timestamp_ms is not None
                    elapsed_ms = max(0, timestamp_ms - started_timestamp_ms) if is_tightening else None
                    self._set_sop_row(
                        row,
                        str(row + 1),
                        f"步骤 {row + 1}",
                        f"{key[0]}-{key[1]}",
                        "L 型工具紧固" if is_tightening else "放入零件",
                        "紧固中" if is_tightening else "等待零件",
                        self._format_duration(elapsed_ms) if elapsed_ms is not None else "-",
                    )
                else:
                    self._set_sop_row(
                        row,
                        str(row + 1),
                        f"步骤 {row + 1}",
                        f"{key[0]}-{key[1]}",
                        "等待前序步骤",
                        "等待",
                        "-",
                    )

        def _update_step_result(
            self,
            key: tuple[str, str],
            result: str,
            duration_ms: int | None,
            next_action: str | None = None,
        ):
            if key not in self.step_rows:
                return
            row = self.step_rows.index(key)
            duration = self._format_duration(duration_ms) if duration_ms is not None else "-"
            action = next_action or ("已完成" if result == "完成" else "处理异常")
            self._set_sop_row(row, str(row + 1), f"步骤 {row + 1}", f"{key[0]}-{key[1]}", action, result, duration)

        @staticmethod
        def _format_duration(duration_ms: int) -> str:
            """把毫秒耗时格式化为表格中的秒数。"""

            return f"{max(0, duration_ms) / 1000:.1f}s"

        def _append_abnormal_event(self, message: str):
            if self.event_list.count() == 1 and self.event_list.item(0).text() == "暂无异常":
                self.event_list.clear()
            occurred_at = datetime.now().strftime("%H:%M:%S")
            self.event_list.insertItem(0, f"{occurred_at}  {message}")
            self.event_list.scrollToTop()

        def _refresh_stats(self):
            normal_count = len(self.installed_keys)
            self.stat_values["total"].setText(str(normal_count))
            self.stat_values["normal"].setText(str(normal_count))
            self.stat_values["abnormal"].setText(str(self.abnormal_count))
            self.stats_pie.completed_count = normal_count
            self.stats_pie.update()

        def update_hand_status(self, status: str):
            self._set_summary_value(self.hand_metric, status)

        def on_camera_finished(self):
            self.camera_active = False
            self.camera_btn.setText("打开摄像头")
            if self.export_options.enabled:
                self._release_client_export()
                QApplication.instance().quit()

        def closeEvent(self, event):
            self.stop_camera()
            self._release_client_export()
            event.accept()

        def _set_summary_value(self, card: QFrame, value: str):
            labels = card.findChildren(QLabel)
            if len(labels) >= 2:
                labels[1].setText(value)

        def _apply_style(self):
            self.setStyleSheet("""
                QWidget {
                    background: #f4f6f8;
                    color: #1f2933;
                    font-family: "PingFang SC", "Microsoft YaHei", Arial;
                    font-size: 13px;
                }
                #title {
                    font-size: 22px;
                    font-weight: 700;
                }
                #subtitle, #summaryLabel, #metricLabel {
                    color: #697586;
                    font-size: 12px;
                }
                #summaryCard, #metricCard, #panel, #statsPanel {
                    background: #ffffff;
                    border: 1px solid #d9e1ea;
                    border-radius: 8px;
                }
                #statCell {
                    background: #f8fafc;
                    border: 1px solid #e5ebf2;
                    border-radius: 6px;
                }
                #sidePanel {
                    background: transparent;
                    border: 0;
                }
                #summaryCard {
                    max-height: 42px;
                }
                #summaryValue {
                    font-size: 17px;
                    font-weight: 700;
                }
                #panelTitle {
                    font-size: 15px;
                    font-weight: 700;
                }
                #toolbarButton {
                    background: #253142;
                    color: #ffffff;
                    border: 0;
                    border-radius: 6px;
                    padding: 0 11px;
                    font-weight: 600;
                }
                #toolbarButton:hover {
                    background: #344256;
                }
                #videoLabel {
                    background: #111827;
                    color: #d5dce7;
                    border-radius: 8px;
                    font-size: 20px;
                }
                #metricValue {
                    font-size: 17px;
                    font-weight: 700;
                }
                #metricCard {
                    max-height: 42px;
                }
                #statsPanel {
                    min-height: 118px;
                    max-height: 138px;
                }
                QListWidget {
                    background: #ffffff;
                    border: 1px solid #e1e7ef;
                    border-radius: 6px;
                    padding: 4px;
                }
                QListWidget::item {
                    padding: 5px 6px;
                    border-radius: 5px;
                }
                QListWidget::item:selected {
                    background: #dbeafe;
                    color: #1f2933;
                }
                #eventList::item {
                    padding: 3px 6px;
                }
                QTableWidget {
                    background: #ffffff;
                    alternate-background-color: #f8fafc;
                    border: 1px solid #e1e7ef;
                    border-radius: 6px;
                    gridline-color: #edf1f5;
                    selection-background-color: transparent;
                    selection-color: #1f2933;
                }
                QTableWidget::item {
                    padding: 3px;
                    border: 0;
                }
                QHeaderView::section {
                    background: #eef2f6;
                    color: #5f6f82;
                    border: 0;
                    border-right: 1px solid #dfe6ee;
                    border-bottom: 1px solid #dfe6ee;
                    padding: 6px 3px;
                    font-weight: 700;
                }
            """)

    app = QApplication(sys.argv)
    window = SopDesktopWindow(config, export_options)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
