"""AI SOP PySide6 桌面客户端。

本文件提供现场部署用的原生桌面界面：自动打开摄像头预览，展示区域 SOP、
装配画面、状态概览和异常记录。当前版本先完成客户端框架和摄像头画面接入，
监控控制按钮只做界面预留，后续再接入真实的开始、暂停、恢复、复位和异常确认逻辑。
"""

from __future__ import annotations

import argparse
import sys

from sop_monitor.config import load_config
from sop_monitor.models import MonitorConfig


def build_parser() -> argparse.ArgumentParser:
    """创建桌面客户端命令行参数。"""

    parser = argparse.ArgumentParser(description="AI SOP PySide6 桌面客户端")
    parser.add_argument("--config", default="configs/sample_sop.json", help="SOP 配置 JSON。")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，内置摄像头通常是 0。")
    parser.add_argument("--width", type=int, default=1280, help="摄像头采集宽度。")
    parser.add_argument("--height", type=int, default=720, help="摄像头采集高度。")
    parser.add_argument("--hands", action="store_true", help="开启 MediaPipe 手部监控展示。")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe 手部模型路径。")
    return parser


def main() -> int:
    """启动 PySide6 桌面客户端。"""

    args = build_parser().parse_args()
    config = load_config(args.config)
    return run_qt_app(config, args.camera, args.width, args.height, args.hands, args.hand_model)


def run_qt_app(
    config: MonitorConfig,
    camera_index: int,
    width: int | None,
    height: int | None,
    enable_hands: bool,
    hand_model: str,
) -> int:
    """延迟导入 PySide6 并启动应用，避免测试环境没有 Qt 时影响核心模块。"""

    try:
        from PySide6.QtCore import Qt, QThread, Signal
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QPushButton,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 PySide6，请先安装依赖：.venv/bin/python -m pip install -r requirements.txt") from exc

    import cv2
    from sop_monitor.hand_detector import MediaPipeHandDetector, any_hand_near_roi, draw_hand_overlay

    class CameraWorker(QThread):
        """在独立线程中读取摄像头，避免阻塞 Qt 主界面。"""

        frame_ready = Signal(QImage)
        error_ready = Signal(str)
        hand_status_ready = Signal(str)

        def __init__(
            self,
            camera_index: int,
            width: int | None,
            height: int | None,
            config: MonitorConfig,
            enable_hands: bool,
            hand_model: str,
        ):
            super().__init__()
            self.camera_index = camera_index
            self.width = width
            self.height = height
            self.config = config
            self.enable_hands = enable_hands
            self.hand_model = hand_model
            self._running = True

        def stop(self):
            """请求摄像头线程停止。"""

            self._running = False

        def run(self):
            capture = cv2.VideoCapture(self.camera_index)
            if self.width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not capture.isOpened():
                self.error_ready.emit(f"无法打开摄像头 {self.camera_index}")
                return

            hand_detector = None
            if self.enable_hands:
                try:
                    hand_detector = MediaPipeHandDetector(model_path=self.hand_model)
                    self.hand_status_ready.emit("未检测")
                except Exception as exc:  # noqa: BLE001 - 现场客户端需要把启动错误展示到界面。
                    self.hand_status_ready.emit("手部模型异常")
                    self.error_ready.emit(f"手部监控启动失败：{exc}")

            try:
                frame_index = 0
                while self._running:
                    ok, frame = capture.read()
                    if not ok:
                        self.error_ready.emit("读取摄像头画面失败")
                        break
                    frame_index += 1
                    if hand_detector:
                        try:
                            hands = hand_detector.detect(frame, timestamp_ms=frame_index * 33)
                            near_active_roi = any_hand_near_roi(
                                hands,
                                self._active_roi(),
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
                    self.frame_ready.emit(image)
                    self.msleep(20)
            finally:
                if hand_detector:
                    hand_detector.close()
                capture.release()

        def _active_roi(self) -> tuple[float, float, float, float] | None:
            if not self.config.regions or not self.config.regions[0].steps:
                return None
            return self.config.regions[0].steps[0].roi

    class SopDesktopWindow(QMainWindow):
        """现场工位使用的 AI SOP 桌面主窗口。"""

        def __init__(self, config: MonitorConfig):
            super().__init__()
            self.config = config
            self.camera_worker: CameraWorker | None = None
            self.latest_image: QImage | None = None
            self.camera_active = False

            self.setWindowTitle("AI SOP 监控台")
            self.resize(1440, 900)
            self.setMinimumSize(1180, 720)
            self._build_ui()
            self._apply_style()
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
            self.region_label = self._summary_card("当前区域", self.config.regions[0].name if self.config.regions else "-")
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
            body.addWidget(workbench, 8)

            side_panel = QFrame()
            side_panel.setObjectName("sidePanel")
            side_layout = QVBoxLayout(side_panel)
            side_layout.setContentsMargins(0, 0, 0, 0)
            side_layout.setSpacing(8)
            body.addWidget(side_panel, 2)

            sop_panel = self._panel("区域 SOP")
            sop_layout = sop_panel.layout()
            sop_panel.setMinimumWidth(300)
            self.sop_list = QListWidget()
            self.sop_list.setObjectName("sopList")
            self._populate_sop_list()
            sop_layout.addWidget(self.sop_list)
            side_layout.addWidget(sop_panel, 1)

            metrics = QGridLayout()
            metrics.setSpacing(6)
            self.done_metric = self._metric_card("完成孔位", "0")
            self.error_metric = self._metric_card("异常次数", "0")
            self.stable_metric = self._metric_card("稳定帧", "0")
            self.hand_metric = self._metric_card("手部状态", "画面稳定")
            metrics.addWidget(self.done_metric, 0, 0)
            metrics.addWidget(self.error_metric, 1, 0)
            metrics.addWidget(self.stable_metric, 2, 0)
            metrics.addWidget(self.hand_metric, 3, 0)
            side_layout.addLayout(metrics)

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

        def _populate_sop_list(self):
            self.sop_list.clear()
            for region in self.config.regions:
                region_item = QListWidgetItem(f"{region.name} · {region.region_id}")
                region_item.setData(Qt.ItemDataRole.UserRole, "region")
                self.sop_list.addItem(region_item)
                for step in region.steps:
                    self.sop_list.addItem(f"步骤 {step.step}  {step.hole_id}  已装确认")

        def mark_control_pending(self):
            self._set_summary_value(self.status_label, "待接入")

        def start_camera(self):
            if self.camera_active:
                return
            self.camera_worker = CameraWorker(camera_index, width, height, self.config, enable_hands, hand_model)
            self.camera_worker.frame_ready.connect(self.update_frame)
            self.camera_worker.error_ready.connect(self.show_camera_error)
            self.camera_worker.hand_status_ready.connect(self.update_hand_status)
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

        def update_frame(self, image: QImage):
            self.latest_image = image
            pixmap = QPixmap.fromImage(image)
            scaled = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self.latest_image:
                self.update_frame(self.latest_image)

        def show_camera_error(self, message: str):
            self.video_label.setText(message)
            self._set_summary_value(self.status_label, "摄像头异常")

        def update_hand_status(self, status: str):
            self._set_summary_value(self.hand_metric, status)

        def on_camera_finished(self):
            self.camera_active = False
            self.camera_btn.setText("打开摄像头")

        def closeEvent(self, event):
            self.stop_camera()
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
                #summaryCard, #metricCard, #panel {
                    background: #ffffff;
                    border: 1px solid #d9e1ea;
                    border-radius: 8px;
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
            """)

    app = QApplication(sys.argv)
    window = SopDesktopWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
