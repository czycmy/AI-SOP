"""AI SOP PySide6 桌面客户端。

本文件提供现场部署用的原生桌面界面：自动打开摄像头预览，展示区域 SOP、
装配画面、状态概览和异常记录。当前版本先完成客户端框架和摄像头画面接入，
监控控制按钮只做界面预留，后续再接入真实的开始、暂停、恢复、复位和异常确认逻辑。
"""

from __future__ import annotations

import argparse
import sys

from sop_monitor.camera_utils import add_camera_source_arguments, open_camera, resolve_camera_source
from sop_monitor.config import load_config
from sop_monitor.models import MonitorConfig


def build_parser() -> argparse.ArgumentParser:
    """创建桌面客户端命令行参数。"""

    parser = argparse.ArgumentParser(description="AI SOP PySide6 桌面客户端")
    parser.add_argument("--config", default="configs/sample_sop.json", help="SOP 配置 JSON。")
    add_camera_source_arguments(parser)
    parser.add_argument("--width", type=int, default=1280, help="摄像头采集宽度。")
    parser.add_argument("--height", type=int, default=720, help="摄像头采集高度。")
    parser.add_argument("--hands", action="store_true", help="开启 MediaPipe 手部监控展示。")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe 手部模型路径。")
    parser.add_argument("--hand-interval", type=int, default=5, help="手部检测间隔帧数，值越大延迟越低但手部刷新越慢。")
    return parser


def main() -> int:
    """启动 PySide6 桌面客户端。"""

    args = build_parser().parse_args()
    config = load_config(args.config)
    camera_source = resolve_camera_source(args)
    return run_qt_app(config, camera_source, args.width, args.height, args.hands, args.hand_model, args.hand_interval)


def run_qt_app(
    config: MonitorConfig,
    camera_source: str,
    width: int | None,
    height: int | None,
    enable_hands: bool,
    hand_model: str,
    hand_interval: int,
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
    from sop_monitor.hand_detector import MediaPipeHandDetector, any_hand_near_roi, draw_hand_overlay

    class PieChartWidget(QWidget):
        """右侧加工统计的小型饼图。"""

        def __init__(self, normal_count: int = 0, abnormal_count: int = 0):
            super().__init__()
            self.normal_count = normal_count
            self.abnormal_count = abnormal_count
            self.setMinimumSize(86, 86)
            self.setMaximumHeight(112)

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            side = min(self.width(), self.height()) - 16
            rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
            total = self.normal_count + self.abnormal_count
            if total <= 0:
                painter.setPen(QPen(QColor("#d9e1ea"), 10))
                painter.drawEllipse(rect)
                painter.setPen(QColor("#52606d"))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "0")
                return

            normal_span = int(360 * 16 * self.normal_count / total)
            abnormal_span = 360 * 16 - normal_span
            painter.setPen(QPen(QColor("#16a34a"), 10))
            painter.drawArc(rect, 90 * 16, -normal_span)
            if abnormal_span:
                painter.setPen(QPen(QColor("#dc2626"), 10))
                painter.drawArc(rect, 90 * 16 - normal_span, -abnormal_span)
            painter.setPen(QColor("#1f2933"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(total))

    class CameraWorker(QThread):
        """在独立线程中读取摄像头，避免阻塞 Qt 主界面。"""

        frame_ready = Signal(QImage)
        error_ready = Signal(str)
        hand_status_ready = Signal(str)

        def __init__(
            self,
            camera_source: str,
            width: int | None,
            height: int | None,
            config: MonitorConfig,
            enable_hands: bool,
            hand_model: str,
            hand_interval: int,
        ):
            super().__init__()
            self.camera_source = camera_source
            self.width = width
            self.height = height
            self.config = config
            self.enable_hands = enable_hands
            self.hand_model = hand_model
            self.hand_interval = max(1, hand_interval)
            self._running = True

        def stop(self):
            """请求摄像头线程停止。"""

            self._running = False

        def run(self):
            try:
                capture = open_camera(self.camera_source, self.width, self.height)
            except RuntimeError as exc:
                self.error_ready.emit(str(exc))
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
                    if hand_detector and frame_index % self.hand_interval == 0:
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
            count_layout.addWidget(self._stat_cell("加工量", "0"), 0, 0)
            count_layout.addWidget(self._stat_cell("正常", "0"), 1, 0)
            count_layout.addWidget(self._stat_cell("异常", "0"), 2, 0)
            layout.addLayout(count_layout, 1)

            self.stats_pie = PieChartWidget(normal_count=0, abnormal_count=0)
            layout.addWidget(self.stats_pie)
            return panel

        def _stat_cell(self, label: str, value: str) -> QFrame:
            cell = QFrame()
            cell.setObjectName("statCell")
            layout = QHBoxLayout(cell)
            layout.setContentsMargins(8, 5, 8, 5)
            label_widget = QLabel(label)
            label_widget.setObjectName("metricLabel")
            value_widget = QLabel(value)
            value_widget.setObjectName("metricValue")
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
            self.sop_table.setRowCount(len(steps))
            for row, (region_id, hole_id) in enumerate(steps):
                status = "当前" if row == 0 else "等待"
                duration = "0.0s" if row == 0 else "-"
                self._set_sop_row(row, str(row + 1), f"步骤 {row + 1}", f"{region_id}-{hole_id}", "装配确认", status, duration)

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

        def start_camera(self):
            if self.camera_active:
                return
            self.camera_worker = CameraWorker(
                camera_source,
                width,
                height,
                self.config,
                enable_hands,
                hand_model,
                hand_interval,
            )
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
                Qt.TransformationMode.FastTransformation,
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
    window = SopDesktopWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
