"""摄像头和画面标注工具。

本模块提供 OpenCV 摄像头打开、孔位 ROI 映射、画面覆盖绘制等公共能力。
摄像头实时检测命令和摄像头预览命令都会复用这里的函数。
"""

from __future__ import annotations

from typing import Iterable

from sop_monitor.models import Detection, MonitorConfig, StepSpec


def open_camera(camera_index: int, width: int | None = None, height: int | None = None):
    """打开摄像头，并按需设置采集分辨率。"""

    import cv2

    capture = cv2.VideoCapture(camera_index)
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}。请检查编号、连接和系统权限。")
    return capture


def iter_steps(config: MonitorConfig) -> Iterable[tuple[str, StepSpec]]:
    """按配置顺序遍历所有区域和孔位步骤。"""

    for region in config.regions:
        for step in region.steps:
            yield region.region_id, step


def match_detection_to_hole(
    config: MonitorConfig,
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[str, StepSpec] | None:
    """根据检测框中心点匹配孔位 ROI。

    bbox 是像素坐标 xyxy，StepSpec.roi 是归一化坐标 xyxy。
    """

    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2) / frame_width
    center_y = ((y1 + y2) / 2) / frame_height
    for region_id, step in iter_steps(config):
        if step.roi is None:
            continue
        roi_x1, roi_y1, roi_x2, roi_y2 = step.roi
        if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
            return region_id, step
    return None


def has_any_roi(config: MonitorConfig) -> bool:
    """判断 SOP 配置是否已经标定孔位 ROI。"""

    return any(step.roi is not None for _, step in iter_steps(config))


def draw_monitor_overlay(
    frame,
    config: MonitorConfig,
    detections: list[Detection],
    active_region_id: str | None,
    active_hole_id: str | None,
):
    """在摄像头画面上绘制孔位 ROI、检测框和当前步骤提示。"""

    import cv2

    height, width = frame.shape[:2]
    detected_keys = {(item.region_id, item.hole_id) for item in detections}
    for region_id, step in iter_steps(config):
        if step.roi is None:
            continue
        x1, y1, x2, y2 = normalized_roi_to_pixels(step.roi, width, height)
        is_active = region_id == active_region_id and step.hole_id == active_hole_id
        is_detected = (region_id, step.hole_id) in detected_keys
        color = (255, 128, 0) if is_active else (130, 130, 130)
        if is_detected:
            color = (0, 170, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{region_id}:{step.hole_id}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    for detection in detections:
        if detection.bbox is None:
            continue
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 90), 2)
        cv2.putText(
            frame,
            f"{detection.hole_id} {detection.confidence:.2f}",
            (x1, min(height - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 130, 70),
            2,
            cv2.LINE_AA,
        )


def normalized_roi_to_pixels(
    roi: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """把归一化 ROI 转成像素坐标。"""

    x1, y1, x2, y2 = roi
    return (
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    )

