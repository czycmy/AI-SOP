"""摄像头和画面标注工具。

本模块提供 OpenCV 摄像头打开、孔位 ROI 映射、画面覆盖绘制等公共能力。
摄像头实时检测命令和摄像头预览命令都会复用这里的函数。
"""

from __future__ import annotations

import os
from typing import Iterable

from sop_monitor.models import Detection, MonitorConfig, StepSpec


def add_camera_source_arguments(parser) -> None:
    """为命令行工具添加统一的摄像头来源参数。"""

    parser.add_argument(
        "--camera-backend",
        default="opencv",
        choices=["opencv", "hikvision-sdk"],
        help="摄像头后端：opencv 使用本地/RTSP；hikvision-sdk 预留海康 SDK 低延迟取流。",
    )
    parser.add_argument("--camera", default="0", help="摄像头来源：本地编号、RTSP 地址或视频路径。")
    parser.add_argument("--hikvision-ip", default=None, help="海康摄像头 IP；提供后会自动生成 RTSP 地址。")
    parser.add_argument("--hikvision-user", default="admin", help="海康摄像头账号。")
    parser.add_argument("--hikvision-password", default=None, help="海康摄像头密码。")
    parser.add_argument("--hikvision-port", type=int, default=8000, help="海康 SDK 登录端口，通常是 8000。")
    parser.add_argument("--hikvision-channel", default="101", help="海康 RTSP 通道，主码流通常是 101。")
    parser.add_argument(
        "--hikvision-sdk-dir",
        default="third_party/hikvision",
        help="海康 Windows SDK DLL 目录，使用 hikvision-sdk 后端时需要。",
    )


def resolve_camera_source(args) -> str:
    """根据普通摄像头参数或海康参数得到最终视频源。"""

    if args.hikvision_ip:
        if not args.hikvision_password:
            raise ValueError("使用 --hikvision-ip 时必须提供 --hikvision-password。")
        return build_hikvision_rtsp_url(
            ip=args.hikvision_ip,
            user=args.hikvision_user,
            password=args.hikvision_password,
            channel=args.hikvision_channel,
        )
    return args.camera


def normalize_camera_source(source: int | str) -> int | str:
    """把命令行传入的摄像头来源转换为 OpenCV 可接受的格式。

    纯数字字符串会转成本地摄像头编号；RTSP/HTTP/file path 等保持字符串。
    """

    if isinstance(source, int):
        return source
    text = source.strip()
    if text.isdigit():
        return int(text)
    return text


def is_rtsp_source(source: int | str) -> bool:
    """判断摄像头来源是否为 RTSP 流。"""

    return isinstance(source, str) and source.strip().lower().startswith("rtsp://")


def build_hikvision_rtsp_url(
    ip: str,
    user: str,
    password: str,
    channel: str = "101",
) -> str:
    """根据海康摄像头参数生成 RTSP 地址。"""

    return f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}"


def open_camera(source: int | str, width: int | None = None, height: int | None = None):
    """打开本地摄像头编号或 RTSP 视频流，并按需设置采集分辨率。"""

    import cv2

    camera_source = normalize_camera_source(source)
    if is_rtsp_source(camera_source):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
        )
        capture = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
    else:
        capture = cv2.VideoCapture(camera_source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头/视频流 {source}。请检查编号、RTSP 地址、账号密码和网络。")
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
    for region in config.regions:
        # 区域总 ROI 是第一层过滤，避免模具其他位置的相似目标进入 SOP 判断。
        if region.roi is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = region.roi
            if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
                continue
        for step in region.steps:
            if step.roi is None:
                continue
            roi_x1, roi_y1, roi_x2, roi_y2 = step.roi
            if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                return region.region_id, step
    return None


def match_detection_to_region(
    config: MonitorConfig,
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> str | None:
    """根据检测框中心点匹配总监控区域，供移动中的工具目标使用。"""

    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2) / frame_width
    center_y = ((y1 + y2) / 2) / frame_height
    for region in config.regions:
        if region.roi is None:
            continue
        roi_x1, roi_y1, roi_x2, roi_y2 = region.roi
        if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
            return region.region_id
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
    for region in config.regions:
        if region.roi is None:
            continue
        x1, y1, x2, y2 = normalized_roi_to_pixels(region.roi, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(
            frame,
            f"{region.region_id} MONITOR AREA",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
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


def draw_visible_detection_boxes(
    frame,
    detections: list[Detection],
    part_class: str = "installed_part",
    forbidden_tool_class: str = "forbidden_tool",
) -> None:
    """只绘制已装零件和锉刀框，不显示 ROI、文字或置信度。"""

    import cv2

    height, width = frame.shape[:2]
    for detection in detections:
        if detection.part_type == part_class:
            box_color = (32, 220, 96)
        elif detection.part_type == forbidden_tool_class:
            box_color = (40, 40, 235)
        else:
            continue
        if detection.bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)


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
