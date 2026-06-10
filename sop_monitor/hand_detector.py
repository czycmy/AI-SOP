"""后端手部检测。

本模块使用 MediaPipe Hands 在摄像头帧里检测手部关键点，并提供简单的
ROI 接近判断。当前手部结果只用于画面展示和状态提示，不改变 SOP 状态机判定。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HandObservation:
    """单只手的检测结果。"""

    landmarks: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    score: float | None = None


class MediaPipeHandDetector:
    """MediaPipe Hand Landmarker 检测器封装。"""

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        from pathlib import Path

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到 MediaPipe 手部模型：{self.model_path}。"
                "请先下载 hand_landmarker.task，或使用 --no-hands 关闭手部检测。"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame, timestamp_ms: int | None = None) -> list[HandObservation]:
        """检测一帧 BGR 图像里的手部关键点。"""

        import cv2
        import mediapipe as mp

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if timestamp_ms is None:
            timestamp_ms = self._last_timestamp_ms + 1
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return []

        observations: list[HandObservation] = []
        for index, hand_landmarks in enumerate(result.hand_landmarks):
            points = [
                (
                    landmark.x * width,
                    landmark.y * height,
                )
                for landmark in hand_landmarks.landmark
            ]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            score = None
            if index < len(result.handedness) and result.handedness[index]:
                score = result.handedness[index][0].score
            observations.append(HandObservation(
                landmarks=points,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                score=score,
            ))
        return observations

    def close(self) -> None:
        """释放 MediaPipe 资源。"""

        self._landmarker.close()


def bbox_overlaps_roi(
    bbox: tuple[float, float, float, float],
    roi: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> bool:
    """判断像素 bbox 是否和归一化 ROI 有交集。"""

    x1, y1, x2, y2 = bbox
    roi_x1, roi_y1, roi_x2, roi_y2 = (
        roi[0] * frame_width,
        roi[1] * frame_height,
        roi[2] * frame_width,
        roi[3] * frame_height,
    )
    return not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2)


def any_hand_near_roi(
    hands: list[HandObservation],
    roi: tuple[float, float, float, float] | None,
    frame_width: int,
    frame_height: int,
) -> bool:
    """判断是否有手部框接近指定 ROI。"""

    if roi is None:
        return False
    return any(bbox_overlaps_roi(hand.bbox, roi, frame_width, frame_height) for hand in hands)


def draw_hand_overlay(frame, hands: list[HandObservation], near_active_roi: bool) -> None:
    """在画面上绘制手部关键点、骨架和状态提示。"""

    import cv2

    color = (0, 180, 255) if near_active_roi else (255, 120, 0)
    status = "HAND NEAR ROI" if near_active_roi else "HAND TRACKING"
    if not hands:
        cv2.putText(
            frame,
            "HAND STATUS: CLEAR",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (90, 170, 90),
            2,
            cv2.LINE_AA,
        )
        return

    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for hand in hands:
        x1, y1, x2, y2 = [int(value) for value in hand.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for start, end in bones:
            p1 = tuple(int(value) for value in hand.landmarks[start])
            p2 = tuple(int(value) for value in hand.landmarks[end])
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        for point in hand.landmarks:
            cv2.circle(frame, tuple(int(value) for value in point), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(int(value) for value in point), 4, color, 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        status,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
