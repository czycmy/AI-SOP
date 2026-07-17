"""从现场录制视频中抽取训练图片。

默认每 1 秒抽取 1 张图片，输出到 dataset/frames/。文件名包含帧号和时间点，
方便后续人工筛选、删除遮挡严重的图片，并进入标注流程。

如果希望避免 JPG 二次压缩，可以使用 --ext png 输出无损 PNG 图片。注意：PNG
只能避免抽帧保存阶段的压缩损失，无法恢复视频录制时已经损失的画质。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="从视频中抽取数据集图片")
    parser.add_argument("video", help="输入视频路径，例如 dataset/luzhi.mp4。")
    parser.add_argument("--output", default="dataset/frames", help="图片输出目录。")
    parser.add_argument("--interval", type=float, default=1.0, help="抽帧间隔，单位秒。")
    parser.add_argument("--prefix", default="luzhi", help="输出图片文件名前缀。")
    parser.add_argument("--start", type=float, default=0.0, help="开始时间，单位秒。")
    parser.add_argument("--end", type=float, default=None, help="结束时间，单位秒。")
    parser.add_argument("--ext", choices=("jpg", "png"), default="jpg", help="输出图片格式。png 为无损保存。")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG 保存质量，范围 1-100。")
    return parser


def main() -> int:
    """执行抽帧。"""

    args = build_parser().parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output)
    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")
    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0。")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality 必须在 1-100 范围内。")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0.0
    end_time = args.end if args.end is not None else duration
    frame_step = max(1, int(round(args.interval * fps)))
    start_frame = max(0, int(round(args.start * fps)))
    end_frame = min(total_frames - 1, int(round(end_time * fps))) if total_frames else None

    saved = 0
    frame_index = start_frame
    try:
        while end_frame is None or frame_index <= end_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            image_path = output_dir / f"{args.prefix}_f{frame_index:06d}_t{timestamp:07.2f}s.{args.ext}"
            if args.ext == "jpg":
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            else:
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            saved += 1
            frame_index += frame_step
    finally:
        capture.release()

    print(f"video: {video_path}")
    print(f"fps: {fps:.2f}, frames: {total_frames}, duration: {duration:.2f}s")
    print(f"interval: {args.interval:.2f}s")
    print(f"saved: {saved}")
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
