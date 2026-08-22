"""从完整正常作业视频切出动作分类困难负样本。

本脚本按照固定窗口和步长切分视频，并使用 FFmpeg 重新编码为规范 H.264。
适合从完整正常安装流程中补充放件、调整、双手遮挡和工具经过等非锉削动作。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    """创建视频窗口切分参数。"""

    parser = argparse.ArgumentParser(description="提取动作识别困难负样本")
    parser.add_argument("--source", required=True, help="完整正常作业视频。")
    parser.add_argument("--output", required=True, help="短视频输出目录。")
    parser.add_argument("--window", type=float, default=3.0, help="每段时长，单位秒。")
    parser.add_argument("--stride", type=float, default=3.0, help="相邻片段起点间隔，单位秒。")
    parser.add_argument("--min-last", type=float, default=1.8, help="最后一段允许的最小时长。")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg可执行文件路径。")
    return parser


def main() -> int:
    """读取视频时长并逐段无交叠或滑动切分。"""

    args = build_parser().parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"找不到源视频：{source}")
    if args.window <= 0 or args.stride <= 0:
        raise ValueError("--window 和 --stride 必须大于0。")

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"无法读取视频时长：{source}")
    duration = frame_count / fps

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = 0.0
    index = 1
    while start < duration:
        clip_duration = min(args.window, duration - start)
        if clip_duration < args.min_last:
            break
        output = output_dir / f"other_{index:03d}_{start:07.2f}s.mp4"
        command = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{clip_duration:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        subprocess.run(command, check=True)
        print(f"已输出：{output}")
        start += args.stride
        index += 1
    print(f"完成：共生成 {index - 1} 段，源视频时长 {duration:.2f} 秒。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
