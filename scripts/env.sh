#!/usr/bin/env sh

# 项目本地运行环境变量。
# 让深度学习/绘图库缓存写到项目目录，避免写用户目录时出现权限警告。
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$(pwd)/.cache/ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.cache/matplotlib}"

