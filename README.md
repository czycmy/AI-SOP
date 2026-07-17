# AI SOP 监控台

面向模具中心固定工位的装配过程监控项目。系统通过固定摄像头、YOLO 目标检测、孔位 ROI 和时序状态机，判断零件是否按 SOP 顺序放入并完成紧固，同时记录错序和禁止工具异常。

当前现场客户端为 PySide6 原生桌面程序，不依赖浏览器。目标部署环境为 Windows GPU 边缘电脑，取流支持本地摄像头、离线视频、RTSP 和 Windows 海康 SDK。

## 当前试点

- 监控区域：`R1`。
- 孔位顺序：`H1 -> H2 -> H3 -> H4 -> H5 -> H6`。
- 当前不区分零件型号，只判断孔位内是否出现已装零件。
- 模型类别：`installed_part`、`l_tool_visible`、`forbidden_tool`。
- 手部监控为可选展示，不参与 SOP 完成或异常判定。

## 核心判定逻辑

单个孔位不是“检测到零件就完成”，而是：

```text
当前孔位零件稳定出现
-> 记录步骤开始并开始计时
-> L 型工具参与紧固
-> 工具离开
-> 当前零件再次稳定可见
-> 当前孔位完成并进入下一孔位
```

当前配置使用的主要阈值：

- 零件稳定确认：8 个检测帧。
- L 型工具证据：累计 5 个检测帧。
- 工具离开：有时间戳时等待 4500 ms，无时间戳时使用帧数回退。
- 锉刀报警：连续 3 个检测帧。
- 锉刀报警复位：消失 5000 ms。
- 漏装超时：当前为 `0`，即默认关闭；可靠的工件周期末漏装结算尚待实现。

错序后系统记录异常但继续监控。已经提前安装并记录错序的孔位，在前序步骤完成后会被跳过，不再重复计时。

## 已实现能力

- PySide6 现场桌面客户端和自动取流。
- 固定 ROI 下的六孔位映射。
- 三类 YOLO 检测结果接入。
- 零件落位、工具紧固、工具离开和最终确认状态机。
- 顺序异常去重及错序后继续推进。
- 锉刀独立报警、锁存和清除复位。
- 孔位安装耗时和实际已装孔位统计。
- OpenCV 最新帧 RTSP 取流。
- Windows 海康 HCNetSDK/PlayCtrl 低延迟取流。
- 可选 MediaPipe 手部关键点展示。
- LabelMe 转 YOLO、视频抽帧和 ROI 标定脚本。

## 当前限制

- 工件开始、结束和下一工件自动复位尚未闭环。
- 默认不使用固定等待时间判定漏装。
- `开始监控`、`暂停监控`、`恢复监控`、`结束/复位`、`异常确认`仍是界面预留按钮。
- 桌面客户端的异常记录尚未持久化到磁盘。
- 手套场景下 MediaPipe 只适合辅助展示。
- 当前训练集规模较小，特别是 L 型工具类别仍需补充数据。

## 目录结构

```text
.
├── configs/
│   ├── sample_sop.json          # 六孔位示例及业务阈值，无现场 ROI
│   └── calibrated_sop.json      # 当前现场标定 ROI
├── examples/                    # JSONL 状态机回放样例
├── scripts/
│   ├── env.sh                   # 本地缓存环境变量
│   ├── extract_frames.py        # 视频定时抽帧
│   ├── labelme_to_yolo.py       # 三类 LabelMe 标注转 YOLO
│   └── roi_calibrator.py        # 总区域和孔位 ROI 标定
├── sop_monitor/
│   ├── camera_source.py         # OpenCV/RTSP/海康 SDK 统一取流层
│   ├── camera_monitor.py        # YOLO 检测结果转换与 CLI 监控
│   ├── camera_preview.py        # 简单摄像头预览
│   ├── camera_utils.py          # 相机参数、ROI 映射和画框
│   ├── config.py                # SOP 配置读取
│   ├── desktop_app.py           # PySide6 现场客户端
│   ├── detector.py              # JSONL 检测输入
│   ├── event_log.py             # JSONL 事件日志
│   ├── hand_detector.py         # MediaPipe 手部展示
│   ├── models.py                # 核心数据结构
│   └── state_machine.py         # SOP 核心状态机
├── tests/                       # 单元测试
├── third_party/hikvision/       # 海康 Windows SDK DLL 放置目录
├── requirements.txt
└── requirements-windows.txt
```

`dataset/`、`runs/`、模型权重、MediaPipe 模型、海康 DLL 和输出文档不提交到 Git。换电脑时需要单独传输这些运行资产。

## 环境安装

### macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
source scripts/env.sh
```

### Windows GPU

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements-windows.txt
```

CUDA 版 PyTorch 命令应根据目标显卡驱动和 PyTorch 官方版本调整。

## 模型文件

当前主模型是三类 YOLOv8n 权重：

```text
runs/sop_objects_v1/weights/best.pt
```

该文件属于本地训练产物，不在 GitHub 中。旧的 `runs/installed_part_roi/weights/best.pt` 只有 `installed_part` 一类，不能支持当前 L 型工具证据和锉刀报警。

手部展示模型：

```text
models/hand_landmarker.task
```

## 启动客户端

### 离线视频测试

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --config configs/calibrated_sop.json \
  --camera dataset/raw_videos/normal/1.mp4 \
  --model runs/sop_objects_v1/weights/best.pt \
  --conf 0.35
```

更换视频即可测试错序、漏装或锉刀场景。离线视频默认按正常速度播放，并自动每 3 帧执行一次 YOLO 推理。

### 本地摄像头预览

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --config configs/calibrated_sop.json \
  --camera 0
```

没有传入 `--model` 时只显示画面，不执行 SOP 检测。

### RTSP/OpenCV

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --camera-backend opencv \
  --hikvision-ip 相机IP \
  --hikvision-user admin \
  --hikvision-password 相机密码 \
  --hikvision-channel 101 \
  --config configs/calibrated_sop.json \
  --model runs/sop_objects_v1/weights/best.pt \
  --conf 0.35
```

OpenCV RTSP 后端使用后台线程持续读流，只保留最新帧，避免推理慢时旧帧堆积。

### Windows 海康 SDK

先将 Windows 64 位海康 SDK 的运行 DLL 放入 `third_party\hikvision\`，然后运行：

```bat
.venv\Scripts\python -m sop_monitor.desktop_app ^
  --camera-backend hikvision-sdk ^
  --hikvision-sdk-dir third_party\hikvision ^
  --hikvision-ip 相机IP ^
  --hikvision-user admin ^
  --hikvision-password 相机密码 ^
  --hikvision-port 8000 ^
  --hikvision-channel 101 ^
  --config configs\calibrated_sop.json ^
  --model runs\sop_objects_v1\weights\best.pt ^
  --conf 0.35
```

`101` 通常表示第 1 通道主码流，`102` 表示第 1 通道子码流。

### 开启手部展示

在客户端命令后追加：

```text
--hands --hand-model models/hand_landmarker.task --hand-interval 5
```

手部结果只影响画面和右侧手部状态，不参与孔位完成、错序、漏装或锉刀判定。

## ROI 标定

使用固定相机下的一张清晰图片，先框总监控区域，再按顺序框 H1-H6：

```bash
source scripts/env.sh
.venv/bin/python scripts/roi_calibrator.py \
  --image dataset/用于标定的图片.png \
  --config configs/sample_sop.json \
  --output configs/calibrated_sop.json
```

只重新标定总区域时追加 `--region-only`。ROI 使用归一化 `[x1, y1, x2, y2]`，相机位置、角度或焦距变化后需要重新检查。

## 数据准备

按每秒 2 帧抽取无损 PNG：

```bash
.venv/bin/python scripts/extract_frames.py dataset/raw_videos/normal/1.mp4 \
  --output dataset/frames_sop_png \
  --interval 0.5 \
  --prefix normal_1 \
  --ext png
```

当前 LabelMe 类别必须为：

```text
installed_part
l_tool_visible
forbidden_tool
```

转换为 YOLO 数据集：

```bash
.venv/bin/python scripts/labelme_to_yolo.py \
  --input dataset/sop_png \
  --output dataset/yolo_sop_objects \
  --val-ratio 0.2 \
  --seed 42
```

没有 LabelMe JSON 的图片会生成空标签，作为负样本参与训练。

## 训练示例

```bash
source scripts/env.sh
.venv/bin/yolo detect train \
  model=yolov8n.pt \
  data=dataset/yolo_sop_objects/data.yaml \
  imgsz=640 \
  epochs=50 \
  batch=4 \
  workers=0 \
  project=runs \
  name=sop_objects_v1
```

在 Windows GPU 上将 `device=0` 加入训练或验证参数。

## JSONL 状态机回放

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.cli \
  --config configs/sample_sop.json \
  --detections examples/normal_run.jsonl \
  --events runs/normal_events.jsonl
```

该入口用于验证业务状态机，不调用 YOLO 或摄像头。

## 测试

```bash
source scripts/env.sh
.venv/bin/python -m unittest discover -s tests -v
```

当前共有 39 项测试，覆盖状态机、顺序异常、工具证据、锉刀报警、ROI、海康通道解析和手部 ROI 判断。
