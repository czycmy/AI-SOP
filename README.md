# AI SOP 监控台

面向模具中心固定工位的装配过程监控项目。系统通过固定摄像头、YOLO 目标检测、孔位 ROI 和时序状态机，判断零件是否按 SOP 顺序放入并完成紧固，同时记录错序和禁止工具异常。

当前现场客户端为 PySide6 原生桌面程序，不依赖浏览器。目标部署环境为 Windows GPU 边缘电脑，取流支持本地摄像头、离线视频、RTSP 和 Windows 海康 SDK。

> 第一次接手、部署或维护本项目，请先阅读
> [《AI SOP 项目技术交接手册》](docs/AI_SOP项目技术交接手册.md)。手册按零基础接手场景说明业务逻辑、环境、模型资产、启动、训练和故障排查。

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
│   ├── sample_sop.json                   # 六孔位模板及业务阈值，无现场 ROI
│   ├── calibrated_sop.json               # 原现场画面的孔位 ROI
│   ├── calibrated_sop_bare_file.json     # 新测试视频专用孔位 ROI
│   └── action_rois.json                   # H3/H4 连续动作大 ROI
├── docs/
│   └── AI_SOP项目技术交接手册.md         # 面向新接手人的完整技术手册
├── examples/                    # JSONL 状态机回放样例
├── scripts/
│   ├── env.sh                   # 本地缓存环境变量
│   ├── extract_frames.py        # 视频定时抽帧
│   ├── labelme_to_yolo.py       # 三类 LabelMe 标注转 YOLO
│   ├── roi_calibrator.py        # 总区域和孔位 ROI 标定
│   ├── action_roi_calibrator.py # H3/H4 动作 ROI 标定
│   ├── extract_action_windows.py # 连续动作困难负样本切片
│   ├── train_action_classifier.py # RGB/光流动作模型训练
│   └── test_action_video.py     # 双模型完整视频测试
├── sop_monitor/
│   ├── action_recognition.py    # 动作预处理、概率融合和时序投票
│   ├── action_runtime.py        # 客户端双动作模型实时推理
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

裸锉刀连续动作使用两个配套权重：

```text
runs/filing_action_rgb_fusion_v1/best.pt
runs/filing_action_flow_fusion_v1/best.pt
```

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

### macOS客户端测试双模型动作报警

当前Mac环境没有可用的CUDA/MPS，下面命令只用于确认客户端报警界面和事件记录，
视频运行会比正常速度慢。先不加载YOLO，避免三个模型同时占用CPU：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --config configs/calibrated_sop.json \
  --camera dataset/action_test/bare_file_full_test.mp4 \
  --action-rgb-model runs/filing_action_rgb_fusion_v1/best.pt \
  --action-flow-model runs/filing_action_flow_fusion_v1/best.pt \
  --action-device cpu \
  --action-interval 1.0 \
  --action-rgb-weight 0.7 \
  --action-threshold 0.5 \
  --action-clear-threshold 0.35
```

不弹出客户端窗口、直接导出完整客户端演示视频时，使用下面命令。该模式会同时
运行YOLO与双动作模型，把装配画面、状态、SOP表格、统计和异常记录一起写入MP4，
处理完成后自动退出：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --config configs/calibrated_sop.json \
  --camera dataset/action_test/bare_file_full_test.mp4 \
  --model runs/sop_objects_v1/weights/best.pt \
  --conf 0.35 \
  --detect-interval 3 \
  --action-rgb-model runs/filing_action_rgb_fusion_v1/best.pt \
  --action-flow-model runs/filing_action_flow_fusion_v1/best.pt \
  --action-device cpu \
  --action-interval 1.0 \
  --action-rgb-weight 0.7 \
  --action-threshold 0.5 \
  --action-clear-threshold 0.35 \
  --export-client-video runs/client_demo_fusion_v1.mp4 \
  --export-fps 10 \
  --export-width 1440 \
  --export-height 900
```

Windows GPU现场或离线完整联调时，再同时加载YOLO并使用CUDA：

```powershell
python -m sop_monitor.desktop_app --config configs\calibrated_sop.json --camera dataset\action_test\bare_file_full_test.mp4 --model runs\sop_objects_v1\weights\best.pt --conf 0.35 --action-rgb-model runs\filing_action_rgb_fusion_v1\best.pt --action-flow-model runs\filing_action_flow_fusion_v1\best.pt --action-device cuda --action-interval 0.2 --action-rgb-weight 0.7 --action-threshold 0.5 --action-clear-threshold 0.35
```

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

如果新视频与原相机画面只有轻微偏移，可以继承旧配置并只重画指定孔位。下面的命令从视频 140 秒处取帧，只更新 H2，并另存为新配置；原配置不会被覆盖：

```bash
source scripts/env.sh
.venv/bin/python scripts/roi_calibrator.py \
  --video dataset/action_test/bare_file_full_test.mp4 \
  --timestamp 140 \
  --config configs/calibrated_sop.json \
  --output configs/calibrated_sop_bare_file.json \
  --only-hole H2
```

多个孔位可写成 `--only-hole H2 H3`。旧视频继续使用 `configs/calibrated_sop.json`，新视频启动或导出时改用对应的新配置文件。

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

当前测试覆盖状态机、顺序异常、工具证据、锉刀报警、ROI、海康通道解析、手部 ROI 和动作预处理判断。

## 裸锉刀锉削动作训练与双路融合

黄色包装被撕掉后，单帧 YOLO 可能无法可靠识别锉刀。项目提供
`scripts/train_action_classifier.py`，用短视频分别训练 RGB 外观模型和方向光流
模型，作为现有 `forbidden_tool` 外观检测的补充。RGB 路负责工具、手部和场景
外观，方向光流路保留水平/垂直运动方向，用于识别锉刀的连续往复动作。

数据不需要逐帧画框，按动作类别放入对应文件夹：

```text
dataset/action_videos/
├── filing_action/
├── normal_tightening/
└── other_action/          # 放件、调整、遮挡等非锉削困难负样本
```

脚本会对每个类别分别随机划分 `80%/10%/10%`，因此正常紧固与锉削视频会均衡
分配到训练、验证和测试集合。固定随机种子默认为42，划分明细保存在
`dataset_split.csv`。脚本同时读取 `configs/action_rois.json` 中的H3/H4动作ROI，
排除画面顶部时间戳，并按当前窗口的运动量选择正在操作的ROI。默认使用24帧、
10 FPS、160×160输入，对应约2.4秒动作窗口。短于1.8秒的视频默认跳过。

默认使用三分类：`filing_action`、`normal_tightening`、`other_action`，最终只把
`filing_action`作为报警类别。这样可以让模型分别学习正常紧固和其他正常动作，
不再把所有负样本强行压成一种外观。旧权重需要复现时可增加
`--label-mode binary`。

### 以下两项是一次性数据准备，当前数据已完成

**FFmpeg处理不是每次训练都要执行。** 前期短视频出现H.264缺帧和OpenCV解码
报错，因此才用FFmpeg重新封装/编码，并把可正常读取的数据放入
`dataset/action_videos_fixed`。只要现在三个类别中的视频都能读取，就直接训练，
无需再次转换。下面命令仅用于今后从新的完整正常视频补充 `other_action` 困难
负样本：

```bat
python scripts\extract_action_windows.py ^
  --source dataset\raw_videos\normal\1.mp4 ^
  --output dataset\action_videos_fixed\other_action ^
  --window 3 ^
  --stride 3 ^
  --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
```

**动作ROI也只需标定一次。** `configs/action_rois.json` 中的H3/H4是锉削动作模型
使用的两个较大观察区域，不是六个孔位ROI，也不是逐帧标注框。当前配置已经
完成，只有相机位置、画面裁切或模具位置明显变化时才重新执行：

```bash
.venv/bin/python scripts/action_roi_calibrator.py \
  --source dataset/action_test/bare_file_full_test.mp4 \
  --timestamp 20 \
  --output configs/action_rois.json
```

依次框选H3、H4附近较大的动作范围，框内需要保留孔位、手、L型工具与锉刀的
主要运动轨迹。动作ROI仅用于动作模型，不会修改原有六个孔位的装配判断ROI。

### Windows PowerShell：从训练到测试

以下命令全部在项目根目录的 **PowerShell** 中执行，每个代码块是一条完整命令，
不要输入 CMD 使用的 `^`，也不要输入网页中的 `<br/>`。

#### 第1步：进入项目并激活环境

```powershell
cd C:\Users\zhengyu_chen\PycharmProjects\AI-SOP-main\AI-SOP-main
.\.venv\Scripts\Activate.ps1
```

命令行左侧出现 `(.venv)` 后，再继续下面步骤。

#### 第2步：确认 CUDA 和三类数据

```powershell
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

结果必须显示 `CUDA可用: True`。数据目录必须是：

```text
dataset\action_videos_fixed\filing_action\          锉刀动作视频
dataset\action_videos_fixed\normal_tightening\     正常L型工具紧固视频
dataset\action_videos_fixed\other_action\           放件、拿取、手经过等正常动作
```

同时确认 `configs\action_rois.json` 存在。RGB和方向光流训练必须使用这同一份ROI。

#### 第3步：重新训练 RGB 模型

本版本改成三分类并增加了小数据时序增强，因此建议重新训练RGB，不再混用前面
实验产生的RGB权重：

```powershell
python scripts\train_action_classifier.py --data dataset\action_videos_fixed --output runs\filing_action_rgb_fusion_v1 --input-mode rgb --label-mode three-class --epochs 30 --batch-size 4 --frames 24 --sample-fps 10 --action-rois configs\action_rois.json --seed 42 --device cuda
```

训练结束后确认下面文件存在：

```powershell
Test-Path runs\filing_action_rgb_fusion_v1\best.pt
```

输出 `True` 才进入下一步。

#### 第4步：训练方向光流模型

使用完全相同的数据、ROI、随机种子和采样参数，只改变输入模式和输出目录：

```powershell
python scripts\train_action_classifier.py --data dataset\action_videos_fixed --output runs\filing_action_flow_fusion_v1 --input-mode flow --label-mode three-class --epochs 30 --batch-size 4 --frames 24 --sample-fps 10 --action-rois configs\action_rois.json --seed 42 --device cuda
```

训练结束后确认：

```powershell
Test-Path runs\filing_action_flow_fusion_v1\best.pt
```

输出 `True` 表示两个模型都准备好了。显存不足时，只把两条训练命令中的
`--batch-size 4` 改成 `--batch-size 2`，其他参数不要改。

#### 第5步：测试完整视频

```powershell
python scripts\test_action_video.py --rgb-model runs\filing_action_rgb_fusion_v1\best.pt --flow-model runs\filing_action_flow_fusion_v1\best.pt --source dataset\action_test\bare_file_full_test.mp4 --output runs\filing_action_fusion_test_v1 --rgb-weight 0.7 --threshold 0.5 --clear-threshold 0.35 --stride-seconds 0.2 --vote-window 4 --alarm-windows 3 --clear-windows 4 --device cuda
```

该命令按 `RGB 0.7 + 方向光流 0.3` 融合。4个滑动窗口中至少3个达到0.5才
触发报警；报警后连续4个窗口低于0.35才结束事件，所以一次连续锉削只累计一次。

结果目录会生成：

- `result.mp4`：带融合概率、报警状态和事件次数的视频。
- `predictions.csv`：每个滑动窗口的 RGB、方向光流、融合概率及H3/H4明细。
- `events.csv`：去重后的锉削事件起止时间、持续时间、峰值概率和区域。

本次测试视频的已知锉削区间约为 `24～42秒` 和 `73～94秒`。第一轮验收只看：

1. `events.csv` 是否正好有2条事件。
2. 两条事件是否分别与上述两个区间重叠。
3. 其他时间是否没有新增事件。

窗口本身覆盖约2.4秒，再加连续投票，因此报警时间比真实动作开始晚约1～3秒
属于正常现象。测试结果不满足时，先保留 `predictions.csv` 和 `events.csv`，不要
反复随机改阈值或重新训练；应根据RGB、光流各自概率确定是哪一路造成漏报或误报。

旧的单模型命令 `--model xxx\best.pt` 仍可用于对照。`motion` 绝对帧差模式也保留
用于读取旧权重，但它不保留运动方向，不再作为锉削动作的推荐训练方式。小数据
随机验证集出现 `acc=1.000` 只代表该次划分，最终仍应以完整视频中目标锉削区间
是否各产生一次事件、其他时段是否零报警作为验收标准。
