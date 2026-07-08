# AI SOP 监控台

本项目用于模具中心装配 SOP 监控。当前核心目标是：按物理区域串行监控孔位装配，确认每个孔位是否按预设顺序完成安装，并记录顺序异常、漏装超时等异常事件。

当前版本的装配判定逻辑是：

```text
摄像头/检测结果 -> 孔位已装状态 -> SOP 状态机 -> PySide6 客户端展示/事件日志
```

模型暂时只需要判断“孔位有没有装零件”，不判断零件类型是否正确。手部监控作为画面状态展示和后续遮挡控制预留，不改变当前 SOP 判定结果。

## 当前能力

- 分区域串行监控：第一区域通过后自动切换到第二区域。
- 孔位顺序校验：检测到后续孔位提前安装时记录顺序异常。
- 孔位完整性校验：当前孔位连续多帧确认已装后才判定完成。
- 漏装超时记录：当前孔位长时间未确认完成时记录漏装异常。
- PySide6 桌面客户端：展示区域 SOP、装配画面、总览状态、异常记录。
- 摄像头预览：用于确认摄像头编号和系统权限。
- YOLO 实时检测入口：接入训练好的 `best.pt` 后可做摄像头实时 SOP 监控。
- 低延迟取流后端：OpenCV/RTSP 默认只保留最新帧，避免旧帧堆积。
- 海康 SDK 后端：Windows 下可通过 `--camera-backend hikvision-sdk` 使用 HCNetSDK/PlayCtrl 取流。
- MediaPipe 手部监控展示：可选功能，现场默认不建议开启，不参与 SOP 判定。

## 目录结构

```text
.
├── configs/
│   └── sample_sop.json
├── examples/
│   ├── normal_run.jsonl
│   └── error_run.jsonl
├── scripts/
│   └── env.sh
├── sop_monitor/
│   ├── __init__.py
│   ├── camera_source.py
│   ├── camera_monitor.py
│   ├── camera_preview.py
│   ├── camera_utils.py
│   ├── cli.py
│   ├── config.py
│   ├── desktop_app.py
│   ├── detector.py
│   ├── event_log.py
│   ├── hand_detector.py
│   ├── models.py
│   └── state_machine.py
├── tests/
│   ├── test_camera_utils.py
│   ├── test_hand_detector.py
│   └── test_state_machine.py
├── third_party/
│   └── hikvision/
│       └── README.md
├── requirements.txt
├── requirements-windows.txt
└── README.md
```

## 文件说明

### 配置和样例

- `configs/sample_sop.json`
  - 示例 SOP 配置。
  - 定义区域顺序、孔位顺序、稳定帧阈值、漏装超时帧数、孔位 ROI。
  - `roi` 是归一化坐标 `[x1, y1, x2, y2]`，用于把 YOLO 检测框映射到 H1/H2 等孔位。

- `examples/normal_run.jsonl`
  - 正常装配流程的逐帧检测结果样例。
  - 用于验证状态机能按顺序完成 R1 内的 3 个孔位。

- `examples/error_run.jsonl`
  - 异常流程的逐帧检测结果样例。
  - 用于验证顺序异常记录。

### 运行环境

- `requirements.txt`
  - Python 依赖列表。
  - 当前包含 `PySide6`、`ultralytics`、`opencv-python`、`onnxruntime`、`mediapipe`。

- `requirements-windows.txt`
  - Windows 现场部署依赖。
  - 不包含 `torch`，需要先按 PyTorch 官方命令单独安装 CUDA 版 PyTorch。
  - 不默认包含 `mediapipe`，现场主线先关闭手部监控。

- `scripts/env.sh`
  - 项目本地运行环境变量。
  - 将 YOLO 和 Matplotlib 缓存写到项目 `.cache/`，避免写用户目录时出现权限警告。

### 后端核心代码

- `sop_monitor/models.py`
  - 项目核心数据模型。
  - 包括 `MonitorConfig`、`RegionSpec`、`StepSpec`、`Detection`、`FrameObservation`、`MonitorEvent`。

- `sop_monitor/config.py`
  - SOP 配置加载器。
  - 读取 JSON 配置并转换为 `MonitorConfig`。

- `sop_monitor/state_machine.py`
  - SOP 状态机，项目核心业务逻辑。
  - 负责区域切换、当前孔位判断、顺序异常、漏装超时、稳定帧投票。

- `sop_monitor/detector.py`
  - 检测输入适配器。
  - 当前实现 `JsonlDetectionReader`，用于读取 JSONL 检测结果。
  - 后续 YOLO/ONNX/TensorRT 检测器只要输出同样的 `FrameObservation` 即可接入。

- `sop_monitor/event_log.py`
  - 事件日志写入器。
  - 将 SOP 事件写入 JSONL 文件，默认输出到 `runs/`。

- `sop_monitor/cli.py`
  - JSONL 回放命令入口。
  - 用于在没有摄像头和模型时验证 SOP 状态机逻辑。

- `sop_monitor/desktop_app.py`
  - PySide6 桌面客户端入口。
  - 面向现场边缘机器部署，不需要操作浏览器。
  - 打开后自动接入摄像头预览，展示区域 SOP、装配画面、状态概览和异常记录。
  - 可通过 `--hands` 开启手部关键点展示。
  - 当前 `开始监控`、`暂停监控`、`恢复监控`、`结束/复位`、`异常确认` 是界面预留。

### 摄像头和视觉模块

- `sop_monitor/camera_source.py`
  - 摄像头后端抽象层。
  - `OpenCvFrameSource` 支持本地摄像头和离线视频。
  - `LatestFrameOpenCvSource` 支持 RTSP 最新帧缓存，推理慢时丢旧帧降低延迟。
  - `HikvisionSdkFrameSource` 支持 Windows 海康 HCNetSDK/PlayCtrl 取流。

- `sop_monitor/camera_preview.py`
  - 摄像头预览命令。
  - 用于确认笔记本内置摄像头或外接摄像头是否可用。

- `sop_monitor/camera_utils.py`
  - 摄像头公共工具。
  - 包括打开摄像头、ROI 映射、画面覆盖绘制等。

- `sop_monitor/camera_monitor.py`
  - 摄像头实时 SOP 监控命令。
  - 使用 YOLO 模型检测已装零件，再通过 ROI 映射到孔位，最后交给状态机判断。
  - 可选开启后端 MediaPipe 手部检测。

- `sop_monitor/hand_detector.py`
  - 后端手部检测模块。
  - 使用 MediaPipe Hand Landmarker 检测手部关键点。
  - 当前仅用于画面状态提示，不改变 SOP 判定。

### 测试

- `tests/test_state_machine.py`
  - SOP 状态机测试。
  - 覆盖区域串行完成、顺序异常、漏装超时等核心逻辑。

- `tests/test_camera_utils.py`
  - 摄像头 ROI 工具测试。
  - 验证检测框中心点能正确匹配孔位 ROI。

- `tests/test_hand_detector.py`
  - 手部检测辅助逻辑测试。
  - 验证手部框和 ROI 的相交判断。

## 环境准备

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source scripts/env.sh
```

### Windows

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

如果是 Windows + NVIDIA GPU，先安装 CUDA 版 PyTorch，再安装项目依赖。示例：

```bat
.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements-windows.txt
```

如果暂时只做普通开发、不使用 GPU，也可以用 `requirements.txt`：

```bat
.venv\Scripts\python -m pip install -r requirements.txt
```

Windows 没有 `source scripts/env.sh`，可以先不设置缓存环境变量；如需设置，可在 PowerShell 中配置 `YOLO_CONFIG_DIR` 和 `MPLCONFIGDIR`。

## 启动方式

### 1. 启动 PySide6 桌面客户端

macOS / Linux：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --config configs/sample_sop.json \
  --camera 0
```

Windows：

```bat
.venv\Scripts\python -m sop_monitor.desktop_app --config configs\sample_sop.json --camera 0
```

说明：

- 这是现场部署优先使用的入口，不需要打开浏览器。
- 窗口启动后会自动打开摄像头预览。
- `--camera-backend opencv` 是默认后端，支持本地编号、RTSP 地址或视频路径。
- `--camera-backend hikvision-sdk` 是 Windows 海康 SDK 后端，需要放好 SDK DLL。
- 当前桌面客户端已支持传入 YOLO 模型后做 ROI/SOP 实时判断；开始/暂停/恢复/复位按钮仍是界面预留。
- 客户端实时监控建议优先使用海康子码流 `102` 降低延迟；需要高分辨率标定或截图时再切主码流 `101`。

海康 RTSP/OpenCV 后端启动示例：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --camera-backend opencv \
  --config configs/sample_sop.json \
  --model runs/installed_part_roi/weights/best.pt \
  --hikvision-ip 192.168.114.222 \
  --hikvision-user admin \
  --hikvision-password '<password>' \
  --hikvision-channel 102 \
  --conf 0.35
```

Windows 写法：

```bat
.venv\Scripts\python -m sop_monitor.desktop_app ^
  --camera-backend opencv ^
  --config configs\calibrated_sop.json ^
  --model runs\installed_part_roi\weights\best.pt ^
  --hikvision-ip 192.168.114.222 ^
  --hikvision-user admin ^
  --hikvision-password "<password>" ^
  --hikvision-channel 102 ^
  --conf 0.35
```

也可以直接传完整 RTSP 地址：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --camera-backend opencv \
  --config configs/sample_sop.json \
  --camera 'rtsp://admin:<password>@192.168.114.222:554/Streaming/Channels/101'
```

海康 SDK 后端启动示例，仅 Windows 可用：

```bat
.venv\Scripts\python -m sop_monitor.desktop_app ^
  --camera-backend hikvision-sdk ^
  --hikvision-sdk-dir third_party\hikvision ^
  --hikvision-ip 192.168.114.222 ^
  --hikvision-user admin ^
  --hikvision-password "<password>" ^
  --hikvision-port 8000 ^
  --hikvision-channel 102 ^
  --config configs\calibrated_sop.json ^
  --model runs\installed_part_roi\weights\best.pt ^
  --conf 0.35
```

SDK 后端需要把海康 Windows 64 位 SDK 的 DLL 放到：

```text
third_party\hikvision\
```

常见 DLL 包括：

```text
HCNetSDK.dll
PlayCtrl.dll
HCCore.dll
hpr.dll
libcrypto-*.dll
libssl-*.dll
其他 SDK 随包 DLL
```

`--hikvision-channel 101` 表示 1 通道主码流，`102` 表示 1 通道子码流。SDK 登录端口通常是 `8000`，RTSP 端口通常是 `554`。

开启客户端手部监控展示：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.desktop_app \
  --config configs/sample_sop.json \
  --hikvision-ip 192.168.114.222 \
  --hikvision-user admin \
  --hikvision-password '<password>' \
  --hands \
  --hand-interval 5 \
  --hand-model models/hand_landmarker.task
```

说明：

- 手部监控只影响画面展示和右侧“手部状态”，不参与 SOP 顺序和孔位完成判定。
- 画面中检测到手时会叠加手部框、关键点和骨架。
- 如果手部框接近当前孔位 ROI，右侧状态会显示“靠近区域”。
- `--hand-interval` 表示每隔多少帧做一次手部检测，数值越大延迟越低但手部刷新越慢。

### 2. JSONL 回放验证

正常流程：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.cli \
  --config configs/sample_sop.json \
  --detections examples/normal_run.jsonl
```

异常流程：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.cli \
  --config configs/sample_sop.json \
  --detections examples/error_run.jsonl
```

默认事件日志输出：

```text
runs/events.jsonl
```

### 3. 摄像头预览

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.camera_preview --camera 0
```

预览海康 RTSP 摄像头：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.camera_preview \
  --hikvision-ip 192.168.114.222 \
  --hikvision-user admin \
  --hikvision-password '<password>' \
  --hikvision-channel 101
```

说明：

- `--camera 0` 通常是笔记本自带摄像头。
- 外接 USB 摄像头可能是 `--camera 1` 或 `--camera 2`。
- 海康主码流通常是 `--hikvision-channel 101`，子码流通常是 `102`。
- macOS 第一次打开摄像头时，需要允许终端/Python 访问摄像头。
- 预览窗口按 `q` 或 `Esc` 退出。

### 4. 摄像头实时 SOP 监控

需要先准备训练好的 YOLO 模型，例如：

```text
weights/best.pt
```

运行：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.camera_monitor \
  --hikvision-ip 192.168.114.222 \
  --hikvision-user admin \
  --hikvision-password '<password>' \
  --config configs/sample_sop.json \
  --model weights/best.pt \
  --display
```

流程：

```text
摄像头画面
-> YOLO 检测已装零件
-> 检测框中心点匹配孔位 ROI
-> 生成 H1/H2 已装状态
-> SOP 状态机判断顺序和完整性
-> 输出事件日志
```

### 5. 命令行实时监控开启手部检测

后端手部检测使用 MediaPipe Hand Landmarker。当前不默认开启，需要显式加 `--hands`：

```bash
source scripts/env.sh
.venv/bin/python -m sop_monitor.camera_monitor \
  --hikvision-ip 192.168.114.222 \
  --hikvision-user admin \
  --hikvision-password '<password>' \
  --config configs/sample_sop.json \
  --model weights/best.pt \
  --display \
  --hands \
  --hand-model models/hand_landmarker.task
```

注意：

- `models/hand_landmarker.task` 是 MediaPipe 手部模型文件。
- 该模型文件较大，已被 `.gitignore` 忽略。
- 部分 macOS 或无头环境下 MediaPipe 图形后端可能不稳定，所以默认不开启。
- 当前手部检测只做画面提示，不参与 SOP 判定。

## SOP 配置说明

示例：

```json
{
  "confidence_threshold": 0.75,
  "stable_frames_required": 3,
  "missing_timeout_frames": 12,
  "regions": [
    {
      "region_id": "R1",
      "name": "第一片区域",
      "steps": [
        {
          "step": 1,
          "hole_id": "H1",
          "part_type": "installed_part",
          "roi": [0.12, 0.22, 0.28, 0.46]
        }
      ]
    }
  ]
}
```

字段说明：

- `confidence_threshold`：SOP 状态机接受检测结果的最低置信度。
- `stable_frames_required`：连续多少帧确认后判定孔位完成。
- `missing_timeout_frames`：当前孔位等待超过多少帧仍未完成时记录漏装超时。
- `regions`：物理区域列表，按顺序执行。
- `region_id`：区域 ID，例如 `R1`。
- `steps`：区域内孔位步骤，按 SOP 顺序执行。
- `hole_id`：孔位 ID，例如 `H1`、`H2`。
- `part_type`：兼容字段；当前阶段不校验零件类型，可统一写 `installed_part`。
- `roi`：孔位在摄像头画面中的归一化位置 `[x1, y1, x2, y2]`。

ROI 需要基于真实现场相机画面标定。当前 `sample_sop.json` 中的 ROI 是示例值，不代表真实工位。

## ROI 标定

从固定相机位图片中框选每个孔位的判断区域：

```bash
.venv/bin/python scripts/roi_calibrator.py \
  --image dataset/frames_data/luzhi_f001980_t0066.00s.jpg \
  --config configs/sample_sop.json \
  --output configs/calibrated_sop.json
```

说明：

- 工具会按 SOP 顺序依次提示框选 `R1-H1`、`R1-H2` 等孔位。
- 鼠标拖拽框选 ROI，按 Enter 或 Space 确认。
- 如果当前孔位不想修改，直接取消/不框选会保留原 ROI。
- 输出坐标是归一化 `[x1, y1, x2, y2]`，可直接用于后续检测。
- 建议 ROI 只覆盖当前孔位安装后零件会出现的位置，避免包含旁边已有的相似零件。

## 检测结果 JSONL 格式

每一行代表一帧检测结果：

```json
{"frame_index": 1, "detections": [{"region_id": "R1", "hole_id": "H1", "part_type": "installed_part", "present": true, "confidence": 0.91}]}
```

字段说明：

- `frame_index`：帧序号。
- `detections`：当前帧检测结果列表。
- `region_id`：区域 ID。
- `hole_id`：孔位 ID。
- `part_type`：兼容字段，当前不参与“装对/装错”判断。
- `present`：是否检测到已装零件。
- `confidence`：检测置信度。

## YOLO 模型接入

当前项目没有内置训练好的 `weights/best.pt`。拿到现场数据后，需要训练一个检测“已装零件”的 YOLO 模型。

推荐第一阶段类别：

```yaml
names:
  0: installed_part
```

训练完成后，将模型放到：

```text
weights/best.pt
```

摄像头实时监控会把 YOLO 输出转换为：

```python
Detection(
    region_id="R1",
    hole_id="H1",
    part_type="installed_part",
    present=True,
    confidence=0.9,
)
```

判断孔位的方式：

```text
YOLO 检测框中心点落入哪个孔位 ROI，就认为哪个孔位已装。
```

## 测试

运行全部测试：

```bash
source scripts/env.sh
.venv/bin/python -m unittest discover -s tests
```

当前测试覆盖：

- SOP 状态机
- ROI 映射
- 手部 ROI 接近判断

## 部署说明

### macOS

- 已在 macOS 上开发和验证。
- 第一次访问摄像头时需要在系统设置中允许终端/Python/Codex 访问相机。
- MediaPipe 后端手部检测在部分 macOS 图形环境下可能不稳定，默认关闭。

### Windows

项目主体可迁移到 Windows：

- Python 代码跨平台。
- 现场主界面使用 PySide6 桌面客户端，不需要操作浏览器。
- 摄像头后端支持 `opencv` 和 `hikvision-sdk`。
- `opencv` 后端支持本地摄像头、视频文件和 RTSP，并对 RTSP 做最新帧缓存以降低堆积延迟。
- `hikvision-sdk` 后端需要 Windows 64 位海康 SDK DLL，DLL 放在 `third_party\hikvision\`。
- Windows 命令中的 Python 路径使用 `.venv\Scripts\python`。
- 如果有 NVIDIA GPU，建议单独安装 CUDA 版 PyTorch，再安装 `requirements-windows.txt`。
- 现场主线建议先关闭 `--hands`，手部监控不参与 SOP 判定且会增加延迟。

Windows RTSP/OpenCV 运行示例：

```bat
.venv\Scripts\python -m sop_monitor.desktop_app ^
  --camera-backend opencv ^
  --config configs\calibrated_sop.json ^
  --model runs\installed_part_roi\weights\best.pt ^
  --hikvision-ip 192.168.114.222 ^
  --hikvision-user admin ^
  --hikvision-password "<password>" ^
  --hikvision-channel 102 ^
  --conf 0.35
```

Windows 海康 SDK 运行示例：

```bat
.venv\Scripts\python -m sop_monitor.desktop_app ^
  --camera-backend hikvision-sdk ^
  --hikvision-sdk-dir third_party\hikvision ^
  --hikvision-ip 192.168.114.222 ^
  --hikvision-user admin ^
  --hikvision-password "<password>" ^
  --hikvision-port 8000 ^
  --hikvision-channel 102 ^
  --config configs\calibrated_sop.json ^
  --model runs\installed_part_roi\weights\best.pt ^
  --conf 0.35
```

## 当前限制

- `configs/sample_sop.json` 里的 ROI 是示例值，需要现场标定。
- PySide6 客户端监控按钮目前只做界面预留，还没有接入生产控制接口。
- 后端手部检测为可选功能，当前不参与 SOP 判定。
- 海康 SDK 后端需要在 Windows 真机、真实 DLL 和真实相机网络下验证。
