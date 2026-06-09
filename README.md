# AI SOP AOI Monitor MVP

第一阶段 MVP：按区域串行监控孔位装配，校验 SOP 顺序和装配完整性，并记录异常。

当前版本不绑定具体视觉模型。检测器只需要输出结构化结果，后续可以接入 YOLO/ONNX/TensorRT。

## 功能

- 分区域串行监控：当前区域通过后自动切换到下一区域
- 孔位装配顺序校验：发现提前装后续孔位时记录异常
- 孔位装配完整性校验：连续多帧确认零件存在后才判定完成
- 顺序异常/漏装超时记录：输出 JSONL 事件日志
- 手部检测预留：结构上保留扩展点，但第一阶段不影响业务判断

## 快速运行

```bash
python3 -m sop_monitor.cli --config configs/sample_sop.json --detections examples/normal_run.jsonl
```

查看异常场景：

```bash
python3 -m sop_monitor.cli --config configs/sample_sop.json --detections examples/error_run.jsonl
```

默认事件日志输出到 `runs/events.jsonl`。

## 前端界面

启动静态服务：

```bash
python3 -m http.server 8000
```

浏览器访问：

```text
http://localhost:8000/web/
```

前端会读取 `configs/sample_sop.json` 和 `examples/*.jsonl`，展示区域进度、当前孔位、
装配画面、总览状态和异常记录。异常记录只展示顺序异常、漏装超时等异常事件。

装配画面包含一个手部监控展示层。当前版本先使用模拟关键点显示“检测到手/手部靠近区域”，
只用于演示和遮挡提示，不参与 SOP 顺序、孔位完成、区域切换等核心判定。后续可替换为
MediaPipe Hand Landmarker 输出的 21 个手部关键点。

## 检测结果格式

检测器每帧输出一行 JSON：

```json
{"frame_index": 1, "detections": [{"region_id": "R1", "hole_id": "H1", "part_type": "screw_A", "present": true, "confidence": 0.91}]}
```

字段说明：

- `region_id`：区域 ID
- `hole_id`：孔位 ID
- `part_type`：兼容字段；当前阶段不校验零件类型，可统一传 `installed_part`
- `present`：是否存在
- `confidence`：检测置信度

## SOP 配置格式

见 [configs/sample_sop.json](/Users/apple/Documents/AOI/configs/sample_sop.json)。

核心参数：

- `stable_frames_required`：连续多少帧确认后判定装配完成
- `confidence_threshold`：最低检测置信度
- `missing_timeout_frames`：当前步骤等待超过多少帧仍未确认则记录漏装

## 后续接 YOLO

只需要实现一个检测器，把 YOLO 输出转换成 `FrameObservation`：

```python
from sop_monitor.models import Detection, FrameObservation

FrameObservation(
    frame_index=frame_index,
    detections=[
        Detection(region_id="R1", hole_id="H1", part_type="screw_A", present=True, confidence=0.9)
    ],
)
```
