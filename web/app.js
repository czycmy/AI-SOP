// AI SOP 前端演示逻辑。
// 本文件在浏览器中复刻后端第一阶段状态机：读取 SOP 配置和检测 JSONL，
// 按帧回放检测结果，展示当前区域、当前孔位、稳定帧投票和异常事件。
// 当前阶段只判断孔位有没有装，不判断零件类型是否正确。

const state = {
  config: null,
  frames: [],
  scenario: "normal",
  timer: null,
  cursor: 0,
  regionIndex: 0,
  stepIndex: 0,
  stableFrames: 0,
  elapsedFrames: 0,
  completed: false,
  completedKeys: new Set(),
  errorKeys: new Set(),
  events: [],
  latestDetections: [],
  latestFrame: null,
  hand: {
    status: "idle",
    hint: "画面稳定",
    landmarks: [],
    occluding: false,
  },
};

const samples = {
  normal: "../examples/normal_run.jsonl",
  error: "../examples/error_run.jsonl",
};

const layout = {
  R1: {
    box: { left: 8, top: 12, width: 42, height: 76 },
    holes: {
      H1: { left: 20, top: 34 },
      H2: { left: 37, top: 56 },
      H3: { left: 24, top: 76 },
    },
  },
  R2: {
    box: { left: 56, top: 18, width: 36, height: 68 },
    holes: {
      H4: { left: 68, top: 40 },
      H5: { left: 80, top: 68 },
    },
  },
};

const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  bindElements();
  bindActions();
  await loadConfig();
  await loadScenario("normal");
});

function bindElements() {
  [
    "activeRegion",
    "activeHole",
    "runState",
    "progressText",
    "regionList",
    "stage",
    "frameText",
    "normalBtn",
    "errorBtn",
    "playBtn",
    "stepBtn",
    "resetBtn",
    "doneCount",
    "errorCount",
    "stableCount",
    "eventList",
    "eventCount",
    "handState",
    "handHint",
    "handLayer",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindActions() {
  els.normalBtn.addEventListener("click", () => loadScenario("normal"));
  els.errorBtn.addEventListener("click", () => loadScenario("error"));
  els.playBtn.addEventListener("click", togglePlayback);
  els.stepBtn.addEventListener("click", stepFrame);
  els.resetBtn.addEventListener("click", resetRuntime);
}

async function loadConfig() {
  const response = await fetch("../configs/sample_sop.json");
  state.config = await response.json();
}

async function loadScenario(name) {
  stopPlayback();
  state.scenario = name;
  const response = await fetch(samples[name]);
  const text = await response.text();
  state.frames = parseJsonl(text);
  els.normalBtn.classList.toggle("active", name === "normal");
  els.errorBtn.classList.toggle("active", name === "error");
  resetRuntime();
}

function parseJsonl(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function resetRuntime() {
  stopPlayback();
  state.cursor = 0;
  state.regionIndex = 0;
  state.stepIndex = 0;
  state.stableFrames = 0;
  state.elapsedFrames = 0;
  state.completed = false;
  state.completedKeys = new Set();
  state.errorKeys = new Set();
  state.events = [];
  state.latestDetections = [];
  state.latestFrame = null;
  state.hand = {
    status: "idle",
    hint: "画面稳定",
    landmarks: [],
    occluding: false,
  };
  render();
}

function togglePlayback() {
  if (state.timer) {
    stopPlayback();
    return;
  }
  els.playBtn.classList.add("active");
  els.playBtn.textContent = "Ⅱ";
  state.timer = window.setInterval(() => {
    if (!stepFrame()) {
      stopPlayback();
    }
  }, 650);
}

function stopPlayback() {
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  if (els.playBtn) {
    els.playBtn.classList.remove("active");
    els.playBtn.textContent = "▶";
  }
}

function stepFrame() {
  if (state.cursor >= state.frames.length || state.completed) {
    return false;
  }

  const frame = state.frames[state.cursor];
  state.cursor += 1;
  updateStateMachine(frame);
  render();
  return state.cursor < state.frames.length && !state.completed;
}

function updateStateMachine(frame) {
  state.latestFrame = frame.frame_index;
  updateHandMonitor(frame.frame_index);
  const region = activeRegion();
  const expected = expectedStep();
  if (!region || !expected) return;

  state.elapsedFrames += 1;
  const detections = frame.detections.filter((item) => item.region_id === region.region_id);
  state.latestDetections = frame.detections;

  // 顺序校验：当前区域内其他孔位先出现高置信度零件，即判为顺序异常。
  // 这种情况已经隐含“当前孔位没先完成”，所以不再重复报漏装。
  let hasOutOfOrderDetection = false;
  detections.forEach((detection) => {
    if (detection.hole_id === expected.hole_id) return;
    if (!isHighConfidencePresent(detection)) return;
    hasOutOfOrderDetection = true;
    emitOnce(
      "order_error",
      frame.frame_index,
      expected,
      detection,
      `当前应装 ${expected.hole_id}，但检测到 ${detection.hole_id} 已有零件。`,
    );
  });

  const expectedDetection = bestDetection(detections, expected);
  if (!expectedDetection) {
    state.stableFrames = 0;
    if (!hasOutOfOrderDetection) {
      maybeMissing(frame.frame_index, expected);
    }
    return;
  }

  // 稳定帧投票：连续多帧确认后才真正推进 SOP。
  state.stableFrames += 1;
  if (state.stableFrames < state.config.stable_frames_required) return;

  pushEvent("step_completed", frame.frame_index, expected, expectedDetection, `${region.name} ${expected.hole_id} 装配确认完成。`);
  state.completedKeys.add(`${region.region_id}:${expected.hole_id}`);
  advance(frame.frame_index, region, expected);
}

function updateHandMonitor(frameIndex) {
  if (frameIndex === null || frameIndex === undefined) {
    state.hand = {
      status: "idle",
      hint: "画面稳定",
      landmarks: [],
      occluding: false,
    };
    return;
  }

  const progress = (frameIndex % 9) / 8;
  const centerX = 18 + progress * 58;
  const centerY = 28 + Math.sin(frameIndex * 0.7) * 6 + (frameIndex > 9 ? 18 : 0);
  const landmarks = buildHandLandmarks(centerX, centerY);
  const occluding = isHandNearActiveRegion(centerX, centerY);

  state.hand = {
    status: occluding ? "occluding" : "detected",
    hint: occluding ? "靠近装配区域" : "关键点跟踪中",
    landmarks,
    occluding,
  };
}

function buildHandLandmarks(centerX, centerY) {
  const fingers = [
    [[-13, -4], [-19, -11], [-23, -18], [-26, -25]],
    [[-6, -9], [-9, -19], [-10, -28], [-11, -36]],
    [[1, -10], [1, -21], [1, -31], [2, -40]],
    [[8, -7], [11, -17], [14, -26], [17, -34]],
    [[14, 0], [20, -6], [25, -12], [30, -18]],
  ];
  const points = [{ x: centerX, y: centerY + 15 }];
  fingers.forEach((finger) => {
    finger.forEach(([x, y]) => points.push({ x: centerX + x, y: centerY + y }));
  });
  return points;
}

function isHandNearActiveRegion(centerX, centerY) {
  const region = activeRegion();
  if (!region) return false;
  const box = layout[region.region_id]?.box;
  if (!box) return false;
  return (
    centerX >= box.left
    && centerX <= box.left + box.width
    && centerY >= box.top
    && centerY <= box.top + box.height
  );
}

function activeRegion() {
  if (state.completed || !state.config) return null;
  return state.config.regions[state.regionIndex];
}

function expectedStep() {
  const region = activeRegion();
  if (!region) return null;
  return region.steps[state.stepIndex];
}

function isHighConfidencePresent(detection) {
  return detection.present && detection.confidence >= state.config.confidence_threshold;
}

function bestDetection(detections, expected) {
  return detections
    .filter((item) => item.hole_id === expected.hole_id && isHighConfidencePresent(item))
    .sort((a, b) => b.confidence - a.confidence)[0] || null;
}

function advance(frameIndex, region, expected) {
  state.stepIndex += 1;
  state.stableFrames = 0;
  state.elapsedFrames = 0;
  state.errorKeys = new Set();

  if (state.stepIndex < region.steps.length) return;

  pushEvent("region_completed", frameIndex, expected, null, `${region.name} 校验通过。`);
  state.regionIndex += 1;
  state.stepIndex = 0;

  if (state.regionIndex >= state.config.regions.length) {
    state.completed = true;
    pushEvent("all_completed", frameIndex, expected, null, "全部区域 SOP 校验完成。");
  }
}

function maybeMissing(frameIndex, expected) {
  if (state.elapsedFrames < state.config.missing_timeout_frames) return;
  emitOnce("missing_part", frameIndex, expected, null, `等待超时，${expected.hole_id} 未确认装配完成。`);
}

function emitOnce(type, frameIndex, expected, detection, message) {
  const key = `${type}:${detection?.hole_id || ""}:${detection?.part_type || ""}`;
  if (state.errorKeys.has(key)) return;
  state.errorKeys.add(key);
  pushEvent(type, frameIndex, expected, detection, message);
}

function pushEvent(type, frameIndex, expected, detection, message) {
  state.events.unshift({
    type,
    frameIndex,
    regionId: activeRegion()?.region_id || state.config.regions[Math.min(state.regionIndex, state.config.regions.length - 1)].region_id,
    expected,
    detection,
    message,
  });
}

function render() {
  renderSummary();
  renderRegions();
  renderStage();
  renderHandMonitor();
  renderEvents();
}

function renderSummary() {
  const region = activeRegion();
  const step = expectedStep();
  const totalSteps = state.config.regions.reduce((sum, item) => sum + item.steps.length, 0);
  const doneSteps = state.completedKeys.size;
  const errorCount = state.events.filter((item) => isErrorEvent(item.type)).length;

  els.activeRegion.textContent = region ? region.name : "全部完成";
  els.activeHole.textContent = step ? step.hole_id : "-";
  if (errorCount > 0) {
    els.runState.textContent = `异常 ${errorCount}`;
  } else if (state.completed) {
    els.runState.textContent = "完成";
  } else if (state.latestFrame !== null) {
    els.runState.textContent = "正常";
  } else {
    els.runState.textContent = state.timer ? "运行中" : "待机";
  }
  els.progressText.textContent = `${doneSteps} / ${totalSteps}`;
  els.frameText.textContent = `Frame ${state.latestFrame ?? "-"}`;
  els.doneCount.textContent = doneSteps;
  els.errorCount.textContent = errorCount;
  els.stableCount.textContent = state.stableFrames;
  els.eventCount.textContent = `${errorCount} 条`;
}

function renderRegions() {
  els.regionList.innerHTML = state.config.regions.map((region, regionIndex) => {
    const isRegionActive = regionIndex === state.regionIndex && !state.completed;
    const steps = region.steps.map((step, stepIndex) => {
      const key = `${region.region_id}:${step.hole_id}`;
      const isDone = state.completedKeys.has(key);
      const isStepActive = isRegionActive && stepIndex === state.stepIndex;
      const error = state.events.find((event) => (
        isErrorEvent(event.type)
        && event.regionId === region.region_id
        && event.expected.hole_id === step.hole_id
      ));
      const status = isDone ? "完成" : error ? "异常" : isStepActive ? "当前" : "等待";
      const statusClass = isDone ? "done" : error ? "error" : isStepActive ? "active" : "";
      return `
        <div class="step-row ${statusClass}">
          <div class="step-index">${step.step}</div>
          <div class="step-main">
            <div class="step-hole">${step.hole_id}</div>
            <div class="step-part">确认已装</div>
          </div>
          <div class="status-pill ${statusClass}">${status}</div>
        </div>
      `;
    }).join("");

    return `
      <div class="region-block">
        <div class="region-title ${isRegionActive ? "active" : ""}">
          <span>${region.name}</span>
          <span>${region.region_id}</span>
        </div>
        <div class="step-list">${steps}</div>
      </div>
    `;
  }).join("");
}

function renderStage() {
  const active = activeRegion();
  const expected = expectedStep();
  const detectedKeys = new Set(state.latestDetections.map((item) => `${item.region_id}:${item.hole_id}`));
  const errorHoleKeys = new Set(state.events
    .filter((item) => isErrorEvent(item.type))
    .map((item) => `${item.regionId}:${item.detection?.hole_id || item.expected.hole_id}`));

  els.stage.innerHTML = state.config.regions.map((region) => {
    const regionLayout = layout[region.region_id];
    const box = regionLayout.box;
    const holes = region.steps.map((step) => {
      const pos = regionLayout.holes[step.hole_id];
      const key = `${region.region_id}:${step.hole_id}`;
      const classes = ["hole"];
      if (state.completedKeys.has(key)) classes.push("done");
      if (detectedKeys.has(key)) classes.push("detected");
      if (errorHoleKeys.has(key)) classes.push("error");
      if (active?.region_id === region.region_id && expected?.hole_id === step.hole_id) classes.push("active");
      return `
        <div class="${classes.join(" ")}" style="left:${pos.left}%; top:${pos.top}%;">
          <span>${step.hole_id}</span>
        </div>
      `;
    }).join("");

    return `
      <div class="part-zone ${active?.region_id === region.region_id ? "active" : ""}"
        style="left:${box.left}%; top:${box.top}%; width:${box.width}%; height:${box.height}%;">
        <div class="zone-label">${region.name}</div>
        ${holes}
      </div>
    `;
  }).join("");
}

function renderHandMonitor() {
  const statusText = {
    idle: "手部状态",
    detected: "手部跟踪",
    occluding: "区域接近",
  };
  const statusBox = document.querySelector(".hand-status");
  statusBox.classList.toggle("detected", state.hand.status === "detected");
  statusBox.classList.toggle("occluding", state.hand.status === "occluding");
  els.handState.textContent = statusText[state.hand.status] || "手部状态";
  els.handHint.textContent = state.hand.hint;

  if (!state.hand.landmarks.length) {
    els.handLayer.innerHTML = "";
    return;
  }

  const bones = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
  ];
  const points = state.hand.landmarks;
  const palm = [0, 5, 9, 13, 17].map((index) => `${points[index].x},${points[index].y}`).join(" ");
  const occlusion = state.hand.occluding
    ? `<ellipse class="hand-occlusion" cx="${points[0].x}" cy="${points[0].y - 8}" rx="19" ry="26"></ellipse>`
    : "";
  const boneMarkup = bones.map(([from, to]) => (
    `<line class="hand-bone" x1="${points[from].x}" y1="${points[from].y}" x2="${points[to].x}" y2="${points[to].y}"></line>`
  )).join("");
  const pointMarkup = points.map((point) => (
    `<circle class="hand-point" cx="${point.x}" cy="${point.y}" r="1.25"></circle>`
  )).join("");

  els.handLayer.innerHTML = `
    ${occlusion}
    <polygon class="hand-palm" points="${palm}"></polygon>
    ${boneMarkup}
    ${pointMarkup}
  `;
}

function renderEvents() {
  const abnormalEvents = state.events.filter((event) => isErrorEvent(event.type));
  if (!abnormalEvents.length) {
    els.eventList.innerHTML = `<div class="empty">暂无异常</div>`;
    return;
  }

  els.eventList.innerHTML = abnormalEvents.map((event) => {
    return `
      <div class="event-item bad">
        <div class="event-line">
          <span>${eventName(event.type)}</span>
          <span>Frame ${event.frameIndex}</span>
        </div>
        <div class="event-step">${event.regionId} · 步骤 ${event.expected.step} · ${event.expected.hole_id}</div>
        <div class="event-msg">${event.message}</div>
      </div>
    `;
  }).join("");
}

function eventName(type) {
  const names = {
    step_completed: "孔位完成",
    region_completed: "区域通过",
    all_completed: "全部完成",
    order_error: "顺序异常",
    missing_part: "漏装异常",
  };
  return names[type] || type;
}

function isErrorEvent(type) {
  return ["order_error", "missing_part"].includes(type);
}
