// AI SOP 前端监控台界面逻辑。
// 本文件负责加载 SOP 配置、展示区域/孔位/异常记录，并接入后端摄像头画面。
// 监控控制按钮当前只做界面预留，真实开始、暂停、恢复、复位、异常确认逻辑后续接入后端。

const state = {
  config: null,
  regionIndex: 0,
  stepIndex: 0,
  stableFrames: 0,
  completed: false,
  completedKeys: new Set(),
  events: [],
  latestDetections: [],
  latestFrame: null,
  cameraLive: false,
  hand: {
    status: "idle",
    hint: "画面稳定",
    landmarks: [],
    occluding: false,
  },
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
  resetRuntime();
  startCameraFeed();
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
    "startMonitorBtn",
    "pauseMonitorBtn",
    "resumeMonitorBtn",
    "finishMonitorBtn",
    "ackErrorBtn",
    "cameraToggleBtn",
    "doneCount",
    "errorCount",
    "stableCount",
    "eventList",
    "eventCount",
    "handState",
    "handHint",
    "handLayer",
    "cameraFeed",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindActions() {
  els.startMonitorBtn.addEventListener("click", markMonitorControlPending);
  els.pauseMonitorBtn.addEventListener("click", markMonitorControlPending);
  els.resumeMonitorBtn.addEventListener("click", markMonitorControlPending);
  els.finishMonitorBtn.addEventListener("click", markMonitorControlPending);
  els.ackErrorBtn.addEventListener("click", markMonitorControlPending);
  els.cameraToggleBtn.addEventListener("click", toggleCameraFeed);
}

function markMonitorControlPending() {
  els.runState.textContent = "待接入";
}

function toggleCameraFeed() {
  if (state.cameraLive) {
    stopCameraFeed();
    return;
  }
  startCameraFeed();
}

function startCameraFeed() {
  if (!els.cameraFeed || window.location.protocol === "file:") return;
  state.cameraLive = true;
  els.cameraFeed.src = `/camera.mjpg?t=${Date.now()}`;
  els.cameraFeed.classList.add("live");
  els.cameraToggleBtn.classList.add("active");
  els.cameraToggleBtn.textContent = "关闭";
}

function stopCameraFeed() {
  state.cameraLive = false;
  if (els.cameraFeed) {
    els.cameraFeed.removeAttribute("src");
    els.cameraFeed.classList.remove("live");
  }
  els.cameraToggleBtn.classList.remove("active");
  els.cameraToggleBtn.textContent = "摄像头";
}

async function loadConfig() {
  const response = await fetch("../configs/sample_sop.json");
  state.config = await response.json();
}

function resetRuntime() {
  state.regionIndex = 0;
  state.stepIndex = 0;
  state.stableFrames = 0;
  state.completed = false;
  state.completedKeys = new Set();
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

function activeRegion() {
  if (state.completed || !state.config) return null;
  return state.config.regions[state.regionIndex];
}

function expectedStep() {
  const region = activeRegion();
  if (!region) return null;
  return region.steps[state.stepIndex];
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
    els.runState.textContent = "待机";
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
