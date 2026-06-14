/**
 * TID İşaret Dili — GitHub Pages Frontend
 * Kamera, MediaPipe Holistic, Hugging Face API iletişimi
 */

// ═══════════════════════════════════════
// ★★★ HUGGING FACE SPACE URL'İNİ BURAYA YAZ ★★★
// ═══════════════════════════════════════
const API_BASE = "https://hidrqx-tid-api.hf.space";
// Örnek: "https://semihesim-tid-api.hf.space"


// ═══════ STATE ═══════
const state = {
  selectedModel: "rf",
  streaming: false,
  stream: null,
  holistic: null,
  holisticReady: false,
  animationId: null,
  apiConnected: false,
  // RF
  lastSendTime: 0,
  sendInterval: 400,
  prevLabel: "",
  // CNN1D
  cnnRecording: false,
  cnnSequence: [],
  cnnStartTime: 0,
  cnnDuration: 3.0,
  cnnCooldown: false,
  // History
  history: [],
  // Holistic results
  results: null,
};

const $ = (s) => document.querySelector(s);
const dom = {
  video: $("#videoElement"),
  canvas: $("#overlayCanvas"),
  placeholder: $("#videoPlaceholder"),
  btnCamera: $("#btnCamera"),
  btnStop: $("#btnStop"),
  btnClear: $("#btnClear"),
  tabRF: $("#tabRF"),
  tabCNN: $("#tabCNN"),
  predLetter: $("#predLetter"),
  predLabel: $("#predLabel"),
  confFill: $("#confFill"),
  confPct: $("#confPct"),
  top5List: $("#top5List"),
  historyItems: $("#historyItems"),
  apiDot: $("#apiDot"),
  apiText: $("#apiText"),
  recordingBar: $("#recordingBar"),
  recordingFill: $("#recordingFill"),
  recordingTime: $("#recordingTime"),
  videoWrapper: $("#videoWrapper"),
};


// ═══════ API ═══════
async function checkAPI() {
  try {
    const res = await fetch(API_BASE + "/", { signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      state.apiConnected = true;
      dom.apiDot.classList.add("connected");
      dom.apiDot.classList.remove("error");
      dom.apiText.textContent = "API bağlı ✓";
      return true;
    }
  } catch (e) {
    console.warn("[API] Bağlantı hatası:", e.message);
  }
  state.apiConnected = false;
  dom.apiDot.classList.add("error");
  dom.apiDot.classList.remove("connected");
  dom.apiText.textContent = "API bağlanamadı — URL'yi kontrol edin";
  return false;
}

async function predictRF(landmarks, handCount) {
  try {
    const res = await fetch(API_BASE + "/predict/rf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks, hand_count: handCount }),
    });
    return await res.json();
  } catch (e) {
    console.error("[RF]", e.message);
    return null;
  }
}

async function predictCNN1D(sequence) {
  try {
    const res = await fetch(API_BASE + "/predict/cnn1d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    return await res.json();
  } catch (e) {
    console.error("[CNN1D]", e.message);
    return null;
  }
}


// ═══════ MEDIAPIPE HOLISTIC ═══════
function initHolistic() {
  if (state.holistic) return;
  if (typeof Holistic === "undefined") {
    console.error("[MediaPipe] Holistic yüklenemedi!");
    return;
  }

  state.holistic = new Holistic({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
  });

  state.holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  state.holistic.onResults((results) => {
    state.results = results;
    state.holisticReady = true;
    drawLandmarks(results);
  });

  console.log("[MediaPipe] Holistic başlatıldı");
}

function drawLandmarks(results) {
  const ctx = dom.canvas.getContext("2d");
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);

  const hasDrawing = typeof drawConnectors === "function" && typeof drawLandmarks === "function";

  // Hands
  if (results.leftHandLandmarks && hasDrawing) {
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "rgba(99,102,241,0.7)", lineWidth: 2 });
    window.drawLandmarks(ctx, results.leftHandLandmarks, { color: "rgba(139,92,246,0.9)", lineWidth: 1, radius: 3 });
  }
  if (results.rightHandLandmarks && hasDrawing) {
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "rgba(6,182,212,0.7)", lineWidth: 2 });
    window.drawLandmarks(ctx, results.rightHandLandmarks, { color: "rgba(16,185,129,0.9)", lineWidth: 1, radius: 3 });
  }
  // Pose (minimal)
  if (results.poseLandmarks && hasDrawing) {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "rgba(255,255,255,0.15)", lineWidth: 1 });
  }
}


// ═══════ LANDMARK EXTRACTION ═══════
function extractRFLandmarks(results) {
  const hands = [];
  let handCount = 0;

  function normalizeHand(lm) {
    const w = lm[0];
    const feat = [];
    for (const pt of lm) {
      feat.push(pt.x - w.x, pt.y - w.y, pt.z - w.z);
    }
    return feat;
  }

  if (results.rightHandLandmarks) {
    hands.push(...normalizeHand(results.rightHandLandmarks));
    handCount++;
  } else {
    hands.push(...new Array(63).fill(0));
  }

  if (results.leftHandLandmarks) {
    if (handCount === 0) {
      // Sol el varsa sağ elin yerine koy
      hands.splice(0, 63, ...normalizeHand(results.leftHandLandmarks));
    } else {
      hands.push(...normalizeHand(results.leftHandLandmarks));
    }
    handCount++;
  } else if (handCount === 1) {
    hands.push(...new Array(63).fill(0));
  }

  return { landmarks: hands.slice(0, handCount === 2 ? 126 : 63), hand_count: handCount };
}

function extractCNN1DKeypoints(results) {
  const lh = results.leftHandLandmarks
    ? results.leftHandLandmarks.flatMap((p) => [p.x, p.y, p.z])
    : new Array(63).fill(0);
  const rh = results.rightHandLandmarks
    ? results.rightHandLandmarks.flatMap((p) => [p.x, p.y, p.z])
    : new Array(63).fill(0);
  const pose = results.poseLandmarks
    ? results.poseLandmarks.flatMap((p) => [p.x, p.y, p.z])
    : new Array(99).fill(0);
  return [...lh, ...rh, ...pose]; // 225
}


// ═══════ CAMERA ═══════
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    dom.video.srcObject = state.stream;
    await dom.video.play();
    state.streaming = true;
    dom.placeholder.style.display = "none";
    dom.btnCamera.style.display = "none";
    dom.btnStop.style.display = "inline-flex";
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    initHolistic();
    frameLoop();
  } catch (e) {
    alert("Kamera hatası: " + e.message);
  }
}

function stopCamera() {
  state.streaming = false;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  dom.video.srcObject = null;
  dom.placeholder.style.display = "";
  dom.btnCamera.style.display = "inline-flex";
  dom.btnStop.style.display = "none";
  if (state.animationId) cancelAnimationFrame(state.animationId);
  window.removeEventListener("resize", resizeCanvas);
  state.cnnRecording = false;
  state.cnnSequence = [];
  dom.recordingBar.classList.remove("visible");
}

function resizeCanvas() {
  dom.canvas.width = dom.video.videoWidth || 640;
  dom.canvas.height = dom.video.videoHeight || 480;
}


// ═══════ FRAME LOOP ═══════
let mpBusy = false;

function frameLoop() {
  if (!state.streaming) return;
  state.animationId = requestAnimationFrame(() => {
    // Send to MediaPipe
    if (state.holistic && !mpBusy && dom.video.readyState >= 2) {
      mpBusy = true;
      state.holistic.send({ image: dom.video }).then(() => { mpBusy = false; }).catch(() => { mpBusy = false; });
    }

    const now = performance.now();

    if (state.results && state.apiConnected) {
      if (state.selectedModel === "rf") {
        handleRF(now);
      } else {
        handleCNN1D(now);
      }
    }

    frameLoop();
  });
}


// ═══════ RF HANDLER ═══════
async function handleRF(now) {
  if (now - state.lastSendTime < state.sendInterval) return;
  const results = state.results;
  if (!results.leftHandLandmarks && !results.rightHandLandmarks) return;

  state.lastSendTime = now;
  const { landmarks, hand_count } = extractRFLandmarks(results);
  if (hand_count === 0) return;

  const pred = await predictRF(landmarks, hand_count);
  if (pred) showPrediction(pred);
}


// ═══════ CNN1D HANDLER ═══════
function handleCNN1D(now) {
  const results = state.results;
  const hasHand = results.leftHandLandmarks || results.rightHandLandmarks;

  if (!state.cnnRecording && hasHand && !state.cnnCooldown) {
    // Auto-start recording
    state.cnnRecording = true;
    state.cnnSequence = [];
    state.cnnStartTime = performance.now();
    dom.recordingBar.classList.add("visible");
  }

  if (state.cnnRecording) {
    const kp = extractCNN1DKeypoints(results);
    state.cnnSequence.push(kp);

    const elapsed = (now - state.cnnStartTime) / 1000;
    const ratio = Math.min(1, elapsed / state.cnnDuration);
    dom.recordingFill.style.width = (ratio * 100) + "%";
    dom.recordingTime.textContent = `${elapsed.toFixed(1)}/${state.cnnDuration}s`;

    if (elapsed >= state.cnnDuration) {
      finishCNN1DRecording();
    }
  }
}

async function finishCNN1DRecording() {
  state.cnnRecording = false;
  state.cnnCooldown = true;
  dom.recordingBar.classList.remove("visible");

  dom.predLetter.textContent = "⏳";
  dom.predLabel.textContent = "Analiz ediliyor...";

  const pred = await predictCNN1D(state.cnnSequence);
  if (pred) showPrediction(pred);

  state.cnnSequence = [];

  // 1.5sn bekleme
  setTimeout(() => { state.cnnCooldown = false; }, 1500);
}


// ═══════ UI UPDATES ═══════
function showPrediction(pred) {
  const label = pred.label;
  const conf = pred.confidence;
  const pct = Math.round(conf * 100);

  dom.predLetter.textContent = label.toUpperCase();
  dom.predLabel.textContent = conf > 0.5 ? "Algılandı" : "Düşük güven";
  dom.confFill.style.width = pct + "%";
  dom.confPct.textContent = "%" + pct;

  // Top 5
  if (pred.top5 && pred.top5.length) {
    dom.top5List.innerHTML = pred.top5.map((t, i) => {
      const p = Math.round(t.confidence * 100);
      return `<div class="top5-item">
        <span class="top5-rank ${i === 0 ? "rank-1" : ""}">${i + 1}</span>
        <span class="top5-label">${t.label}</span>
        <div class="top5-bar"><div class="top5-bar-fill" style="width:${p}%"></div></div>
        <span class="top5-conf">%${p}</span>
      </div>`;
    }).join("");
  }

  // History
  if (conf > 0.5 && label !== state.prevLabel) {
    state.prevLabel = label;
    state.history.push(label);
    if (state.history.length > 40) state.history.shift();
    renderHistory();
  }
}

function renderHistory() {
  if (!state.history.length) {
    dom.historyItems.innerHTML = '<span class="history-empty">Henüz algılama yok</span>';
    return;
  }
  const isWord = state.selectedModel === "cnn1d";
  dom.historyItems.innerHTML = state.history
    .map((h) =>
      `<span class="history-item ${isWord ? "word" : ""}">${h.toUpperCase()}</span>`
    )
    .join("");
}


// ═══════ MODEL SWITCH ═══════
function selectModel(model) {
  state.selectedModel = model;
  state.prevLabel = "";
  state.cnnRecording = false;
  state.cnnSequence = [];
  dom.recordingBar.classList.remove("visible");

  dom.tabRF.classList.toggle("active", model === "rf");
  dom.tabCNN.classList.toggle("active", model === "cnn1d");

  dom.predLetter.textContent = "?";
  dom.predLabel.textContent = "Bekleniyor...";
  dom.confFill.style.width = "0%";
  dom.confPct.textContent = "%0";
  dom.top5List.innerHTML = "";
}


// ═══════ EVENTS ═══════
function init() {
  dom.btnCamera.addEventListener("click", startCamera);
  dom.btnStop.addEventListener("click", stopCamera);
  dom.btnClear.addEventListener("click", () => {
    state.history = [];
    state.prevLabel = "";
    renderHistory();
  });
  dom.tabRF.addEventListener("click", () => selectModel("rf"));
  dom.tabCNN.addEventListener("click", () => selectModel("cnn1d"));

  // Smooth scroll
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      const target = document.querySelector(a.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Navbar scroll effect
  window.addEventListener("scroll", () => {
    const nav = document.querySelector(".navbar");
    nav.style.background = window.scrollY > 50
      ? "rgba(5, 8, 22, 0.95)"
      : "rgba(5, 8, 22, 0.8)";
  });

  // API bağlantı kontrolü
  checkAPI();
  setInterval(checkAPI, 30000);
}

document.addEventListener("DOMContentLoaded", init);
