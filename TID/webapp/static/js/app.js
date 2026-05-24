/**
 * app.js — TID İşaret Dili Web Uygulaması
 * Kamera, MediaPipe, WebSocket, UI yönetimi
 */

const state = {
  socket: null,
  connected: false,
  streaming: false,
  selectedModel: "efficientnet",
  confThreshold: 0.5,
  history: [],
  totalPredictions: 0,
  confidenceSum: 0,
  startTime: Date.now(),
  fps: 0,
  lastFrameTime: 0,
  frameCount: 0,
  stream: null,
  animationId: null,
  modelInfo: {},
  lastSendTime: 0,
  sendIntervalMs: 200,
  mpHands: null,
  handResults: null,
  mpProcessing: false,
  modelLoading: false,
  prevLabel: "",
};

const $ = (sel) => document.querySelector(sel);
const dom = {
  video: $("#videoElement"),
  overlay: $("#overlayCanvas"),
  placeholder: $("#videoPlaceholder"),
  btnCamera: $("#btnCamera"),
  btnScreen: $("#btnScreen"),
  btnStop: $("#btnStop"),
  btnClear: $("#btnClearHistory"),
  cameraSelect: $("#cameraSelect"),
  connectionBadge: $("#connectionBadge"),
  connectionText: $("#connectionText"),
  fpsDisplay: $("#fpsDisplay"),
  predLetter: $("#predictionLetter"),
  predSublabel: $("#predictionSublabel"),
  predLabel: $("#predictionLabel"),
  confBar: $("#confidenceBar"),
  confValue: $("#confidenceValue"),
  modelBadge: $("#modelBadge"),
  modelGrid: $("#modelGrid"),
  confThreshold: $("#confThreshold"),
  confThresholdValue: $("#confThresholdValue"),
  top5List: $("#top5List"),
  historyDisplay: $("#historyDisplay"),
  historyCount: $("#historyCount"),
  toastContainer: $("#toastContainer"),
  modelLoading: $("#modelLoading"),
  tcnProgress: $("#tcnProgress"),
  tcnProgressFill: $("#tcnProgressFill"),
  statPredictions: $("#statPredictions"),
  statLetters: $("#statLetters"),
  statAvgConf: $("#statAvgConf"),
  statUptime: $("#statUptime"),
};

// ═══════ TOAST ═══════
function showToast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  dom.toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }, 3000);
}

// ═══════ SOCKET.IO ═══════
function initSocket() {
  console.log("[WS] Bağlanılıyor...");
  state.socket = io({ transports: ["websocket", "polling"] });

  state.socket.on("connect", () => {
    console.log("[WS] Bağlandı!");
    state.connected = true;
    dom.connectionBadge.className = "connection-badge connected";
    dom.connectionText.textContent = "Bağlandı";
    showToast("Sunucuya bağlandı", "success");
    fetchModels();
  });

  state.socket.on("disconnect", () => {
    console.log("[WS] Bağlantı kesildi");
    state.connected = false;
    dom.connectionBadge.className = "connection-badge disconnected";
    dom.connectionText.textContent = "Bağlantı kesildi";
  });

  state.socket.on("prediction", (data) => {
    console.log("[Tahmin]", data.label, data.confidence?.toFixed(2));
    handlePrediction(data);
  });

  state.socket.on("prediction_error", (data) => {
    console.error("[Tahmin Hatası]", data.error);
  });

  state.socket.on("model_loaded", (data) => {
    console.log("[Model]", data.model, data.status);
    state.modelLoading = false;
    dom.modelLoading.classList.remove("visible");
    if (data.status === "ok") {
      showToast(`${data.model} modeli hazır`, "success");
    } else {
      showToast(`Model hatası: ${data.message}`, "error");
    }
  });

  state.socket.on("buffers_cleared", () => {
    console.log("[Buffer] Temizlendi");
  });
}

// ═══════ MODEL YÖNETİMİ ═══════
async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    state.modelInfo = await res.json();
    console.log("[Modeller]", Object.keys(state.modelInfo));
    renderModelGrid();
    // Varsayılan modeli yükle
    if (state.connected) {
      state.modelLoading = true;
      dom.modelLoading.classList.add("visible");
      state.socket.emit("load_model", { model: state.selectedModel });
    }
  } catch (e) {
    console.error("[Modeller] Hata:", e);
  }
}

function renderModelGrid() {
  dom.modelGrid.innerHTML = "";
  const order = ["efficientnet", "mobilenet", "random_forest", "tcn"];
  for (const key of order) {
    const info = state.modelInfo[key];
    if (!info) continue;
    const el = document.createElement("div");
    el.className = `model-option ${key === state.selectedModel ? "selected" : ""} ${!info.available ? "unavailable" : ""}`;
    el.dataset.model = key;
    const typeLabel = info.type === "kelime" ? "🔤 Kelime" : "🔠 Harf";
    el.innerHTML = `
      <div class="model-option-name">${info.name}</div>
      <div class="model-option-type">${typeLabel}</div>
      <div class="model-option-accuracy">%${info.accuracy.toFixed(1)}</div>
    `;
    if (info.available) {
      el.addEventListener("click", () => selectModel(key));
    }
    dom.modelGrid.appendChild(el);
  }
}

function selectModel(key) {
  state.selectedModel = key;
  state.prevLabel = "";
  document.querySelectorAll(".model-option").forEach((el) => {
    el.classList.toggle("selected", el.dataset.model === key);
  });
  const info = state.modelInfo[key];
  if (info) dom.modelBadge.textContent = info.name;
  dom.tcnProgress.classList.toggle("visible", key === "tcn");
  if (state.connected) {
    state.modelLoading = true;
    dom.modelLoading.classList.add("visible");
    state.socket.emit("load_model", { model: key });
    state.socket.emit("clear_buffers");
  }
  showToast(`Model: ${info?.name || key}`, "info");
}

// ═══════ KAMERA / EKRAN ═══════
async function listCameras() {
  try {
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
    tempStream.getTracks().forEach((t) => t.stop());
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((d) => d.kind === "videoinput");
    dom.cameraSelect.innerHTML = cameras.length
      ? cameras.map((c, i) => `<option value="${c.deviceId}">${c.label || "Kamera " + (i + 1)}</option>`).join("")
      : '<option value="">Kamera bulunamadı</option>';
  } catch (e) {
    console.warn("Kamera listesi:", e);
  }
}

async function startCamera() {
  try {
    const deviceId = dom.cameraSelect.value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    });
    dom.video.srcObject = state.stream;
    await dom.video.play();
    onStreamStarted();
    showToast("Kamera başlatıldı", "success");
  } catch (e) {
    showToast("Kamera hatası: " + e.message, "error");
  }
}

async function startScreen() {
  try {
    state.stream = await navigator.mediaDevices.getDisplayMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
    });
    dom.video.srcObject = state.stream;
    await dom.video.play();
    state.stream.getVideoTracks()[0].addEventListener("ended", () => {
      stopStream();
      showToast("Ekran paylaşımı sonlandırıldı", "info");
    });
    onStreamStarted();
    showToast("Ekran paylaşımı başlatıldı", "success");
  } catch (e) {
    showToast("Ekran paylaşımı hatası: " + e.message, "error");
  }
}

function onStreamStarted() {
  state.streaming = true;
  dom.placeholder.style.display = "none";
  dom.btnCamera.style.display = "none";
  dom.btnScreen.style.display = "none";
  dom.btnStop.style.display = "inline-flex";
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  initMediaPipe();
  frameLoop();
}

function stopStream() {
  state.streaming = false;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  dom.video.srcObject = null;
  dom.placeholder.style.display = "";
  dom.btnCamera.style.display = "inline-flex";
  dom.btnScreen.style.display = "inline-flex";
  dom.btnStop.style.display = "none";
  if (state.animationId) { cancelAnimationFrame(state.animationId); state.animationId = null; }
  window.removeEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
  const vw = dom.video.videoWidth || 640;
  const vh = dom.video.videoHeight || 480;
  dom.overlay.width = vw;
  dom.overlay.height = vh;
}

// ═══════ MEDIAPIPE HANDS ═══════
function initMediaPipe() {
  if (state.mpHands) return;

  // MediaPipe yüklenmiş mi kontrol et
  if (typeof Hands === "undefined") {
    console.error("[MediaPipe] Hands kütüphanesi yüklenemedi! Random Forest ve TCN modelleri el noktaları olmadan çalışmaz.");
    showToast("⚠️ MediaPipe yüklenemedi — Random Forest/TCN çalışmayabilir", "error");
    return;
  }

  console.log("[MediaPipe] Başlatılıyor...");
  try {
    state.mpHands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });
    state.mpHands.setOptions({
      maxNumHands: 2, modelComplexity: 1,
      minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    state.mpHands.onResults((results) => {
      state.handResults = results;
      state.mpProcessing = false;
      drawHandOverlay(results);
    });
    console.log("[MediaPipe] Hands başlatıldı ✓");
    showToast("MediaPipe Hands başlatıldı ✓", "success");
  } catch (err) {
    console.error("[MediaPipe] Başlatma hatası:", err);
    showToast("MediaPipe başlatma hatası: " + err.message, "error");
    state.mpHands = null;
  }
}

function drawHandOverlay(results) {
  const ctx = dom.overlay.getContext("2d");
  ctx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
  if (!results.multiHandLandmarks) return;

  // drawing_utils yüklü mü kontrol et
  const hasDrawing = typeof drawConnectors === "function" && typeof drawLandmarks === "function";
  const hasConnections = typeof HAND_CONNECTIONS !== "undefined";

  for (const landmarks of results.multiHandLandmarks) {
    if (hasDrawing && hasConnections) {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "rgba(99,102,241,0.6)", lineWidth: 2 });
      drawLandmarks(ctx, landmarks, { color: "rgba(139,92,246,0.8)", lineWidth: 1, radius: 3 });
    } else {
      // Fallback: basit nokta çizimi
      ctx.fillStyle = "rgba(139,92,246,0.8)";
      for (const pt of landmarks) {
        const x = pt.x * dom.overlay.width;
        const y = pt.y * dom.overlay.height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

// ═══════ FRAME DÖNGÜSÜ ═══════
function frameLoop() {
  if (!state.streaming) return;
  state.animationId = requestAnimationFrame(() => {
    const now = performance.now();

    // FPS
    state.frameCount++;
    if (now - state.lastFrameTime >= 1000) {
      state.fps = state.frameCount;
      state.frameCount = 0;
      state.lastFrameTime = now;
      dom.fpsDisplay.textContent = state.fps + " FPS";
    }

    // MediaPipe — async, beklemeden
    if (state.mpHands && !state.mpProcessing && dom.video.readyState >= 2) {
      state.mpProcessing = true;
      state.mpHands.send({ image: dom.video }).catch(() => { state.mpProcessing = false; });
    }

    // Backend'e gönder (throttled) — model yüklendiyse
    if (now - state.lastSendTime >= state.sendIntervalMs && state.connected && !state.modelLoading) {
      state.lastSendTime = now;
      sendToBackend();
    }

    frameLoop();
  });
}

function sendToBackend() {
  if (!state.connected || !state.streaming || state.modelLoading) return;
  const model = state.selectedModel;

  if (model === "random_forest") {
    sendLandmarks();
  } else if (model === "tcn") {
    sendTCNKeypoints();
  } else {
    sendFrame(model);
  }
}

function sendFrame(model) {
  const vw = dom.video.videoWidth;
  const vh = dom.video.videoHeight;
  if (!vw || !vh) return;

  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");

  // Tüm frame'i gönder (ROI yerine tam görüntü)
  ctx.drawImage(dom.video, 0, 0, vw, vh, 0, 0, 224, 224);

  canvas.toBlob((blob) => {
    if (!blob) return;
    const reader = new FileReader();
    reader.onload = () => {
      const b64 = reader.result.split(",")[1];
      state.socket.emit("predict_frame", { model: model, frame: b64 });
    };
    reader.readAsDataURL(blob);
  }, "image/jpeg", 0.85);
}

function sendLandmarks() {
  if (!state.handResults || !state.handResults.multiHandLandmarks) return;
  const hands = state.handResults.multiHandLandmarks;
  const handedness = state.handResults.multiHandedness;
  if (!hands.length) return;

  function normalizeLandmarks(lm) {
    const wrist = lm[0];
    const features = [];
    for (const pt of lm) {
      features.push(pt.x - wrist.x);
      features.push(pt.y - wrist.y);
      features.push(pt.z - wrist.z);
    }
    return features;
  }

  let landmarks;
  const handCount = hands.length;
  if (handCount === 1) {
    landmarks = normalizeLandmarks(hands[0]);
  } else {
    let sol = Array(63).fill(0), sag = Array(63).fill(0);
    for (let i = 0; i < hands.length; i++) {
      const label = handedness[i]?.label || "Right";
      const feat = normalizeLandmarks(hands[i]);
      if (label === "Left") sol = feat; else sag = feat;
    }
    landmarks = [...sol, ...sag];
  }
  state.socket.emit("predict_landmarks", { landmarks, hand_count: handCount });
}

function sendTCNKeypoints() {
  if (!state.handResults) return;
  const hands = state.handResults.multiHandLandmarks || [];
  const handedness = state.handResults.multiHandedness || [];
  const features = new Array(138).fill(0);
  if (hands.length > 0) {
    for (let i = 0; i < hands.length; i++) {
      const label = handedness[i]?.label || "Right";
      const lm = hands[i];
      const offset = label === "Left" ? 0 : 63;
      for (let j = 0; j < 21; j++) {
        features[offset + j * 3] = lm[j].x;
        features[offset + j * 3 + 1] = lm[j].y;
        features[offset + j * 3 + 2] = lm[j].z;
      }
    }
  }
  state.socket.emit("predict_tcn", { keypoints: features });
}

// ═══════ TAHMİN SONUÇLARI ═══════
function handlePrediction(data) {
  const { label, confidence, top5, buffer_progress } = data;
  const displayLabel = confidence >= state.confThreshold ? label : "?";

  // Tahmin göster
  dom.predLetter.textContent = displayLabel;
  if (dom.predSublabel) {
    dom.predSublabel.textContent = displayLabel === "?" ? "" : displayLabel;
  }
  dom.predLabel.textContent =
    displayLabel === "?" ? "Düşük güven" :
    displayLabel === "Bekleniyor..." ? "Bekleniyor..." : "Algılandı";

  // Güven barı
  const pct = Math.round(confidence * 100);
  dom.confBar.style.width = pct + "%";
  dom.confBar.className = `confidence-bar-fill ${confidence < 0.5 ? "low" : ""}`;
  dom.confValue.textContent = "%" + pct;
  dom.confValue.className = `confidence-value ${confidence < 0.5 ? "low" : ""}`;

  // Top 5
  if (top5 && top5.length > 0) updateTop5(top5);

  // TCN ilerleme
  if (buffer_progress !== undefined && state.selectedModel === "tcn") {
    dom.tcnProgressFill.style.width = Math.round(buffer_progress * 100) + "%";
  }

  // Geçmiş
  if (displayLabel !== "?" && displayLabel !== "Bekleniyor..." && displayLabel !== state.prevLabel) {
    state.prevLabel = displayLabel;
    addToHistory(displayLabel);
  }

  // İstatistik
  state.totalPredictions++;
  state.confidenceSum += confidence;
  updateStats();
}

function updateTop5(top5) {
  const items = dom.top5List.querySelectorAll(".top5-item");
  for (let i = 0; i < 5; i++) {
    const item = items[i];
    if (!item) break;
    const entry = top5[i];
    if (entry) {
      item.querySelector(".top5-label").textContent = entry.label;
      const pct = Math.round(entry.confidence * 100);
      item.querySelector(".top5-bar-fill").style.width = pct + "%";
      item.querySelector(".top5-conf").textContent = "%" + pct;
    }
  }
}

function addToHistory(label) {
  state.history.push(label);
  if (state.history.length > 50) state.history.shift();
  renderHistory();
}

function renderHistory() {
  if (state.history.length === 0) {
    dom.historyDisplay.innerHTML = '<div class="history-empty">Henüz algılama yapılmadı</div>';
    dom.historyCount.textContent = "0";
    return;
  }
  const info = state.modelInfo[state.selectedModel];
  const isWord = info?.type === "kelime";
  dom.historyDisplay.innerHTML = '<div class="history-letters">' +
    state.history.map((h) =>
      isWord ? `<span class="history-word">${h}</span>` : `<span class="history-letter">${h}</span>`
    ).join("") + "</div>";
  dom.historyCount.textContent = state.history.length;
}

function updateStats() {
  dom.statPredictions.textContent = state.totalPredictions;
  dom.statLetters.textContent = state.history.length;
  const avg = state.totalPredictions > 0 ? (state.confidenceSum / state.totalPredictions) * 100 : 0;
  dom.statAvgConf.textContent = "%" + Math.round(avg);
}

// ═══════ EVENT'LER ═══════
function initEvents() {
  dom.btnCamera.addEventListener("click", startCamera);
  dom.btnScreen.addEventListener("click", startScreen);
  dom.btnStop.addEventListener("click", stopStream);
  dom.btnClear.addEventListener("click", () => {
    state.history = [];
    state.prevLabel = "";
    renderHistory();
    if (state.connected) state.socket.emit("clear_buffers");
  });
  dom.confThreshold.addEventListener("input", (e) => {
    state.confThreshold = parseFloat(e.target.value);
    dom.confThresholdValue.textContent = "%" + Math.round(state.confThreshold * 100);
  });
  // Uptime
  setInterval(() => {
    const sec = Math.round((Date.now() - state.startTime) / 1000);
    const m = Math.floor(sec / 60), s = sec % 60;
    dom.statUptime.textContent = m > 0 ? `${m}dk ${s}s` : `${s}s`;
  }, 1000);
}

// ═══════ BAŞLAT ═══════
document.addEventListener("DOMContentLoaded", () => {
  initSocket();
  initEvents();
  listCameras();
  dom.modelBadge.textContent = "EfficientNetB0";
});
