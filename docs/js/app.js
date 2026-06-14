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
  cnnSentence: [],
  cnnLastHandTime: 0,
  // History
  history: [],
  // Holistic results
  results: null,
  // New features
  soundEnabled: false,
  isScreen: false,
  rfBuffer: [],
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
  btnScreen: $("#btnScreen"),
  btnSound: $("#btnSound"),
  soundIcon: $("#soundIcon"),
  sentenceBox: document.createElement("div"), // Dinamik cümle kutusu
};

// Cümle kutusunu UI'a ekle (Sadece CNN modunda görünür olacak)
dom.sentenceBox.className = "cnn-sentence-box";
dom.sentenceBox.style.display = "none";
dom.sentenceBox.innerHTML = `<strong>Kurulan Cümle:</strong> <span id="cnnSentenceText"></span>`;
document.querySelector(".demo-result-card").insertBefore(dom.sentenceBox, document.querySelector(".history-box"));


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
    selfieMode: !state.isScreen, // Ekran paylaşımında aynalamayı kapat
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
  let leftHand = new Array(63).fill(0);
  let rightHand = new Array(63).fill(0);
  let handCount = 0;

  function normalizeHand(lm) {
    const w = lm[0];
    const feat = [];
    for (const pt of lm) {
      feat.push(pt.x - w.x, pt.y - w.y, pt.z - w.z);
    }
    return feat;
  }

  if (results.leftHandLandmarks) {
    leftHand = normalizeHand(results.leftHandLandmarks);
    handCount++;
  }
  
  if (results.rightHandLandmarks) {
    rightHand = normalizeHand(results.rightHandLandmarks);
    handCount++;
  }

  if (handCount === 0) return { landmarks: [], hand_count: 0 };

  if (handCount === 1) {
    // Python'da 1 el olduğunda sadece o eli 63 feature olarak döndürür
    const activeHand = results.leftHandLandmarks ? leftHand : rightHand;
    return { landmarks: activeHand, hand_count: 1 };
  }

  // 2 el varsa önce sol el sonra sağ el (Python: sol_el + sag_el)
  return { landmarks: [...leftHand, ...rightHand], hand_count: 2 };
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


// ═══════ CAMERA & SCREEN SHARE ═══════
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    state.isScreen = false;
    dom.video.classList.add("mirrored");
    dom.canvas.classList.add("mirrored");
    await setupStream();
  } catch (e) {
    alert("Kamera hatası: " + e.message);
  }
}

async function startScreenShare() {
  try {
    state.stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 30 } },
    });
    state.isScreen = true;
    dom.video.classList.remove("mirrored");
    dom.canvas.classList.remove("mirrored");
    await setupStream();
  } catch (e) {
    alert("Ekran paylaşımı hatası: " + e.message);
  }
}

async function setupStream() {
  dom.video.srcObject = state.stream;
  await dom.video.play();
  state.streaming = true;
  dom.placeholder.style.display = "none";
  dom.btnCamera.style.display = "none";
  dom.btnScreen.style.display = "none";
  dom.btnStop.style.display = "inline-flex";
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  
  if (state.holistic) {
    state.holistic.setOptions({ selfieMode: !state.isScreen });
  } else {
    initHolistic();
  }
  frameLoop();
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
  dom.btnScreen.style.display = "inline-flex";
  dom.btnStop.style.display = "none";
  if (state.animationId) cancelAnimationFrame(state.animationId);
  window.removeEventListener("resize", resizeCanvas);
  state.cnnRecording = false;
  state.cnnSequence = [];
  state.cnnSentence = [];
  state.rfBuffer = [];
  dom.recordingBar.classList.remove("visible");
  if (document.getElementById("cnnSentenceText")) {
    document.getElementById("cnnSentenceText").textContent = "";
  }
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
  if (pred) {
    // Buffer logic
    state.rfBuffer.push(pred);
    if (state.rfBuffer.length > 7) state.rfBuffer.shift();

    const counts = {};
    for (const p of state.rfBuffer) counts[p.label] = (counts[p.label] || 0) + 1;

    let maxLabel = "?";
    let maxCount = 0;
    for (const [lbl, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        maxLabel = lbl;
      }
    }

    if (maxCount >= 4 && maxLabel !== "?") {
      const stablePred = state.rfBuffer.find((p) => p.label === maxLabel);
      showPrediction(stablePred, true);
    } else {
      showPrediction(pred, false);
    }
  }
}


// ═══════ CNN1D HANDLER (Cümle Kurma Mantığı) ═══════
function handleCNN1D(now) {
  const results = state.results;
  const hasHand = results.leftHandLandmarks || results.rightHandLandmarks;

  // Cümle Bitirme Kontrolü (El 3 saniye boyunca yoksa)
  if (hasHand) {
    state.cnnLastHandTime = now;
  } else if (state.cnnLastHandTime > 0 && now - state.cnnLastHandTime > 3000) {
    if (state.cnnSentence.length > 0) {
      const finalSentence = state.cnnSentence.join(" ");
      speak(finalSentence); // Sadece cümle bitince oku
      
      // Geçmişe cümleyi ekle
      state.history.push("💬 " + finalSentence);
      if (state.history.length > 40) state.history.shift();
      renderHistory();
      
      state.cnnSentence = [];
      document.getElementById("cnnSentenceText").textContent = "";
      
      dom.predLetter.textContent = "✓";
      dom.predLabel.textContent = "Cümle Tamamlandı";
    }
    state.cnnLastHandTime = 0;
  }

  if (!state.cnnRecording && hasHand && !state.cnnCooldown) {
    // Auto-start recording
    state.cnnRecording = true;
    state.cnnSequence = [];
    state.cnnStartTime = performance.now();
    dom.recordingBar.classList.add("visible");
    dom.predLetter.textContent = "KAYIT";
    dom.predLabel.textContent = "Hareket bekleniyor...";
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
  if (pred) {
    showCNNPrediction(pred);
  }

  state.cnnSequence = [];

  // Lokaldeki gibi 1 saniye bekleme süresi
  setTimeout(() => { state.cnnCooldown = false; }, 1000);
}


// ═══════ UI & AUDIO UPDATES ═══════
function showPrediction(pred, isStable) {
  // Sadece RF için kullanılır
  if (state.selectedModel !== "rf") return;

  const label = pred.label;
  const conf = pred.confidence;
  const pct = Math.round(conf * 100);

  dom.predLetter.textContent = label.toUpperCase();
  dom.predLabel.textContent = isStable && conf > 0.5 ? "Algılandı" : (conf > 0.5 ? "Sabit tut..." : "Düşük güven");
  dom.confFill.style.width = pct + "%";
  dom.confPct.textContent = "%" + pct;

  updateTop5(pred);

  // History & Sound (Sadece stabilse ve değiştiyse)
  if (isStable && conf > 0.5 && label !== state.prevLabel) {
    state.prevLabel = label;
    state.history.push(label);
    if (state.history.length > 40) state.history.shift();
    renderHistory();
    speak(label);
  }
}

function showCNNPrediction(pred) {
  // Sadece CNN için kullanılır
  const label = pred.label;
  const conf = pred.confidence;
  const pct = Math.round(conf * 100);

  if (conf > 0.5) {
    dom.predLetter.textContent = label.toUpperCase();
    dom.predLabel.textContent = "Kelime Eklendi";
    
    // Aynı kelimenin art arda eklenmesini engelle
    if (state.cnnSentence.length === 0 || state.cnnSentence[state.cnnSentence.length - 1] !== label) {
      state.cnnSentence.push(label);
      document.getElementById("cnnSentenceText").textContent = state.cnnSentence.join(" ").toUpperCase();
    }
  } else {
    dom.predLetter.textContent = "?";
    dom.predLabel.textContent = "Anlaşılamadı";
  }

  dom.confFill.style.width = pct + "%";
  dom.confPct.textContent = "%" + pct;
  updateTop5(pred);
}

function updateTop5(pred) {
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
}

function speak(text) {
  if (!state.soundEnabled || !window.speechSynthesis) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "tr-TR";
  window.speechSynthesis.speak(utterance);
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
  state.cnnSentence = [];
  state.cnnLastHandTime = 0;
  state.rfBuffer = [];
  dom.recordingBar.classList.remove("visible");
  if (document.getElementById("cnnSentenceText")) {
    document.getElementById("cnnSentenceText").textContent = "";
  }

  dom.tabRF.classList.toggle("active", model === "rf");
  dom.tabCNN.classList.toggle("active", model === "cnn1d");
  
  dom.sentenceBox.style.display = model === "cnn1d" ? "block" : "none";

  dom.predLetter.textContent = "?";
  dom.predLabel.textContent = "Bekleniyor...";
  dom.confFill.style.width = "0%";
  dom.confPct.textContent = "%0";
  dom.top5List.innerHTML = "";
}


// ═══════ EVENTS ═══════
function init() {
  dom.btnCamera.addEventListener("click", startCamera);
  dom.btnScreen.addEventListener("click", startScreenShare);
  dom.btnStop.addEventListener("click", stopCamera);
  
  dom.btnSound.addEventListener("click", () => {
    state.soundEnabled = !state.soundEnabled;
    dom.soundIcon.textContent = state.soundEnabled ? "🔊" : "🔇";
    dom.btnSound.title = state.soundEnabled ? "Sesi Kapat" : "Sesli Çıktı";
  });

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
