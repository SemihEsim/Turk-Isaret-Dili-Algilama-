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
  hands: null,
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


// ═══════ MEDIAPIPE TRACKERS ═══════
function initHands() {
  if (state.hands) return;
  if (typeof Hands === "undefined") {
    console.error("[MediaPipe] Hands yüklenemedi!");
    return;
  }

  state.hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
  });

  state.hands.setOptions({
    selfieMode: !state.isScreen,
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  state.hands.onResults((handsResults) => {
    // Hands çıktısını Holistic formatına çevir (kod uyumluluğu için)
    const results = { leftHandLandmarks: null, rightHandLandmarks: null, poseLandmarks: null };
    if (handsResults.multiHandLandmarks && handsResults.multiHandedness) {
      for (let i = 0; i < handsResults.multiHandLandmarks.length; i++) {
        const lm = handsResults.multiHandLandmarks[i];
        const label = handsResults.multiHandedness[i].label; // "Left" or "Right"
        // SelfieMode nedeniyle fiziksel sağ el "Left" olarak etiketlenir
        if (label === "Left") results.leftHandLandmarks = lm;
        else results.rightHandLandmarks = lm;
      }
    }
    state.results = results;
    drawLandmarks(results);
  });

  console.log("[MediaPipe] Hands başlatıldı");
}

function initHolistic() {
  if (state.holistic) return;
  if (typeof Holistic === "undefined") {
    console.error("[MediaPipe] Holistic yüklenemedi!");
    return;
  }

  state.holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
  });

  state.holistic.setOptions({
    selfieMode: !state.isScreen, // Ekran paylaşımında aynalamayı kapat
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });

  state.holistic.onResults((results) => {
    state.results = results;
    drawLandmarks(results);
  });

  console.log("[MediaPipe] Holistic başlatıldı");
}

function drawHandStyle(ctx, landmarks) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  
  // Lokal Python kodu ile aynı bağlantı renkleri
  const conns = [
    [0, 1, "rgb(128,128,128)"], [1, 2, "rgb(255,229,180)"], [2, 3, "rgb(255,229,180)"], [3, 4, "rgb(255,229,180)"],
    [0, 5, "rgb(128,128,128)"], [5, 6, "rgb(128,64,128)"], [6, 7, "rgb(128,64,128)"], [7, 8, "rgb(128,64,128)"],
    [5, 9, "rgb(128,128,128)"], [9, 10, "rgb(255,204,0)"], [10, 11, "rgb(255,204,0)"], [11, 12, "rgb(255,204,0)"],
    [9, 13, "rgb(128,128,128)"], [13, 14, "rgb(48,255,48)"], [14, 15, "rgb(48,255,48)"], [15, 16, "rgb(48,255,48)"],
    [13, 17, "rgb(128,128,128)"], [0, 17, "rgb(128,128,128)"], [17, 18, "rgb(21,101,192)"], [18, 19, "rgb(21,101,192)"], [19, 20, "rgb(21,101,192)"]
  ];

  ctx.lineWidth = 2;
  for (const [i, j, color] of conns) {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    if (p1 && p2) {
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }

  // Lokal Python kodu ile aynı nokta renkleri
  const colors = [
    "rgb(255,0,0)",
    "rgb(255,229,180)", "rgb(255,229,180)", "rgb(255,229,180)", "rgb(255,229,180)",
    "rgb(128,64,128)", "rgb(128,64,128)", "rgb(128,64,128)", "rgb(128,64,128)",
    "rgb(255,204,0)", "rgb(255,204,0)", "rgb(255,204,0)", "rgb(255,204,0)",
    "rgb(48,255,48)", "rgb(48,255,48)", "rgb(48,255,48)", "rgb(48,255,48)",
    "rgb(21,101,192)", "rgb(21,101,192)", "rgb(21,101,192)", "rgb(21,101,192)"
  ];
  
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i];
    if (p) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, 2 * Math.PI);
      ctx.fillStyle = colors[i];
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

function drawLandmarks(results) {
  const ctx = dom.canvas.getContext("2d");
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);

  // Eller (Yerel Python renkleriyle çizim)
  if (results.leftHandLandmarks) {
    drawHandStyle(ctx, results.leftHandLandmarks);
  }
  if (results.rightHandLandmarks) {
    drawHandStyle(ctx, results.rightHandLandmarks);
  }

  // Pose (Daha belirgin vücut çizgileri, eğer typeof drawConnectors varsa)
  if (results.poseLandmarks && typeof drawConnectors === "function" && typeof POSE_CONNECTIONS !== "undefined") {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "rgba(255, 255, 255, 0.3)", lineWidth: 2 });
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
  
  if (state.selectedModel === "rf") {
    if (!state.hands) initHands();
    else state.hands.setOptions({ selfieMode: !state.isScreen });
  } else {
    if (!state.holistic) initHolistic();
    else state.holistic.setOptions({ selfieMode: !state.isScreen });
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
    // Send to Active MediaPipe Tracker
    const tracker = state.selectedModel === "rf" ? state.hands : state.holistic;
    if (tracker && !mpBusy && dom.video.readyState >= 2) {
      mpBusy = true;
      tracker.send({ image: dom.video }).then(() => { mpBusy = false; }).catch(() => { mpBusy = false; });
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
function handleRF(now) {
  if (now - state.lastSendTime < state.sendInterval) return;
  const results = state.results;
  const hasHand = results.leftHandLandmarks || results.rightHandLandmarks;
  
  if (!hasHand) {
    dom.predLetter.textContent = "?";
    dom.predLabel.textContent = "El Bekleniyor...";
    dom.confFill.style.width = "0%";
    dom.confPct.textContent = "%0";
    state.rfBuffer = []; // El kaybolunca buffer'ı sıfırla
    return;
  }

  state.lastSendTime = now;
  const { landmarks, hand_count } = extractRFLandmarks(results);
  if (hand_count === 0) return;

  // Use an async IIFE to not block the synchronous frame loop execution path
  (async () => {
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
  })();
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

  // Update active tracker if streaming
  if (state.streaming) {
    if (model === "rf" && !state.hands) initHands();
    if (model === "cnn1d" && !state.holistic) initHolistic();
  }
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
