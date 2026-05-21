const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const engineEl = document.getElementById("engine");
const confidenceEl = document.getElementById("confidence");
const latencyEl = document.getElementById("latency");
const tokensEl = document.getElementById("tokens");
const textEl = document.getElementById("text");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const speakBtn = document.getElementById("speakBtn");
const autoSpeak = document.getElementById("autoSpeak");

const ctx = canvas.getContext("2d");

let stream = null;
let sessionId = null;
let captureTimer = null;
let lastSpoken = "";
let lastText = "";
let lastSpokenAt = 0;

const TARGET_FPS = 4;
const FRAME_W = 320;
const FRAME_H = 240;

function setStatus(msg) {
  statusEl.textContent = `Status: ${msg}`;
}

function speak(text) {
  if (!text) return;
  const now = Date.now();
  if (now - lastSpokenAt < 1200) return;
  if (text === lastSpoken) return;
  lastSpoken = text;
  lastSpokenAt = now;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  speechSynthesis.speak(u);
}

async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function startSession() {
  const data = await api("/api/session/start", "POST", {});
  sessionId = data.session_id;
}

async function stopSession() {
  if (!sessionId) return;
  await api("/api/session/stop", "POST", { session_id: sessionId });
  sessionId = null;
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: FRAME_W, height: FRAME_H },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

function getFrameDataUrl() {
  canvas.width = FRAME_W;
  canvas.height = FRAME_H;
  ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);
  return canvas.toDataURL("image/jpeg", 0.75);
}

async function captureLoop() {
  if (!sessionId || !stream) return;
  try {
    const image_b64 = getFrameDataUrl();
    const payload = { session_id: sessionId, image_b64, timestamp_ms: Date.now() };
    const pred = await api("/api/frame", "POST", payload);
    engineEl.textContent = `Engine: ${pred.engine}`;
    confidenceEl.textContent = `Confidence: ${pred.confidence.toFixed(2)}`;
    latencyEl.textContent = `Latency: ${pred.latency_ms} ms`;
    tokensEl.textContent = `Tokens: ${(pred.raw_tokens || []).join(" ") || "-"}`;
    if (pred.text) {
      textEl.textContent = pred.text;
      lastText = pred.text;
      if (autoSpeak.checked && pred.updated) {
        speak(pred.text);
      }
    }
  } catch (err) {
    setStatus(`error (${err.message})`);
  }
}

async function start() {
  setStatus("starting");
  const health = await api("/health");
  engineEl.textContent = `Engine: ${health.engine} (${health.ready ? "ready" : "not ready"})`;
  if (!health.ready) {
    setStatus(`engine not ready: ${health.status_message}`);
  }
  await startCamera();
  await startSession();
  captureTimer = setInterval(captureLoop, Math.floor(1000 / TARGET_FPS));
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("running");
}

async function stop() {
  setStatus("stopping");
  clearInterval(captureTimer);
  captureTimer = null;
  stopCamera();
  await stopSession();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("idle");
}

startBtn.addEventListener("click", () => start().catch((e) => setStatus(e.message)));
stopBtn.addEventListener("click", () => stop().catch((e) => setStatus(e.message)));
speakBtn.addEventListener("click", () => speak(lastText));

window.addEventListener("beforeunload", () => {
  stop().catch(() => null);
});

setStatus("idle");
