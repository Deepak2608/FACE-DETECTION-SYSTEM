/* main.js – MaskGuard frontend */

// ── Sample images (public domain faces) ──────────────────────────────────────
const SAMPLES = {
  mask:   "https://images.unsplash.com/photo-1584634731339-252c581abfc5?w=400&q=80",
  nomask: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80",
};

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab-section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".nav-pill").forEach(p => p.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.querySelectorAll(".nav-pill").forEach(p => {
    if (p.textContent.toLowerCase().includes(name === "image" ? "image" : name === "webcam" ? "webcam" : "metric"))
      p.classList.add("active");
  });
  if (name === "metrics") loadMetrics();
}

// ── Image upload / preview ────────────────────────────────────────────────────
let currentFile = null;

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = ev => showPreview(ev.target.result);
  reader.readAsDataURL(file);
}

function showPreview(src) {
  document.getElementById("dropZone").style.display    = "none";
  document.getElementById("previewWrap").style.display = "block";
  document.getElementById("previewImg").src = src;
  resetResult();
}

function clearImage() {
  document.getElementById("dropZone").style.display    = "block";
  document.getElementById("previewWrap").style.display = "none";
  document.getElementById("fileInput").value = "";
  currentFile = null;
  resetResult();
}

function resetResult() {
  document.getElementById("resultContent").classList.add("hidden");
  document.getElementById("resultPlaceholder").classList.remove("hidden");
}

function loadSample(type) {
  const url = SAMPLES[type];
  fetch(url)
    .then(r => r.blob())
    .then(blob => {
      currentFile = new File([blob], type + ".jpg", { type: "image/jpeg" });
      const reader = new FileReader();
      reader.onload = ev => showPreview(ev.target.result);
      reader.readAsDataURL(currentFile);
    })
    .catch(() => {
      alert("Could not load sample image. Please upload your own image.");
    });
}

// Drag and drop
const dropZone = document.getElementById("dropZone");
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.style.borderColor = "var(--accent)"; });
dropZone.addEventListener("dragleave", () => { dropZone.style.borderColor = ""; });
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.style.borderColor = "";
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = ev => showPreview(ev.target.result);
    reader.readAsDataURL(file);
  }
});

// ── Image detection ───────────────────────────────────────────────────────────
async function detectImage() {
  if (!currentFile) { alert("Please upload an image first."); return; }

  const btn     = document.getElementById("detectBtn");
  const btnTxt  = document.getElementById("detectTxt");
  const spinner = document.getElementById("detectSpinner");

  btn.disabled = true;
  btnTxt.textContent = "Detecting…";
  spinner.classList.remove("hidden");

  try {
    const formData = new FormData();
    formData.append("image", currentFile);

    const res  = await fetch("/api/predict/image", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) throw new Error(data.error);
    renderResult(data);
  } catch (err) {
    alert("Detection error: " + err.message + "\n\nMake sure the server is running.");
  } finally {
    btn.disabled = false;
    btnTxt.textContent = "Detect masks";
    spinner.classList.add("hidden");
  }
}

function renderResult(data) {
  document.getElementById("resultPlaceholder").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");

  document.getElementById("resultImg").src = data.image;

  const statsHtml = `
    <div class="rs-card"><div class="rs-num white">${data.total_faces}</div><div class="rs-label">Faces found</div></div>
    <div class="rs-card"><div class="rs-num green">${data.with_mask}</div><div class="rs-label">With mask</div></div>
    <div class="rs-card"><div class="rs-num red">${data.without_mask}</div><div class="rs-label">No mask</div></div>
  `;
  document.getElementById("resultStats").innerHTML = statsHtml;

  const banner = document.getElementById("verdictBanner");
  if (data.total_faces === 0) {
    banner.className   = "verdict-banner noface";
    banner.textContent = "No faces detected in image";
  } else if (data.safe) {
    banner.className   = "verdict-banner safe";
    banner.textContent = `✓ All ${data.total_faces} person(s) wearing mask — SAFE`;
  } else {
    banner.className   = "verdict-banner unsafe";
    banner.textContent = `✗ ${data.without_mask} person(s) NOT wearing mask — UNSAFE`;
  }
}

// ── Webcam ────────────────────────────────────────────────────────────────────
let camRunning = false;

async function startCam() {
  try {
    await fetch("/api/video/start", { method: "POST" });
    const feed = document.getElementById("videoFeed");
    feed.src   = "/api/video_feed";
    feed.style.display = "block";
    document.getElementById("streamIdle").style.display = "none";
    document.getElementById("startCamBtn").classList.add("hidden");
    document.getElementById("stopCamBtn").classList.remove("hidden");
    camRunning = true;
  } catch (e) {
    alert("Could not start camera. Make sure OpenCV can access your webcam.");
  }
}

async function stopCam() {
  await fetch("/api/video/stop", { method: "POST" });
  const feed = document.getElementById("videoFeed");
  feed.src   = "";
  feed.style.display = "none";
  document.getElementById("streamIdle").style.display = "flex";
  document.getElementById("startCamBtn").classList.remove("hidden");
  document.getElementById("stopCamBtn").classList.add("hidden");
  camRunning = false;
}

// ── Metrics ───────────────────────────────────────────────────────────────────
async function loadMetrics() {
  const wrap = document.getElementById("metricsWrap");
  try {
    const res  = await fetch("/api/metrics");
    const data = await res.json();

    if (!data || !data.accuracy) {
      wrap.innerHTML = `<div class="loading-metrics">Run <code>python train_model.py</code> first to see metrics.</div>`;
      return;
    }

    const cards = [
      { val: data.accuracy + "%",   label: "Validation accuracy" },
      { val: data.epochs,            label: "Training epochs" },
      { val: data.train_size,        label: "Train images" },
      { val: data.test_size,         label: "Test images" },
    ];

    let chartHtml = "";
    if (data.val_acc_history && data.val_acc_history.length) {
      const maxAcc = Math.max(...data.val_acc_history, ...data.train_acc_history);
      chartHtml = `
        <div class="chart-wrap">
          <div class="chart-title">Training accuracy per epoch</div>
          <div class="chart-bars">
            ${data.train_acc_history.map((a, i) => `
              <div class="bar-col">
                <div class="bar-fill train" style="height:${Math.round((a/maxAcc)*120)}px" title="Train ${a}%"></div>
                <div class="bar-fill val"   style="height:${Math.round((data.val_acc_history[i]/maxAcc)*120)}px" title="Val ${data.val_acc_history[i]}%"></div>
                <div class="bar-epoch">${i+1}</div>
              </div>
            `).join("")}
          </div>
          <div class="chart-legend">
            <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>Train accuracy</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--green)"></div>Validation accuracy</div>
          </div>
        </div>`;
    }

    wrap.innerHTML = `
      <div class="m-grid">
        ${cards.map(c => `
          <div class="m-card">
            <div class="m-val">${c.val}</div>
            <div class="m-label">${c.label}</div>
          </div>`).join("")}
      </div>
      ${chartHtml}
    `;
  } catch (e) {
    wrap.innerHTML = `<div class="loading-metrics">Could not load metrics.</div>`;
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  window.addEventListener("beforeunload", () => {
    if (camRunning) fetch("/api/video/stop", { method: "POST" });
  });
});
