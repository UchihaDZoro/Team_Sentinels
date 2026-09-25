/**
 * IBVAP — Intelligent Border Video Analytics Platform
 * Frontend Application Logic
 */

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
const isLocalDev = window.location.protocol === 'file:' || (window.location.port !== '8000' && window.location.port !== '');
const API_BASE = isLocalDev ? 'http://localhost:8000' : window.location.origin;
const WS_URL = isLocalDev ? 'ws://localhost:8000/ws' : `ws://${window.location.host}/ws`;

let cameras = [];
let alerts = [];
let ws = null;
let wsReconnectTimer = null;
let fenceEditorState = {
  cameraId: null,
  points: [],
  zones: [],
};

// ═══════════════════════════════════════════════════════════════
// Initialisation
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  connectWebSocket();
  loadCameras();
  loadAlerts();
  loadDemoVideos();

  // Event listeners
  document.getElementById('add-camera-btn').addEventListener('click', () => openModal('add-camera-modal'));
  document.getElementById('add-camera-submit').addEventListener('click', addCamera);
  document.getElementById('clear-alerts-btn').addEventListener('click', clearAlerts);
  document.getElementById('save-fence-btn').addEventListener('click', saveFence);

  // Quick video launcher listeners
  const playBtn = document.getElementById('play-filepath-btn');
  if (playBtn) playBtn.addEventListener('click', playDirectFilepath);

  const browseBtn = document.getElementById('browse-file-btn');
  const fileInput = document.getElementById('video-file-input');
  if (browseBtn && fileInput) {
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleVideoUpload);
  }

  const pathInput = document.getElementById('quick-filepath-input');
  if (pathInput) {
    pathInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') playDirectFilepath();
    });
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// Clock
// ═══════════════════════════════════════════════════════════════
function startClock() {
  const el = document.getElementById('current-time');
  function tick() {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-IN', { hour12: false });
  }
  tick();
  setInterval(tick, 1000);
}

// ═══════════════════════════════════════════════════════════════
// WebSocket — Real-time alerts & stats
// ═══════════════════════════════════════════════════════════════
function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('[WS] Connected');
    document.getElementById('system-dot').classList.add('active');
    document.getElementById('system-status').textContent = 'ONLINE';
    if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'alert') handleNewAlert(msg.data);
      if (msg.type === 'stats') handleStats(msg.data);
    } catch (e) { console.error('[WS] Parse error:', e); }
  };

  ws.onclose = () => {
    console.log('[WS] Disconnected — reconnecting in 3s...');
    document.getElementById('system-dot').classList.remove('active');
    document.getElementById('system-status').textContent = 'RECONNECTING...';
    wsReconnectTimer = setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => ws.close();

  // Heartbeat
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 30000);
}

// ═══════════════════════════════════════════════════════════════
// Handle real-time stats update
// ═══════════════════════════════════════════════════════════════
function handleStats(stats) {
  document.getElementById('total-persons').textContent = stats.total_persons || 0;
  document.getElementById('total-vehicles').textContent = stats.total_vehicles || 0;
  document.getElementById('total-intrusions').textContent = stats.total_intrusions || 0;

  // Average FPS across cameras
  const camStats = stats.cameras || {};
  const fpsValues = Object.values(camStats).map(c => c.fps || 0).filter(f => f > 0);
  const avgFps = fpsValues.length ? (fpsValues.reduce((a, b) => a + b, 0) / fpsValues.length) : 0;
  document.getElementById('avg-fps').textContent = Math.round(avgFps);

  // Update per-camera stats in cards
  for (const [camId, camStat] of Object.entries(camStats)) {
    const card = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
    if (!card) continue;

    const fpsEl = card.querySelector('.stat-fps');
    const persEl = card.querySelector('.stat-persons');
    const vehEl = card.querySelector('.stat-vehicles');
    if (fpsEl) fpsEl.textContent = Math.round(camStat.fps || 0);
    if (persEl) persEl.textContent = camStat.persons || 0;
    if (vehEl) vehEl.textContent = camStat.vehicles || 0;

    // Intrusion visual indicator
    if (camStat.intrusions > 0) {
      card.classList.add('intrusion');
    } else {
      card.classList.remove('intrusion');
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Handle new alert (real-time)
// ═══════════════════════════════════════════════════════════════
function handleNewAlert(alert) {
  alerts.unshift(alert);
  renderAlerts();
  updateAlertCount();
  showToast(alert);
  playAlertSound(alert.severity);
}

function playAlertSound(severity) {
  // Generate a beep using Web Audio API
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = severity === 'critical' ? 880 : 660;
    gain.gain.value = 0.15;
    osc.start();
    osc.stop(ctx.currentTime + 0.2);
  } catch (e) { /* Audio not available */ }
}

function showToast(alert) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <span class="toast-icon">🚨</span>
    <span class="toast-message">${alert.message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('exiting');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ═══════════════════════════════════════════════════════════════
// API — Load cameras
// ═══════════════════════════════════════════════════════════════
async function loadCameras() {
  try {
    const res = await fetch(`${API_BASE}/api/cameras`);
    const data = await res.json();
    cameras = data.cameras || [];
    document.getElementById('camera-count').textContent = cameras.length;
    renderCameraGrid();
  } catch (e) {
    console.error('Failed to load cameras:', e);
  }
}

async function loadAlerts() {
  try {
    const res = await fetch(`${API_BASE}/api/alerts?limit=100`);
    const data = await res.json();
    alerts = data.alerts || [];
    renderAlerts();
    updateAlertCount();
  } catch (e) {
    console.error('Failed to load alerts:', e);
  }
}

async function loadDemoVideos() {
  try {
    const res = await fetch(`${API_BASE}/api/demo-videos`);
    const data = await res.json();
    const videos = data.videos || [];
    if (videos.length > 0) {
      const section = document.getElementById('demo-videos-section');
      const list = document.getElementById('demo-video-list');
      section.style.display = 'block';
      list.innerHTML = videos.map(v => `
        <button class="btn btn-secondary btn-sm" onclick="quickAddDemo('${v.path.replace(/\\/g, '\\\\')}', '${v.name}')">
          🎬 ${v.name} (${v.size_mb} MB)
        </button>
      `).join('');
    }
  } catch (e) { /* Demo videos not available */ }
}

function quickAddDemo(path, name) {
  document.getElementById('cam-name').value = name;
  document.getElementById('cam-source').value = path;
}

// ═══════════════════════════════════════════════════════════════
// Direct Video Play & Preset Handlers
// ═══════════════════════════════════════════════════════════════
async function playDirectFilepath() {
  const input = document.getElementById('quick-filepath-input');
  const source = input ? input.value.trim() : '';
  if (!source) {
    alert('Please enter or paste a valid YouTube link, live stream URL, or video filepath (e.g. https://www.youtube.com/watch?v=sKcmQQqzQcM)');
    return;
  }

  const btn = document.getElementById('play-filepath-btn');
  const origText = btn ? btn.textContent : '';
  if (btn) {
    btn.textContent = 'CONNECTING...';
    btn.disabled = true;
  }

  try {
    let name = '';
    let location = 'Local Feed';
    if (source.includes('youtube.com') || source.includes('youtu.be')) {
      name = 'YouTube Live CCTV';
      location = 'YouTube Live Stream';
    } else if (source.startsWith('rtsp://')) {
      name = 'RTSP Camera';
      location = 'Network RTSP';
    } else if (source.startsWith('http://') || source.startsWith('https://')) {
      name = 'Online Stream';
      location = 'Online Stream';
    } else {
      const filename = source.split(/[\\/]/).pop() || 'Video Feed';
      name = filename.replace(/\.[^/.]+$/, '').replace(/_/g, ' ').toUpperCase();
    }

    const res = await fetch(`${API_BASE}/api/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, source, location, auto_start: true }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to start video source: ' + source);
      return;
    }

    showToast({
      severity: 'low',
      message: `Surveillance Active: ${name} [Humans: GREEN box | Vehicles: YELLOW box]`
    });
    await loadCameras();
  } catch (e) {
    alert('Connection error: ' + e.message);
  } finally {
    if (btn) {
      btn.textContent = origText;
      btn.disabled = false;
    }
  }
}

async function playYouTubePreset(url, label) {
  document.querySelectorAll('.chip-btn').forEach(btn => btn.classList.remove('active'));
  if (window.event && window.event.target) {
    window.event.target.classList.add('active');
  }

  const input = document.getElementById('quick-filepath-input');
  if (input) input.value = url;

  const btn = document.getElementById('play-filepath-btn');
  if (btn) {
    btn.textContent = 'RESOLVING STREAM...';
    btn.disabled = true;
  }

  try {
    const res = await fetch(`${API_BASE}/api/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: label,
        source: url,
        location: 'YouTube 24/7 Live Cam',
        auto_start: true,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to start YouTube live stream');
      return;
    }

    showToast({
      severity: 'low',
      message: `Live YouTube Feed Active: ${label} [Green: Humans | Yellow: Vehicles]`
    });
    await loadCameras();
  } catch (e) {
    alert('Error connecting live stream: ' + e.message);
  } finally {
    if (btn) {
      btn.textContent = '▶ PLAY & ANALYZE';
      btn.disabled = false;
    }
  }
}

async function playDemoPreset(filename, label) {
  // Update active chip UI
  document.querySelectorAll('.chip-btn').forEach(btn => btn.classList.remove('active'));
  if (window.event && window.event.target) {
    window.event.target.classList.add('active');
  }

  const defaultPath = `D:\\CCTron\\backend\\demo_videos\\${filename}`;
  const input = document.getElementById('quick-filepath-input');
  if (input) input.value = defaultPath;

  const btn = document.getElementById('play-filepath-btn');
  if (btn) {
    btn.textContent = 'STARTING...';
    btn.disabled = true;
  }

  try {
    const res = await fetch(`${API_BASE}/api/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: label,
        source: defaultPath,
        location: 'Border Surveillance Checkpost',
        auto_start: true,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to start demo preset');
      return;
    }

    showToast({
      severity: 'low',
      message: `Demo Active: ${label} [Green: Humans | Yellow: Vehicles]`
    });
    await loadCameras();
  } catch (e) {
    alert('Error starting demo: ' + e.message);
  } finally {
    if (btn) {
      btn.textContent = '▶ PLAY & ANALYZE';
      btn.disabled = false;
    }
  }
}

async function handleVideoUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const browseBtn = document.getElementById('browse-file-btn');
  const origText = browseBtn ? browseBtn.textContent : '';
  if (browseBtn) {
    browseBtn.textContent = '⏳ UPLOADING...';
    browseBtn.disabled = true;
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/api/upload-video`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to upload video');
      return;
    }

    if (data.path) {
      const input = document.getElementById('quick-filepath-input');
      if (input) input.value = data.path;
    }

    showToast({
      severity: 'low',
      message: `Uploaded & Playing: ${file.name} [Green: Humans | Yellow: Vehicles]`
    });
    await loadCameras();
  } catch (e) {
    alert('Upload error: ' + e.message);
  } finally {
    if (browseBtn) {
      browseBtn.textContent = origText;
      browseBtn.disabled = false;
    }
    event.target.value = '';
  }
}


// ═══════════════════════════════════════════════════════════════
// API — Add camera
// ═══════════════════════════════════════════════════════════════
async function addCamera() {
  const name = document.getElementById('cam-name').value.trim();
  const source = document.getElementById('cam-source').value.trim();
  const location = document.getElementById('cam-location').value.trim();

  if (!name || !source) {
    alert('Camera name and source are required.');
    return;
  }

  const btn = document.getElementById('add-camera-submit');
  btn.textContent = 'CONNECTING...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, source, location, auto_start: true }),
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.error || 'Failed to add camera');
      return;
    }

    closeModal('add-camera-modal');
    document.getElementById('cam-name').value = '';
    document.getElementById('cam-source').value = '';
    document.getElementById('cam-location').value = '';
    await loadCameras();
  } catch (e) {
    alert('Connection error: ' + e.message);
  } finally {
    btn.textContent = 'ADD & START';
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════
// API — Camera actions
// ═══════════════════════════════════════════════════════════════
async function startCamera(camId) {
  try {
    const res = await fetch(`${API_BASE}/api/cameras/${camId}/start`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || data.error || 'Failed to start video source. Please check that the file path is a valid video file.');
    }
  } catch (e) {
    alert('Connection error: ' + e.message);
  }
  await loadCameras();
}

async function stopCamera(camId) {
  try {
    await fetch(`${API_BASE}/api/cameras/${camId}/stop`, { method: 'POST' });
  } catch (e) {
    console.error('Stop error:', e);
  }
  await loadCameras();
}

async function deleteCamera(camId) {
  if (!confirm('Delete this camera?')) return;
  await fetch(`${API_BASE}/api/cameras/${camId}`, { method: 'DELETE' });
  await loadCameras();
}

async function clearAllCameras() {
  if (!confirm('Clear all cameras from the dashboard?')) return;
  try {
    await fetch(`${API_BASE}/api/cameras`, { method: 'DELETE' });
    showToast({ severity: 'low', message: 'All cameras cleared from surveillance grid' });
    await loadCameras();
  } catch (e) {
    alert('Failed to clear cameras: ' + e.message);
  }
}

function handleStreamError(img, camId) {
  console.warn(`[Stream] Reconnecting camera ${camId}...`);
  if (img._retryCount && img._retryCount > 15) return;
  img._retryCount = (img._retryCount || 0) + 1;
  setTimeout(() => {
    img.src = `${API_BASE}/api/cameras/${camId}/stream?t=${Date.now()}`;
  }, 1500);
}

async function toggleNightMode(camId, enabled) {
  await fetch(`${API_BASE}/api/cameras/${camId}/night-mode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
}

// ═══════════════════════════════════════════════════════════════
// Render camera grid
// ═══════════════════════════════════════════════════════════════
function renderCameraGrid() {
  const grid = document.getElementById('camera-grid');

  // Set grid class based on camera count
  const count = cameras.length;
  grid.className = count <= 1 ? 'grid-1' : count <= 2 ? 'grid-2' : count <= 4 ? 'grid-4' : 'grid-6';

  // Keep existing cards, update or add new ones
  const existingCards = new Map();
  grid.querySelectorAll('.camera-card').forEach(card => {
    existingCards.set(card.dataset.camId, card);
  });

  // Remove cards for deleted cameras
  for (const [camId, card] of existingCards) {
    if (!cameras.find(c => c.id === camId)) card.remove();
  }

  cameras.forEach(cam => {
    let card = existingCards.get(cam.id);
    if (!card) {
      card = createCameraCard(cam);
      grid.appendChild(card);
    }
    updateCameraCard(card, cam);
  });

  // If no cameras, show empty state
  if (cameras.length === 0) {
    if (!grid.querySelector('.empty-state')) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1; grid-row: 1 / -1;">
          <span class="icon">📹</span>
          <p>NO CAMERAS CONFIGURED</p>
          <p>Click "+ ADD CAMERA" to begin surveillance</p>
        </div>
      `;
    }
  }
}

function createCameraCard(cam) {
  const card = document.createElement('div');
  card.className = 'camera-card';
  card.dataset.camId = cam.id;

  card.innerHTML = `
    <div class="camera-header">
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="camera-name">${escapeHtml(cam.name)}</span>
        <span style="font-size:11px; font-family:var(--font-mono); color:#8899b0;">
          [<span style="color:#00ff00; font-weight:700;">● Human</span> | <span style="color:#ffff00; font-weight:700;">● Vehicle</span>]
        </span>
      </div>
      <div class="camera-status">
        <span class="camera-badge ${cam.status === 'active' ? 'badge-live' : 'badge-offline'}">
          ${cam.status === 'active' ? '● LIVE' : '○ OFF'}
        </span>
      </div>
    </div>
    <div class="camera-feed">
      ${cam.status === 'active'
        ? `<img src="${API_BASE}/api/cameras/${cam.id}/stream" alt="${escapeHtml(cam.name)}" onerror="handleStreamError(this, '${cam.id}')">`
        : `<div class="placeholder"><span class="icon">📹</span>Camera offline</div>`
      }
    </div>
    <div class="camera-actions">
      <div class="cam-stat" title="Humans Detected (Green Box)">
        👥 <span class="val stat-persons" style="color:#00ff00; font-weight:700;">0</span> <span style="color:#00ff00; font-size:10px;">Humans</span>
      </div>
      <div class="cam-stat" title="Vehicles Detected (Yellow Box)">
        🚗 <span class="val stat-vehicles" style="color:#ffff00; font-weight:700;">0</span> <span style="color:#ffff00; font-size:10px;">Vehicles</span>
      </div>
      <div class="cam-stat" title="Stream Framerate">
        ⚡ <span class="val stat-fps">0</span> fps
      </div>
      <div class="spacer"></div>
      <button class="cam-btn" onclick="openFenceEditor('${cam.id}')" title="Virtual Fence">🔲</button>
      <button class="cam-btn" onclick="toggleNightMode('${cam.id}', true)" title="Night Mode">🌙</button>
      ${cam.status === 'active'
        ? `<button class="cam-btn btn-start-stop" onclick="stopCamera('${cam.id}')" title="Stop">⏹</button>`
        : `<button class="cam-btn btn-start-stop" onclick="startCamera('${cam.id}')" title="Start">▶</button>`
      }
      <button class="cam-btn danger" onclick="deleteCamera('${cam.id}')" title="Delete">✕</button>
    </div>
  `;

  return card;
}

function handleStreamError(img, camId) {
  console.warn(`[Stream] Stream reconnecting for ${camId}...`);
  setTimeout(() => {
    if (img && img.parentElement) {
      img.src = `${API_BASE}/api/cameras/${camId}/stream?t=${Date.now()}`;
    }
  }, 1500);
}

function updateCameraCard(card, cam) {
  const badge = card.querySelector('.camera-badge');
  if (badge) {
    badge.className = `camera-badge ${cam.status === 'active' ? 'badge-live' : 'badge-offline'}`;
    badge.textContent = cam.status === 'active' ? '● LIVE' : '○ OFF';
  }

  const feed = card.querySelector('.camera-feed');
  if (feed) {
    if (cam.status === 'active') {
      const existingImg = feed.querySelector('img');
      if (!existingImg) {
        feed.innerHTML = `<img src="${API_BASE}/api/cameras/${cam.id}/stream" alt="${escapeHtml(cam.name)}" onerror="handleStreamError(this, '${cam.id}')">`;
      }
    } else {
      feed.innerHTML = `<div class="placeholder"><span class="icon">📹</span>Camera offline</div>`;
    }
  }

  const startStopBtn = card.querySelector('.btn-start-stop');
  if (startStopBtn) {
    if (cam.status === 'active') {
      startStopBtn.title = 'Stop';
      startStopBtn.textContent = '⏹';
      startStopBtn.setAttribute('onclick', `stopCamera('${cam.id}')`);
    } else {
      startStopBtn.title = 'Start';
      startStopBtn.textContent = '▶';
      startStopBtn.setAttribute('onclick', `startCamera('${cam.id}')`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Render alerts
// ═══════════════════════════════════════════════════════════════
function renderAlerts() {
  const list = document.getElementById('alert-list');
  const empty = document.getElementById('alerts-empty');

  if (alerts.length === 0) {
    if (!empty) {
      list.innerHTML = `
        <div class="empty-state" id="alerts-empty">
          <span class="icon">📡</span>
          <p>NO ALERTS</p>
          <p>System monitoring active</p>
        </div>
      `;
    }
    return;
  }

  list.innerHTML = alerts.slice(0, 100).map((alert, i) => {
    const severityClass = alert.severity || 'medium';
    const typeLabel = (alert.alert_type || '').replace(/_/g, ' ').toUpperCase();
    const time = alert.created_at ? new Date(alert.created_at).toLocaleTimeString('en-IN', { hour12: false }) : '';
    const camName = cameras.find(c => c.id === alert.camera_id)?.name || alert.camera_id;

    return `
      <div class="alert-card ${severityClass} ${i === 0 ? 'new' : ''}">
        <div class="alert-type">
          <span class="severity-dot"></span>
          ${typeLabel}
        </div>
        <div class="alert-message">${escapeHtml(alert.message)}</div>
        <div class="alert-meta">
          <span class="alert-camera">📹 ${escapeHtml(camName)}</span>
          <span>${time}</span>
        </div>
      </div>
    `;
  }).join('');
}

function updateAlertCount() {
  const count = alerts.length;
  document.getElementById('alert-count').textContent = count;
  document.getElementById('sidebar-alert-count').textContent = count;
}

async function clearAlerts() {
  await fetch(`${API_BASE}/api/alerts`, { method: 'DELETE' });
  alerts = [];
  renderAlerts();
  updateAlertCount();
}

// ═══════════════════════════════════════════════════════════════
// Virtual Fence Editor
// ═══════════════════════════════════════════════════════════════
function openFenceEditor(camId) {
  fenceEditorState.cameraId = camId;
  fenceEditorState.points = [];
  fenceEditorState.zones = [];

  // Set a snapshot from the stream as background
  const img = document.getElementById('fence-frame');
  img.src = `/api/cameras/${camId}/stream`;

  // Wait for image to load then setup canvas
  img.onload = () => setupFenceCanvas();
  setTimeout(setupFenceCanvas, 500); // fallback

  openModal('fence-modal');
}

function setupFenceCanvas() {
  const wrapper = document.getElementById('fence-canvas-wrapper');
  const canvas = document.getElementById('fence-canvas');
  const img = document.getElementById('fence-frame');

  canvas.width = wrapper.offsetWidth;
  canvas.height = wrapper.offsetHeight;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Mouse handlers
  canvas.onmousedown = (e) => {
    if (e.button === 2) { // right-click closes polygon
      e.preventDefault();
      closePolygon();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    fenceEditorState.points.push([Math.round(x), Math.round(y)]);
    drawFenceOverlay();
  };

  canvas.oncontextmenu = (e) => e.preventDefault();

  // Keyboard
  canvas.tabIndex = 0;
  canvas.focus();
  canvas.onkeydown = (e) => {
    if (e.key === 'Enter') closePolygon();
    if (e.key === 'Escape') {
      fenceEditorState.points = [];
      drawFenceOverlay();
    }
  };
}

function drawFenceOverlay() {
  const canvas = document.getElementById('fence-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw completed zones
  for (const zone of fenceEditorState.zones) {
    drawPolygon(ctx, zone.points, 'rgba(255, 45, 85, 0.2)', 'rgba(255, 45, 85, 0.8)');
  }

  // Draw current polygon being drawn
  const pts = fenceEditorState.points;
  if (pts.length > 0) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw points
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#00ff88';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

function drawPolygon(ctx, points, fill, stroke) {
  if (points.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function closePolygon() {
  if (fenceEditorState.points.length < 3) return;
  const name = document.getElementById('fence-zone-name').value.trim() || 'Restricted Zone';

  // Scale points from canvas coordinates to actual frame coordinates (960x540)
  const canvas = document.getElementById('fence-canvas');
  const scaleX = 960 / canvas.width;
  const scaleY = 540 / canvas.height;
  const scaledPoints = fenceEditorState.points.map(([x, y]) => [
    Math.round(x * scaleX),
    Math.round(y * scaleY),
  ]);

  fenceEditorState.zones.push({
    id: `zone_${Date.now()}`,
    name: name,
    points: scaledPoints,
    displayPoints: [...fenceEditorState.points], // keep canvas-space points for drawing
  });
  fenceEditorState.points = [];
  drawFenceOverlay();
}

function clearFenceDrawing() {
  fenceEditorState.points = [];
  fenceEditorState.zones = [];
  drawFenceOverlay();
}

async function saveFence() {
  // Close any in-progress polygon first
  if (fenceEditorState.points.length >= 3) closePolygon();

  const camId = fenceEditorState.cameraId;
  const zones = fenceEditorState.zones.map(z => ({
    id: z.id,
    name: z.name,
    points: z.points, // already scaled to frame coords
  }));

  try {
    await fetch(`${API_BASE}/api/cameras/${camId}/fence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones }),
    });
    closeFenceEditor();
    showToast({ message: `Virtual fence set on camera ${camId}`, severity: 'low' });
  } catch (e) {
    alert('Failed to save fence: ' + e.message);
  }
}

async function removeFence() {
  const camId = fenceEditorState.cameraId;
  await fetch(`${API_BASE}/api/cameras/${camId}/fence`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ zones: [] }),
  });
  closeFenceEditor();
}

function closeFenceEditor() {
  fenceEditorState.points = [];
  fenceEditorState.zones = [];
  // Stop the MJPEG stream in the editor
  document.getElementById('fence-frame').src = '';
  closeModal('fence-modal');
}

// ═══════════════════════════════════════════════════════════════
// Modal helpers
// ═══════════════════════════════════════════════════════════════
function openModal(id) {
  document.getElementById(id).classList.add('active');
}

function closeModal(id) {
  document.getElementById(id).classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════
// Utility
// ═══════════════════════════════════════════════════════════════
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
