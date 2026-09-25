/**
 * IBVAP — Intelligent Border Video Analytics Platform
 * Tactical C2 & Military HUD Frontend Application Logic
 * Sashastra Seema Bal (SSB) • Ministry of Home Affairs
 */

// ═══════════════════════════════════════════════════════════════
// Global State & Configuration
// ═══════════════════════════════════════════════════════════════
const isLocalDev = window.location.protocol === 'file:' || (window.location.port !== '8000' && window.location.port !== '');
const API_BASE = isLocalDev ? 'http://localhost:8000' : window.location.origin;
const WS_URL = isLocalDev ? 'ws://localhost:8000/ws' : `ws://${window.location.host}/ws`;

let cameras = [];
let alerts = [];
let watchlistTargets = [];
let hotlistVehicles = [];
let plateScans = [];
let ws = null;
let wsReconnectTimer = null;
let alertFilter = 'all';

// C2 State
let currentThreatLevel = 5; // 5: LOW, 3: ELEVATED, 1: CRITICAL
let currentViewMode = 'grid'; // 'grid', 'map', 'split'
let audioMasterArmed = localStorage.getItem('ibvap_audio_armed') !== 'false';
let audioCtx = null;

// Camera Vision & Overlay State per Camera ID
// { camId: { mode: 'normal'|'clahe'|'flir'|'nvg', showBoxes: true, showDwell: true, showCrawl: true } }
const cameraHUDState = new Map();

// Virtual Fence Editor State
let fenceEditorState = {
  cameraId: null,
  points: [],
  zones: [],
};

// Tactical GIS Map (Leaflet)
let leafletMap = null;
let mapCameraMarkers = new Map();
let mapAlertRings = new Map();

// Preset Military BOP Coordinates along Indo-Nepal Frontier (Sector IV)
const SECTOR_COORDINATES = {
  center: [27.0125, 84.8780], // Sector IV Birgunj-Raxaul Axis
  bops: {
    bop17: { name: 'BOP-17 (Highland Perimeter)', lat: 27.0350, lng: 84.8920, azimuth: 45, camId: 'cam_bop17' },
    bop14: { name: 'BOP-14 (Highway Checkpost)', lat: 27.0080, lng: 84.8650, azimuth: 120, camId: 'cam_bop14' },
    bop12: { name: 'BOP-12 (Riverine Patrol)', lat: 26.9850, lng: 84.8510, azimuth: 270, camId: 'cam_bop12' },
    alpha: { name: 'Checkpost Alpha (Main Transit)', lat: 27.0190, lng: 84.8790, azimuth: 0, camId: 'cam_alpha' },
    hq:    { name: 'SSB Sector IV HQ & QRT Station', lat: 26.9950, lng: 84.8950, azimuth: 180, camId: 'cam_hq' },
  }
};


// ═══════════════════════════════════════════════════════════════
// Initialisation
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initMissionClocks();
  initAudioMaster();
  initThreatLevel();
  connectWebSocket();

  // Load backend data
  loadTacticalStatus();
  loadCameras();
  loadAlerts();
  loadWatchlist();
  loadANPRData();
  loadDemoVideos();

  // Initialize GIS Map (in DOM background)
  initTacticalGISMap();

  // Add Camera submit listener
  const addCamSubmit = document.getElementById('add-camera-submit');
  if (addCamSubmit) addCamSubmit.addEventListener('click', addCamera);

  const clearAlertsBtn = document.getElementById('clear-alerts-btn');
  if (clearAlertsBtn) clearAlertsBtn.addEventListener('click', clearAlerts);

  const saveFenceBtn = document.getElementById('save-fence-btn');
  if (saveFenceBtn) saveFenceBtn.addEventListener('click', saveFence);

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

  // Keyboard Shortcuts (ESC to close modals)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
      dismissFRSBanner();
    }
  });

  // Enable Web Audio on first user interaction (browser policy)
  const unlockAudio = () => {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    document.removeEventListener('click', unlockAudio);
    document.removeEventListener('keydown', unlockAudio);
  };
  document.addEventListener('click', unlockAudio);
  document.addEventListener('keydown', unlockAudio);
});


// ═══════════════════════════════════════════════════════════════
// 1. Dual Mission Clocks (IST and ZULU / UTC)
// ═══════════════════════════════════════════════════════════════
function initMissionClocks() {
  const istEl = document.getElementById('ist-clock');
  const zuluEl = document.getElementById('zulu-clock');

  function updateClocks() {
    const now = new Date();

    // Indian Standard Time (IST) -> UTC + 5:30
    const istOptions = {
      timeZone: 'Asia/Kolkata',
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    };
    const istTime = now.toLocaleTimeString('en-GB', istOptions);
    if (istEl) istEl.textContent = `${istTime} IST`;

    // Zulu / UTC Military Time
    const utcHours = String(now.getUTCHours()).padStart(2, '0');
    const utcMinutes = String(now.getUTCMinutes()).padStart(2, '0');
    const utcSeconds = String(now.getUTCSeconds()).padStart(2, '0');
    if (zuluEl) zuluEl.textContent = `${utcHours}:${utcMinutes}:${utcSeconds} Z`;
  }

  updateClocks();
  setInterval(updateClocks, 1000);
}


// ═══════════════════════════════════════════════════════════════
// 2. DEFCON / Threat Level Management
// ═══════════════════════════════════════════════════════════════
function initThreatLevel() {
  setThreatLevel(currentThreatLevel, false);
}

async function setThreatLevel(level, pushToBackend = true) {
  currentThreatLevel = parseInt(level, 10) || 5;

  const badge = document.getElementById('defcon-badge');
  const text = document.getElementById('defcon-text');
  const sub = document.getElementById('defcon-sub');

  if (!badge || !text || !sub) return;

  // Clear existing level classes
  badge.classList.remove('level-5', 'level-3', 'level-1');

  if (currentThreatLevel === 1) {
    badge.classList.add('level-1');
    text.textContent = 'LEVEL 1 • CRITICAL RED';
    sub.textContent = 'INTRUSION / INTERCEPT DETECTED';
    playTacticalSiren('critical');
  } else if (currentThreatLevel === 3) {
    badge.classList.add('level-3');
    text.textContent = 'LEVEL 3 • ELEVATED';
    sub.textContent = 'LOITERING / CROWD CLUSTER';
    playTacticalSiren('high');
  } else {
    badge.classList.add('level-5');
    text.textContent = 'LEVEL 5 • LOW';
    sub.textContent = 'PEACETIME SURVEILLANCE';
  }

  if (pushToBackend) {
    try {
      await fetch(`${API_BASE}/api/tactical/threat-level`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level: currentThreatLevel }),
      });
    } catch (e) {
      console.warn('Threat level sync failed:', e);
    }
  }
}

function evaluateDynamicThreatLevel() {
  if (alerts.length === 0) {
    setThreatLevel(5, false);
    return;
  }

  // Check recent unacknowledged alerts (within last 3 minutes)
  const now = Date.now();
  let hasCritical = false;
  let hasHigh = false;

  for (const a of alerts.slice(0, 15)) {
    const alertTime = new Date(a.created_at || now).getTime();
    if (now - alertTime < 3 * 60 * 1000) {
      if (a.severity === 'critical') hasCritical = true;
      if (a.severity === 'high') hasHigh = true;
    }
  }

  if (hasCritical) {
    if (currentThreatLevel !== 1) setThreatLevel(1, false);
  } else if (hasHigh) {
    if (currentThreatLevel !== 3 && currentThreatLevel !== 1) setThreatLevel(3, false);
  } else {
    if (currentThreatLevel !== 5) setThreatLevel(5, false);
  }
}


// ═══════════════════════════════════════════════════════════════
// 3. Tactical Audio Siren System (Web Audio API)
// ═══════════════════════════════════════════════════════════════
function initAudioMaster() {
  updateAudioButtonUI();
}

function toggleAudioMaster() {
  audioMasterArmed = !audioMasterArmed;
  localStorage.setItem('ibvap_audio_armed', audioMasterArmed);
  updateAudioButtonUI();

  if (audioMasterArmed) {
    playTacticalSiren('ping');
    showToast({ severity: 'low', message: 'Tactical Audio Siren: ARMED & READY' });
  } else {
    showToast({ severity: 'medium', message: 'Tactical Audio Siren: MUTED' });
  }
}

function updateAudioButtonUI() {
  const btn = document.getElementById('audio-siren-btn');
  const txt = document.getElementById('siren-status-text');
  if (!btn || !txt) return;

  if (audioMasterArmed) {
    btn.className = 'tactical-btn siren-btn armed';
    btn.title = 'Click to Mute Tactical Audio Siren';
    txt.textContent = 'SIREN: ARMED';
    btn.querySelector('.siren-icon').textContent = '🔊';
  } else {
    btn.className = 'tactical-btn siren-btn muted';
    btn.title = 'Click to Arm Tactical Audio Siren';
    txt.textContent = 'SIREN: MUTED';
    btn.querySelector('.siren-icon').textContent = '🔇';
  }
}

function playTacticalSiren(type = 'high') {
  if (!audioMasterArmed) return;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const now = audioCtx.currentTime;

    if (type === 'critical') {
      // High-priority dual-tone military warble siren (880 Hz alternating with 587 Hz)
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sawtooth';

      // 3 warble cycles
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.setValueAtTime(587, now + 0.15);
      osc.frequency.setValueAtTime(880, now + 0.30);
      osc.frequency.setValueAtTime(587, now + 0.45);
      osc.frequency.setValueAtTime(880, now + 0.60);
      osc.frequency.setValueAtTime(587, now + 0.75);

      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.95);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 1.0);

    } else if (type === 'ping' || type === 'frs' || type === 'anpr') {
      // Tactical biometric/plate hit target chirp
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(1200, now);
      osc.frequency.exponentialRampToValueAtTime(700, now + 0.25);

      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.28);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.3);

    } else {
      // Standard elevated alert chime (two-tone soft chime)
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(660, now);
      osc.frequency.setValueAtTime(880, now + 0.12);

      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.4);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.45);
    }
  } catch (e) {
    console.warn('Audio synthesis failed:', e);
  }
}

function testAudioAlert(severity) {
  playTacticalSiren(severity);
  showToast({ severity, message: `Tactical Siren Tested: ${severity.toUpperCase()}` });
}


// ═══════════════════════════════════════════════════════════════
// 4. Tactical GIS Border Map (Leaflet.js)
// ═══════════════════════════════════════════════════════════════
function initTacticalGISMap() {
  const mapContainer = document.getElementById('tactical-gis-map');
  if (!mapContainer || typeof L === 'undefined') return;

  try {
    leafletMap = L.map('tactical-gis-map', {
      center: SECTOR_COORDINATES.center,
      zoom: 13,
      zoomControl: true,
      attributionControl: false,
    });

    // Dark Matter Military Cartographic Tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
      subdomains: 'abcd',
    }).addTo(leafletMap);

    // Mouse coordinates tracker
    leafletMap.on('mousemove', (e) => {
      const coordEl = document.getElementById('mouse-coords');
      if (coordEl) {
        coordEl.textContent = `${e.latlng.lat.toFixed(4)}° N, ${e.latlng.lng.toFixed(4)}° E`;
      }
    });

    // Draw International Border Zero Line (Indo-Nepal Axis)
    const borderZeroLine = [
      [27.0500, 84.8400],
      [27.0350, 84.8700],
      [27.0125, 84.8780],
      [26.9900, 84.8850],
      [26.9700, 84.9100],
    ];
    L.polyline(borderZeroLine, {
      color: '#ff2244',
      weight: 2,
      dashArray: '8, 8',
      opacity: 0.8,
    }).addTo(leafletMap).bindTooltip('INDO-NEPAL ZERO LINE [PILLAR 440 - 446]', { permanent: false, className: 'map-tooltip' });

    // Add Preset Military BOPs & HQ
    for (const [key, bop] of Object.entries(SECTOR_COORDINATES.bops)) {
      renderBOPMarker(key, bop);
    }

  } catch (e) {
    console.error('Failed to init Leaflet GIS map:', e);
  }
}

function renderBOPMarker(key, bop) {
  if (!leafletMap) return;

  const iconHtml = `
    <div class="custom-mil-pin bop-pin" title="${bop.name}">
      <div class="pin-ring"></div>
      <div class="pin-core">🛡️</div>
      <div class="pin-lbl">${key.toUpperCase()}</div>
    </div>
  `;

  const customIcon = L.divIcon({
    html: iconHtml,
    className: 'military-map-marker',
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });

  const marker = L.marker([bop.lat, bop.lng], { icon: customIcon }).addTo(leafletMap);

  // Render Camera FOV Cone (Semi-transparent fan)
  renderCameraFOVCone(bop.lat, bop.lng, bop.azimuth, 400);

  // Popup with live surveillance preview
  const popupHtml = `
    <div class="map-cam-popup">
      <h4>📍 ${escapeHtml(bop.name)}</h4>
      <div class="map-cam-meta">
        <span>COORDINATES: ${bop.lat.toFixed(4)}° N, ${bop.lng.toFixed(4)}° E</span>
        <span>SECTOR: SECTOR-IV (INDO-NEPAL)</span>
        <span>RADAR AZIMUTH: ${bop.azimuth}°</span>
      </div>
      <img class="map-cam-preview-img" src="${API_BASE}/api/cameras/${bop.camId}/stream" onerror="this.src='https://images.unsplash.com/photo-1541888946425-d0fbb186c5f7?w=300&auto=format&fit=crop&q=80'" alt="Camera Live">
      <button class="btn btn-primary btn-sm btn-block" onclick="focusCameraFromMap('${bop.camId}')">
        📹 FOCUS SURVEILLANCE FEED
      </button>
    </div>
  `;

  marker.bindPopup(popupHtml);
  mapCameraMarkers.set(bop.camId, { marker, bop });
}

function renderCameraFOVCone(lat, lng, azimuth, rangeMeters = 300) {
  if (!leafletMap) return;

  // Approximate cone coordinates
  const fovDegrees = 60;
  const halfFov = fovDegrees / 2;
  const startAngle = (azimuth - halfFov) * (Math.PI / 180);
  const endAngle = (azimuth + halfFov) * (Math.PI / 180);

  // Convert meters to lat/lng delta
  const latDelta = rangeMeters / 111320;
  const lngDelta = rangeMeters / (111320 * Math.cos(lat * (Math.PI / 180)));

  const p1 = [lat, lng];
  const p2 = [lat + Math.cos(startAngle) * latDelta, lng + Math.sin(startAngle) * lngDelta];
  const p3 = [lat + Math.cos(endAngle) * latDelta, lng + Math.sin(endAngle) * lngDelta];

  L.polygon([p1, p2, p3], {
    color: '#00ff88',
    weight: 1,
    fillColor: '#00ff88',
    fillOpacity: 0.15,
    dashArray: '3, 3'
  }).addTo(leafletMap);
}

function triggerMapRadarAlert(camId) {
  if (!leafletMap) return;

  let targetMarker = mapCameraMarkers.get(camId);
  if (!targetMarker) {
    // If not found, pulse on BOP-17 or center
    targetMarker = mapCameraMarkers.get('cam_bop17');
  }
  if (!targetMarker) return;

  const latlng = targetMarker.marker.getLatLng();

  // Create pulsing red radar circle
  const ringIcon = L.divIcon({
    html: '<div class="radar-alert-ring"></div>',
    className: 'radar-ring-wrapper',
    iconSize: [40, 40],
    iconAnchor: [20, 20],
  });

  const alertMarker = L.marker(latlng, { icon: ringIcon }).addTo(leafletMap);

  // Auto remove ring after 8 seconds
  setTimeout(() => {
    if (leafletMap && alertMarker) leafletMap.removeLayer(alertMarker);
  }, 8000);
}

function panToBOP(bopKey) {
  const bop = SECTOR_COORDINATES.bops[bopKey];
  if (!bop || !leafletMap) return;

  leafletMap.flyTo([bop.lat, bop.lng], 15, { duration: 1.2 });
  const entry = mapCameraMarkers.get(bop.camId);
  if (entry && entry.marker) {
    entry.marker.openPopup();
  }
}

function focusCameraFromMap(camId) {
  setViewMode('grid');
  const card = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
  if (card) {
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('intrusion');
    setTimeout(() => card.classList.remove('intrusion'), 3000);
  }
}


// ═══════════════════════════════════════════════════════════════
// 5. C2 View Mode Switcher ('grid', 'map', 'split')
// ═══════════════════════════════════════════════════════════════
function setViewMode(mode) {
  currentViewMode = mode;

  const tabGrid = document.getElementById('tab-grid');
  const tabMap = document.getElementById('tab-map');
  const tabSplit = document.getElementById('tab-split');

  const camSection = document.getElementById('camera-section');
  const mapSection = document.getElementById('gis-map-section');
  const appContainer = document.getElementById('app-container');

  [tabGrid, tabMap, tabSplit].forEach(t => t && t.classList.remove('active'));
  appContainer.classList.remove('view-split');

  if (mode === 'grid') {
    if (tabGrid) tabGrid.classList.add('active');
    if (camSection) camSection.style.display = 'flex';
    if (mapSection) mapSection.style.display = 'none';
  } else if (mode === 'map') {
    if (tabMap) tabMap.classList.add('active');
    if (camSection) camSection.style.display = 'none';
    if (mapSection) mapSection.style.display = 'flex';
    if (leafletMap) {
      setTimeout(() => leafletMap.invalidateSize(), 200);
    }
  } else if (mode === 'split') {
    if (tabSplit) tabSplit.classList.add('active');
    appContainer.classList.add('view-split');
    if (camSection) camSection.style.display = 'flex';
    if (mapSection) mapSection.style.display = 'flex';
    if (leafletMap) {
      setTimeout(() => leafletMap.invalidateSize(), 200);
    }
  }
}


// ═══════════════════════════════════════════════════════════════
// 6. FRS Watchlist Management & Live Match Banner
// ═══════════════════════════════════════════════════════════════
async function loadWatchlist() {
  try {
    const res = await fetch(`${API_BASE}/api/frs/watchlist`);
    const data = await res.json();
    watchlistTargets = data.watchlist || [];

    const badge = document.getElementById('frs-badge');
    const targetCount = document.getElementById('frs-target-count');
    if (badge) badge.textContent = watchlistTargets.length;
    if (targetCount) targetCount.textContent = watchlistTargets.length;

    renderWatchlist(watchlistTargets);
  } catch (e) {
    console.error('Failed to load FRS watchlist:', e);
  }
}

function renderWatchlist(targetsToRender) {
  const container = document.getElementById('frs-target-list');
  if (!container) return;

  if (targetsToRender.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding: 20px;">
        <span class="icon">👤</span>
        <p>NO BIOMETRIC TARGETS</p>
      </div>
    `;
    return;
  }

  container.innerHTML = targetsToRender.map(t => {
    const photo = t.photo_url || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80';
    return `
      <div class="frs-target-card">
        <div class="target-thumb">
          <img src="${escapeHtml(photo)}" alt="${escapeHtml(t.name)}" onerror="this.src='https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80'">
          <div class="reticle-box"></div>
        </div>
        <div class="target-details">
          <div class="target-top-row">
            <span class="target-name">${escapeHtml(t.name)}</span>
            <span class="danger-tag ${escapeHtml(t.danger_level)}">${escapeHtml(t.danger_level)}</span>
          </div>
          <div class="target-category">${escapeHtml(t.category)}</div>
          <div class="target-bottom-row">
            <span>Matches: <strong>${t.match_count || 0}</strong></span>
            <span>Last: ${escapeHtml(t.last_seen || 'Never')}</span>
            <div style="display:flex; gap:4px;">
              <button class="btn btn-secondary btn-sm" onclick="triggerSimulatedFRS('${t.id}')" title="Test match this suspect">🎯 MATCH</button>
              <button class="btn btn-danger btn-sm" onclick="deleteTarget('${t.id}')">✕</button>
            </div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function filterWatchlist() {
  const query = document.getElementById('frs-search-input').value.toLowerCase();
  const filtered = watchlistTargets.filter(t => 
    t.name.toLowerCase().includes(query) || 
    t.category.toLowerCase().includes(query)
  );
  renderWatchlist(filtered);
}

async function handleEnrollTarget(event) {
  event.preventDefault();
  const name = document.getElementById('target-name').value.trim();
  const category = document.getElementById('target-category').value;
  const danger_level = document.getElementById('target-danger').value;
  const photo_url = document.getElementById('target-photo').value.trim();
  const notes = document.getElementById('target-notes').value.trim();

  try {
    const res = await fetch(`${API_BASE}/api/frs/watchlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, category, danger_level, photo_url, notes }),
    });
    if (res.ok) {
      showToast({ severity: 'low', message: `Target '${name}' enrolled into FRS registry` });
      document.getElementById('frs-enroll-form').reset();
      await loadWatchlist();
    }
  } catch (e) {
    alert('Enrollment error: ' + e.message);
  }
}

async function deleteTarget(targetId) {
  if (!confirm('Remove this biometric target from FRS watchlist?')) return;
  try {
    await fetch(`${API_BASE}/api/frs/watchlist/${targetId}`, { method: 'DELETE' });
    await loadWatchlist();
  } catch (e) {
    alert('Delete error: ' + e.message);
  }
}

async function triggerSimulatedFRS(targetId) {
  try {
    const url = targetId ? `${API_BASE}/api/frs/simulate-match?target_id=${targetId}` : `${API_BASE}/api/frs/simulate-match`;
    const res = await fetch(url, { method: 'POST' });
    const data = await res.json();
    if (data.target) {
      displayFRSBanner(data.target, data.alert);
      await loadWatchlist();
    }
  } catch (e) {
    console.warn('Simulate FRS error:', e);
  }
}

function displayFRSBanner(target, alertObj) {
  const banner = document.getElementById('frs-intercept-banner');
  if (!banner) return;

  const photo = document.getElementById('frs-banner-photo');
  const name = document.getElementById('frs-banner-name');
  const cat = document.getElementById('frs-banner-cat');
  const loc = document.getElementById('frs-banner-loc');
  const danger = document.getElementById('frs-banner-danger');
  const conf = document.getElementById('frs-banner-conf');
  const time = document.getElementById('frs-banner-time');

  if (photo) photo.src = target.photo_url || 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80';
  if (name) name.textContent = target.name;
  if (cat) cat.textContent = target.category;
  if (loc) loc.textContent = alertObj?.details?.location || 'BOP-17 Sector IV Perimeter';
  if (danger) {
    danger.textContent = target.danger_level;
    danger.className = `target-danger ${target.danger_level}`;
  }
  if (conf) conf.textContent = `${alertObj?.details?.confidence || 98.4}% CONFIDENCE`;
  if (time) time.textContent = new Date().toLocaleTimeString('en-GB');

  banner.classList.remove('frs-banner-hidden');

  // Trigger high alert audio & DEFCON 1
  playTacticalSiren('critical');
  setThreatLevel(1, true);
  triggerMapRadarAlert('cam_bop17');
}

function dismissFRSBanner() {
  const banner = document.getElementById('frs-intercept-banner');
  if (banner) banner.classList.add('frs-banner-hidden');
}

function dispatchQRT() {
  dismissFRSBanner();
  showToast({
    severity: 'critical',
    message: '⚡ SSB QUICK REACTION TEAM (QRT) DISPATCHED TO BOP-17 PERIMETER!'
  });
}

function openFRSModal() { openModal('frs-modal'); }


// ═══════════════════════════════════════════════════════════════
// 7. ANPR Vehicle Scanner & Hotlist Management
// ═══════════════════════════════════════════════════════════════
async function loadANPRData() {
  try {
    const [hotlistRes, scansRes] = await Promise.all([
      fetch(`${API_BASE}/api/anpr/hotlist`),
      fetch(`${API_BASE}/api/anpr/scans`),
    ]);

    const hotlistData = await hotlistRes.json();
    const scansData = await scansRes.json();

    hotlistVehicles = hotlistData.hotlist || [];
    plateScans = scansData.scans || [];

    const badge = document.getElementById('anpr-badge');
    const hotlistCount = document.getElementById('anpr-hotlist-count');
    if (badge) badge.textContent = hotlistVehicles.length;
    if (hotlistCount) hotlistCount.textContent = hotlistVehicles.length;

    renderANPRScans(plateScans);
    renderANPRHotlist(hotlistVehicles);
  } catch (e) {
    console.error('Failed to load ANPR data:', e);
  }
}

function renderANPRScans(scansToRender) {
  const container = document.getElementById('anpr-scans-list');
  if (!container) return;

  if (scansToRender.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>NO RECENT SCANS</p></div>';
    return;
  }

  container.innerHTML = scansToRender.slice(0, 30).map(s => {
    const isCommercial = s.plate_number.startsWith('BR') || s.plate_number.startsWith('NL');
    const isHit = s.is_hotlist === 1;

    return `
      <div class="anpr-scan-card ${isHit ? 'hotlist-hit' : ''}">
        <div class="license-plate ${isCommercial ? 'commercial' : ''}">
          <div class="plate-ind">
            <span>🇮🇳</span>
            <span>IND</span>
          </div>
          <span class="plate-text">${escapeHtml(s.plate_number)}</span>
        </div>
        <div style="flex:1; margin-left: 8px;">
          <div style="font-family:var(--font-heading); font-size:12px; font-weight:700; color:#fff;">
            ${escapeHtml(s.vehicle_type)}
          </div>
          <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-secondary);">
            📹 ${escapeHtml(s.camera_id)} • Confidence: ${(s.confidence * 100).toFixed(1)}%
          </div>
        </div>
        <div class="anpr-scan-meta">
          <span class="anpr-scan-status ${isHit ? 'status-hit' : 'status-cleared'}">
            ${isHit ? '🚨 HOTLIST HIT' : '✓ CLEARED'}
          </span>
          <span style="font-family:var(--font-mono); font-size:9px; color:var(--text-muted);">
            ${escapeHtml(s.timestamp ? s.timestamp.split(' ')[1] : 'NOW')}
          </span>
        </div>
      </div>
    `;
  }).join('');
}

function renderANPRHotlist(hotlistToRender) {
  const container = document.getElementById('anpr-hotlist-items');
  if (!container) return;

  if (hotlistToRender.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>NO HOTLISTED VEHICLES</p></div>';
    return;
  }

  container.innerHTML = hotlistToRender.map(v => `
    <div class="hotlist-item">
      <div>
        <strong style="font-family:var(--font-heading); color:#ff2244;">${escapeHtml(v.plate_number)}</strong>
        <span style="font-size:11px; margin-left:6px; color:#fff;">${escapeHtml(v.vehicle_model)}</span>
        <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">${escapeHtml(v.reason)}</div>
      </div>
      <button class="btn btn-danger btn-sm" onclick="deleteHotlistVehicle('${v.plate_number}')">✕</button>
    </div>
  `).join('');
}

function filterPlates() {
  const query = document.getElementById('plate-search-input').value.trim().toUpperCase();
  const filtered = plateScans.filter(s => s.plate_number.toUpperCase().includes(query));
  renderANPRScans(filtered);
}

async function handleAddToHotlist(event) {
  event.preventDefault();
  const plate_number = document.getElementById('plate-no').value.trim().toUpperCase();
  const vehicle_model = document.getElementById('plate-model').value.trim();
  const reason = document.getElementById('plate-reason').value.trim();
  const danger_level = document.getElementById('plate-danger').value;

  try {
    const res = await fetch(`${API_BASE}/api/anpr/hotlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plate_number, vehicle_model, reason, danger_level }),
    });
    if (res.ok) {
      showToast({ severity: 'high', message: `Vehicle [${plate_number}] blacklisted on ANPR Hotlist` });
      document.getElementById('anpr-hotlist-form').reset();
      await loadANPRData();
    }
  } catch (e) {
    alert('Error adding to hotlist: ' + e.message);
  }
}

async function deleteHotlistVehicle(plate) {
  if (!confirm(`Remove vehicle ${plate} from ANPR hotlist?`)) return;
  try {
    await fetch(`${API_BASE}/api/anpr/hotlist/${encodeURIComponent(plate)}`, { method: 'DELETE' });
    await loadANPRData();
  } catch (e) {
    alert('Delete error: ' + e.message);
  }
}

async function triggerSimulatedANPR() {
  try {
    const res = await fetch(`${API_BASE}/api/anpr/simulate-scan`, { method: 'POST' });
    const data = await res.json();
    if (data.scan) {
      showToast({
        severity: 'critical',
        message: `🚨 ANPR HIT: Flagged Plate [${data.scan.plate_number}] Intercepted at BOP Checkpost!`
      });
      playTacticalSiren('critical');
      setThreatLevel(1, true);
      triggerMapRadarAlert('cam_bop14');
      await loadANPRData();
    }
  } catch (e) {
    console.warn('Simulate ANPR error:', e);
  }
}

function openANPRModal() { openModal('anpr-modal'); }


// ═══════════════════════════════════════════════════════════════
// 8. Tactical Night Vision & Behavioral Analytics Overlay Controls
// ═══════════════════════════════════════════════════════════════
function getCameraHUD(camId) {
  if (!cameraHUDState.has(camId)) {
    cameraHUDState.set(camId, {
      mode: 'normal',
      showBoxes: true,
      showDwell: true,
      showCrawl: true,
    });
  }
  return cameraHUDState.get(camId);
}

async function setCameraVisionMode(camId, mode) {
  const state = getCameraHUD(camId);
  state.mode = mode;

  const card = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
  if (!card) return;

  // Update vision chip active states
  card.querySelectorAll('.vision-chip').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  // Apply CSS filter to video feed img
  const img = card.querySelector('.camera-feed img');
  if (img) {
    img.className = `vision-${mode}`;
  }

  // Toggle NVG phosphor edge vignette on card
  card.classList.toggle('nvg-active', mode === 'nvg');

  // Persist to backend
  try {
    const backendMode = (mode === 'normal' || mode === 'off') ? 'off' : (mode === 'flir' ? 'thermal' : mode);
    await fetch(`${API_BASE}/api/cameras/${camId}/night-mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: backendMode }),
    });
  } catch (e) {
    console.warn('Vision mode sync failed:', e);
  }

}

function toggleHUDOverlay(camId, type) {
  const state = getCameraHUD(camId);
  if (type === 'dwell') state.showDwell = !state.showDwell;
  if (type === 'crawl') state.showCrawl = !state.showCrawl;
  if (type === 'boxes') state.showBoxes = !state.showBoxes;

  const card = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
  if (!card) return;

  const dwellTag = card.querySelector('.hud-dwell-tag');
  const crawlTag = card.querySelector('.hud-crawl-tag');

  if (dwellTag) dwellTag.style.display = state.showDwell ? 'inline-flex' : 'none';
  if (crawlTag) crawlTag.style.display = state.showCrawl ? 'inline-flex' : 'none';
}


// ═══════════════════════════════════════════════════════════════
// 9. Camera Feed Grid & Card Generation
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

function renderCameraGrid() {
  const grid = document.getElementById('camera-grid');
  if (!grid) return;

  const count = cameras.length;
  grid.className = count <= 1 ? 'grid-1' : count <= 2 ? 'grid-2' : count <= 4 ? 'grid-4' : 'grid-6';

  const existingCards = new Map();
  grid.querySelectorAll('.camera-card').forEach(card => {
    existingCards.set(card.dataset.camId, card);
  });

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

  if (cameras.length === 0) {
    if (!grid.querySelector('.empty-state')) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1; grid-row: 1 / -1; padding: 40px; text-align: center;">
          <span class="icon" style="font-size:42px;">📹</span>
          <p style="font-family:var(--font-heading); font-size:16px; margin-top:8px;">NO CAMERAS CONFIGURED</p>
          <p style="color:var(--text-muted); font-size:12px;">Click "+ ADD CAMERA" or launch a Preset above to commence surveillance.</p>
        </div>
      `;
    }
  }
}

function createCameraCard(cam) {
  const card = document.createElement('div');
  card.className = 'camera-card';
  card.dataset.camId = cam.id;

  const hudState = getCameraHUD(cam.id);
  if (cam.vision_mode) hudState.mode = cam.vision_mode;

  card.innerHTML = `
    <!-- Camera Card Header -->
    <div class="camera-header">
      <div class="camera-title-group">
        <span class="camera-name">${escapeHtml(cam.name)}</span>
        <span style="font-size:10px; font-family:var(--font-mono); color:var(--text-muted);">
          [<span style="color:#00ff00; font-weight:700;">● Human</span> | <span style="color:#ffff00; font-weight:700;">● Veh</span>]
        </span>
      </div>
      <div class="camera-header-right">
        <span class="camera-badge ${cam.status === 'active' ? 'badge-live' : 'badge-offline'}">
          ${cam.status === 'active' ? '● LIVE' : '○ OFF'}
        </span>
      </div>
    </div>

    <!-- Video Feed Container -->
    <div class="camera-feed">
      <!-- Behavioral Dwell & Crawling HUD Overlay Badges -->
      <div class="hud-card-overlay">
        <div class="hud-tag hud-dwell-tag" style="${hudState.showDwell ? '' : 'display:none;'}">
          ⏱️ ID #04 • 42s DWELL [LOITERING]
        </div>
        <div class="hud-tag hud-crawl-tag" style="${hudState.showCrawl ? '' : 'display:none;'}">
          ⚠️ PRONE / CRAWLING INTRUDER (AR 2.8:1)
        </div>
      </div>

      ${cam.status === 'active'
        ? `<img class="vision-${hudState.mode}" src="${API_BASE}/api/cameras/${cam.id}/stream" alt="${escapeHtml(cam.name)}" onerror="handleStreamError(this, '${cam.id}')">`
        : `<div class="placeholder"><span class="icon">📹</span>Camera offline</div>`
      }
    </div>

    <!-- Night Vision & HUD Mode Bar -->
    <div class="camera-subtoolbar">
      <div class="vision-mode-cluster">
        <span style="color:var(--text-muted); font-size:9px;">NV:</span>
        <button class="vision-chip ${hudState.mode === 'normal' ? 'active' : ''}" data-mode="normal" onclick="setCameraVisionMode('${cam.id}', 'normal')">NORM</button>
        <button class="vision-chip ${hudState.mode === 'clahe' ? 'active' : ''}" data-mode="clahe" onclick="setCameraVisionMode('${cam.id}', 'clahe')">CLAHE</button>
        <button class="vision-chip ${hudState.mode === 'flir' ? 'active' : ''}" data-mode="flir" onclick="setCameraVisionMode('${cam.id}', 'flir')">FLIR</button>
        <button class="vision-chip ${hudState.mode === 'nvg' ? 'active' : ''}" data-mode="nvg" onclick="setCameraVisionMode('${cam.id}', 'nvg')">NVG</button>
      </div>

      <div class="hud-toggle-cluster">
        <label class="hud-toggle-lbl">
          <input type="checkbox" ${hudState.showDwell ? 'checked' : ''} onchange="toggleHUDOverlay('${cam.id}', 'dwell')">DWELL
        </label>
        <label class="hud-toggle-lbl">
          <input type="checkbox" ${hudState.showCrawl ? 'checked' : ''} onchange="toggleHUDOverlay('${cam.id}', 'crawl')">CRAWL
        </label>
      </div>
    </div>

    <!-- Bottom Actions Bar -->
    <div class="camera-actions">
      <div class="cam-stat" title="Humans Detected">
        👥 <span class="val stat-persons text-green">0</span>
      </div>
      <div class="cam-stat" title="Vehicles Detected">
        🚗 <span class="val stat-vehicles text-amber">0</span>
      </div>
      <div class="cam-stat" title="Framerate">
        ⚡ <span class="val stat-fps">0</span> fps
      </div>
      <div class="spacer"></div>
      <button class="cam-btn" onclick="openFenceEditor('${cam.id}')" title="Virtual Perimeter Fence">🔲</button>
      <button class="cam-btn" onclick="panToCameraMap('${cam.id}')" title="View on Tactical GIS Map">🗺️</button>
      ${cam.status === 'active'
        ? `<button class="cam-btn btn-start-stop" onclick="stopCamera('${cam.id}')" title="Stop">⏹</button>`
        : `<button class="cam-btn btn-start-stop" onclick="startCamera('${cam.id}')" title="Start">▶</button>`
      }
      <button class="cam-btn danger" onclick="deleteCamera('${cam.id}')" title="Delete">✕</button>
    </div>
  `;

  return card;
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
        const hudState = getCameraHUD(cam.id);
        const img = document.createElement('img');
        img.className = `vision-${hudState.mode}`;
        img.src = `${API_BASE}/api/cameras/${cam.id}/stream`;
        img.alt = cam.name;
        img.onerror = () => handleStreamError(img, cam.id);
        feed.appendChild(img);
      }
    } else {
      const existingImg = feed.querySelector('img');
      if (existingImg) existingImg.remove();
      if (!feed.querySelector('.placeholder')) {
        feed.insertAdjacentHTML('beforeend', '<div class="placeholder"><span class="icon">📹</span>Camera offline</div>');
      }
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

function panToCameraMap(camId) {
  setViewMode('map');
  const target = mapCameraMarkers.get(camId);
  if (target && leafletMap) {
    leafletMap.flyTo(target.marker.getLatLng(), 15, { duration: 1.0 });
    target.marker.openPopup();
  }
}

function handleStreamError(img, camId) {
  console.warn(`[Stream] Reconnecting camera ${camId}...`);
  setTimeout(() => {
    if (img && img.parentElement) {
      img.src = `${API_BASE}/api/cameras/${camId}/stream?t=${Date.now()}`;
    }
  }, 1500);
}


// ═══════════════════════════════════════════════════════════════
// 10. Real-time WebSocket, Stats & Alerts
// ═══════════════════════════════════════════════════════════════
function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('[WS] Connected to Tactical C2 Event Stream');
    document.getElementById('system-dot').classList.add('active');
    document.getElementById('system-status').textContent = 'ONLINE';
    if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'alert') handleNewAlert(msg.data);
      if (msg.type === 'stats') handleStats(msg.data);
      if (msg.type === 'threat_level') setThreatLevel(msg.data.level, false);
    } catch (e) {
      console.error('[WS] Parse error:', e);
    }
  };

  ws.onclose = () => {
    document.getElementById('system-dot').classList.remove('active');
    document.getElementById('system-status').textContent = 'RECONNECTING...';
    wsReconnectTimer = setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => ws.close();

  // Keep-alive heartbeat ping
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 30000);
}

function handleStats(stats) {
  document.getElementById('total-persons').textContent = stats.total_persons || 0;
  document.getElementById('total-vehicles').textContent = stats.total_vehicles || 0;
  document.getElementById('total-intrusions').textContent = stats.total_intrusions || 0;

  const camStats = stats.cameras || {};
  const fpsValues = Object.values(camStats).map(c => c.fps || 0).filter(f => f > 0);
  const avgFps = fpsValues.length ? (fpsValues.reduce((a, b) => a + b, 0) / fpsValues.length) : 0;
  document.getElementById('avg-fps').textContent = Math.round(avgFps);

  for (const [camId, camStat] of Object.entries(camStats)) {
    const card = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
    if (!card) continue;

    const fpsEl = card.querySelector('.stat-fps');
    const persEl = card.querySelector('.stat-persons');
    const vehEl = card.querySelector('.stat-vehicles');
    if (fpsEl) fpsEl.textContent = Math.round(camStat.fps || 0);
    if (persEl) persEl.textContent = camStat.persons || 0;
    if (vehEl) vehEl.textContent = camStat.vehicles || 0;

    if (camStat.intrusions > 0) {
      card.classList.add('intrusion');
      triggerMapRadarAlert(camId);
    } else {
      card.classList.remove('intrusion');
    }
  }
}

function handleNewAlert(alert) {
  alerts.unshift(alert);
  renderAlerts();
  updateAlertCount();
  showToast(alert);
  playTacticalSiren(alert.severity);

  // If intrusion, FRS match or ANPR hotlist, pulse map and evaluate threat level
  if (alert.severity === 'critical') {
    setThreatLevel(1, false);
    triggerMapRadarAlert(alert.camera_id);
  } else if (alert.severity === 'high') {
    if (currentThreatLevel !== 1) setThreatLevel(3, false);
  }

  // If FRS Match, show Banner
  if (alert.alert_type === 'frs_watchlist_match' && alert.details) {
    displayFRSBanner(alert.details, alert);
  }
}

async function loadAlerts() {
  try {
    const res = await fetch(`${API_BASE}/api/alerts?limit=100`);
    const data = await res.json();
    alerts = data.alerts || [];
    renderAlerts();
    updateAlertCount();
    evaluateDynamicThreatLevel();
  } catch (e) {
    console.error('Failed to load alerts:', e);
  }
}

function renderAlerts() {
  const list = document.getElementById('alert-list');
  const empty = document.getElementById('alerts-empty');

  const filtered = alertFilter === 'all' 
    ? alerts 
    : alerts.filter(a => a.severity === alertFilter);

  if (filtered.length === 0) {
    list.innerHTML = `
      <div class="empty-state" id="alerts-empty">
        <span class="icon">📡</span>
        <p>NO ACTIVE ALERTS</p>
        <p>SSB Perimeter AI Monitoring Active</p>
      </div>
    `;
    return;
  }

  list.innerHTML = filtered.slice(0, 100).map((alert, i) => {
    const severityClass = alert.severity || 'medium';
    const typeLabel = (alert.alert_type || '').replace(/_/g, ' ').toUpperCase();
    const time = alert.created_at ? new Date(alert.created_at).toLocaleTimeString('en-GB') : '';
    const camName = cameras.find(c => c.id === alert.camera_id)?.name || alert.camera_id;

    return `
      <div class="alert-card ${severityClass} ${i === 0 ? 'new' : ''}" onclick="focusCameraFromMap('${alert.camera_id}')">
        <div class="alert-type">
          <span class="severity-dot"></span>
          ${typeLabel}
        </div>
        <div class="alert-message">${escapeHtml(alert.message)}</div>
        <div class="alert-meta">
          <span>📹 ${escapeHtml(camName)}</span>
          <span>${time}</span>
        </div>
      </div>
    `;
  }).join('');
}

function filterAlerts(severity) {
  alertFilter = severity;
  document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
  if (window.event && window.event.target) window.event.target.classList.add('active');
  renderAlerts();
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
  setThreatLevel(5, true);
}


// ═══════════════════════════════════════════════════════════════
// 11. Tactical Status & Sentry Operator Information
// ═══════════════════════════════════════════════════════════════
async function loadTacticalStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/tactical/status`);
    const data = await res.json();
    if (data.operator_name) {
      const opEl = document.getElementById('operator-display');
      if (opEl) opEl.textContent = data.operator_name;
    }
    if (data.sector_name) {
      const secEl = document.getElementById('sector-display');
      if (secEl) secEl.textContent = data.sector_name;
    }
    if (data.active_bops) {
      const bopEl = document.getElementById('bop-display');
      if (bopEl) bopEl.textContent = `${data.active_bops} / ${data.total_bops} ACTIVE`;
    }
  } catch (e) {
    console.warn('Failed to load tactical status:', e);
  }
}


// ═══════════════════════════════════════════════════════════════
// 12. Quick Presets & Video Stream Launcher
// ═══════════════════════════════════════════════════════════════
async function playDirectFilepath() {
  const input = document.getElementById('quick-filepath-input');
  const source = input ? input.value.trim() : '';
  if (!source) {
    alert('Please enter or paste a valid YouTube link, live stream URL, or video filepath');
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
    let location = 'Border Sector IV Feed';
    if (source.includes('youtube.com') || source.includes('youtu.be')) {
      name = 'YouTube Live CCTV';
      location = 'YouTube Stream';
    } else if (source.startsWith('rtsp://')) {
      name = 'RTSP Camera Feed';
      location = 'Border RTSP Sentry';
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
      message: `Surveillance Active: ${name} [AI Tracking Active]`
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
  if (window.event && window.event.target) window.event.target.classList.add('active');

  const input = document.getElementById('quick-filepath-input');
  if (input) input.value = url;

  const btn = document.getElementById('play-filepath-btn');
  if (btn) {
    btn.textContent = 'RESOLVING...';
    btn.disabled = true;
  }

  try {
    const res = await fetch(`${API_BASE}/api/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: label,
        source: url,
        location: 'YouTube 24/7 Live Feed',
        auto_start: true,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to connect stream');
      return;
    }

    showToast({
      severity: 'low',
      message: `Live YouTube Feed Active: ${label}`
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
  document.querySelectorAll('.chip-btn').forEach(btn => btn.classList.remove('active'));
  if (window.event && window.event.target) window.event.target.classList.add('active');

  const defaultPath = `demo_videos/${filename}`;
  const input = document.getElementById('quick-filepath-input');
  if (input) input.value = defaultPath;

  try {
    const res = await fetch(`${API_BASE}/api/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: label,
        source: defaultPath,
        location: 'Sector IV Border Checkpost',
        auto_start: true,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to start demo preset');
      return;
    }

    showToast({ severity: 'low', message: `Preset Active: ${label}` });
    await loadCameras();
  } catch (e) {
    alert('Error starting demo: ' + e.message);
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
    const res = await fetch(`${API_BASE}/api/upload-video`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to upload video');
      return;
    }
    showToast({ severity: 'low', message: `Uploaded & Playing: ${file.name}` });
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

async function loadDemoVideos() {
  try {
    const res = await fetch(`${API_BASE}/api/demo-videos`);
    const data = await res.json();
    const videos = data.videos || [];
    if (videos.length > 0) {
      const section = document.getElementById('demo-videos-section');
      const list = document.getElementById('demo-video-list');
      if (section && list) {
        section.style.display = 'block';
        list.innerHTML = videos.map(v => `
          <button class="btn btn-secondary btn-sm" onclick="quickAddDemo('${v.path.replace(/\\/g, '\\\\')}', '${v.name}')">
            🎬 ${v.name} (${v.size_mb} MB)
          </button>
        `).join('');
      }
    }
  } catch (e) { /* Demo videos optional */ }
}

function quickAddDemo(path, name) {
  document.getElementById('cam-name').value = name;
  document.getElementById('cam-source').value = path;
}


// ═══════════════════════════════════════════════════════════════
// 13. Camera Actions (Add, Stop, Delete, Clear)
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

async function startCamera(camId) {
  try {
    await fetch(`${API_BASE}/api/cameras/${camId}/start`, { method: 'POST' });
  } catch (e) {
    console.error('Start error:', e);
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
  if (!confirm('Decommission this surveillance camera?')) return;
  await fetch(`${API_BASE}/api/cameras/${camId}`, { method: 'DELETE' });
  await loadCameras();
}

async function clearAllCameras() {
  if (!confirm('Clear all cameras from the surveillance grid?')) return;
  try {
    await fetch(`${API_BASE}/api/cameras`, { method: 'DELETE' });
    showToast({ severity: 'low', message: 'All cameras decommissioned' });
    await loadCameras();
  } catch (e) {
    alert('Failed to clear cameras: ' + e.message);
  }
}


// ═══════════════════════════════════════════════════════════════
// 14. Virtual Fence Editor
// ═══════════════════════════════════════════════════════════════
function openFenceEditor(camId) {
  fenceEditorState.cameraId = camId;
  fenceEditorState.points = [];
  fenceEditorState.zones = [];

  const img = document.getElementById('fence-frame');
  img.src = `${API_BASE}/api/cameras/${camId}/stream`;
  img.onload = () => setupFenceCanvas();
  setTimeout(setupFenceCanvas, 500);

  openModal('fence-modal');
}

function setupFenceCanvas() {
  const wrapper = document.getElementById('fence-canvas-wrapper');
  const canvas = document.getElementById('fence-canvas');
  if (!wrapper || !canvas) return;

  canvas.width = wrapper.offsetWidth;
  canvas.height = wrapper.offsetHeight;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  canvas.onmousedown = (e) => {
    if (e.button === 2) {
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
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const zone of fenceEditorState.zones) {
    drawPolygon(ctx, zone.points, 'rgba(255, 23, 68, 0.25)', 'rgba(255, 23, 68, 0.9)');
  }

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
    displayPoints: [...fenceEditorState.points],
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
  if (fenceEditorState.points.length >= 3) closePolygon();

  const camId = fenceEditorState.cameraId;
  const zones = fenceEditorState.zones.map(z => ({
    id: z.id,
    name: z.name,
    points: z.points,
  }));

  try {
    await fetch(`${API_BASE}/api/cameras/${camId}/fence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones }),
    });
    closeFenceEditor();
    showToast({ message: `Virtual fence armed on camera ${camId}`, severity: 'low' });
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
  const img = document.getElementById('fence-frame');
  if (img) img.src = '';
  closeModal('fence-modal');
}


// ═══════════════════════════════════════════════════════════════
// 15. UI Helpers (Modals & Toast)
// ═══════════════════════════════════════════════════════════════
function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('active');
}

function showToast(alertObj) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  const icon = alertObj.severity === 'critical' ? '🚨' : alertObj.severity === 'high' ? '⚠️' : '📡';
  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <span class="toast-message">${escapeHtml(alertObj.message)}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('exiting');
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
