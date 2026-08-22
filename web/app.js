// ── helpers ──────────────────────────────────────────────────────────
const TOKEN_KEY = 'cameraVisionApiToken';
let currentSettings = null;
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function authHeaders(extra = {}) {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { ...extra, 'X-API-Token': token } : extra;
}

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, { ...options, headers: authHeaders(options.headers || {}) });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ── status bar ───────────────────────────────────────────────────────
function updateStatus(health) {
  const dot = $('#statusDot');
  const text = $('#statusText');

  if (health.status === 'ok') {
    dot.className = 'dot dot-ok';
    text.textContent = `Camera Vision — pipeline: ${health.pipeline?.enabled ? 'on' : 'off'} — yolo: ${health.yolo?.backend || '—'}`;
  } else {
    dot.className = 'dot dot-err';
    text.textContent = `Error: ${health.status}`;
  }

  // Pipeline status indicator
  const pipeStatus = $('#pipelineStatus');
  if (pipeStatus) {
    if (health.pipeline?.enabled) {
      pipeStatus.className = 'active';
      pipeStatus.textContent = 'pipeline on';
    } else {
      pipeStatus.className = '';
      pipeStatus.textContent = 'pipeline off';
    }
  }

  $('#refreshTime').textContent = new Date().toLocaleTimeString();
}

function renderServerStatus(data) {
  function fmtElapsed(isoTs) {
    if (!isoTs) return '—';
    const delta = (Date.now() - new Date(isoTs).getTime()) / 1000;
    if (delta < 60) return Math.floor(delta) + 's';
    if (delta < 3600) return Math.floor(delta / 60) + 'm';
    return Math.floor(delta / 3600) + 'h';
  }

  for (const [name, s] of Object.entries(data)) {
    const card = $('#gemmaCard');
    if (!card || name !== 'gemma') continue;

    if (s.status === 'ok') {
      card.className = 'server-card server-ok';
      card.textContent = name + ' ' + fmtElapsed(s.last_request_ts);
    } else {
      card.className = 'server-card server-err';
      card.textContent = name + ' ✗';
    }
  }
}

// ── event cards + canvas overlays ────────────────────────────────────
function drawBoxes(canvas, img, boxes) {
  if (!img.complete || !img.naturalWidth) {
    // Image not loaded yet — retry once
    img.onload = () => drawBoxes(canvas, img, boxes);
    return;
  }
  const rect = img.getBoundingClientRect();
  const displayW = rect.width;
  const displayH = rect.height;
  const scaleX = displayW / img.naturalWidth;
  const scaleY = displayH / img.naturalHeight;

  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = displayW + 'px';
  canvas.style.height = displayH + 'px';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, displayW, displayH);

  const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  boxes.forEach((b, i) => {
    if (!b.box || b.box.length < 4) return;
    const [x1, y1, x2, y2] = b.box;
    const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX, sy2 = y2 * scaleY;
    const w = sx2 - sx1, h = sy2 - sy1;

    const color = colors[i % colors.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(sx1, sy1, w, h);

    // Label background
    const label = `${b.label || ''} ${(b.score || 0).toFixed(2)}`;
    ctx.font = '12px Inter, system-ui, sans-serif';
    const textW = ctx.measureText(label).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(sx1, sy1 - 18, textW, 18);

    // Label text
    ctx.fillStyle = '#000';
    ctx.fillText(label, sx1 + 4, sy1 - 5);
  });
}

function scoreClass(score) {
  if (score >= 0.7) return 'high';
  if (score >= 0.4) return 'mid';
  return 'low';
}

// ── frame burst cycling ──────────────────────────────────────────────
function setupFrameCycling(card, frames, boxes) {
  const img = card.querySelector('.snapshot-img');
  const canvas = card.querySelector('.overlay-canvas');
  const indicator = card.querySelector('.frame-indicator');
  const prevBtn = card.querySelector('.frame-prev');
  const nextBtn = card.querySelector('.frame-next');
  const snapshotArea = card.querySelector('.event-snapshot');

  if (!frames || frames.length <= 1) {
    // Single frame — hide nav, no cycling
    prevBtn.style.display = 'none';
    nextBtn.style.display = 'none';
    indicator.style.display = 'none';
    return;
  }

  let currentIdx = 0;
  let autoTimer = null;
  const CYCLE_MS = 1800;

  // Build dot indicators
  indicator.style.display = '';
  indicator.innerHTML = '';
  for (let i = 0; i < frames.length; i++) {
    const dot = document.createElement('span');
    dot.className = 'frame-dot';
    dot.addEventListener('click', () => showFrame(i));
    indicator.appendChild(dot);
  }

  function fnameFor(framePath) {
    return framePath.split('/').pop();
  }

  function showFrame(idx) {
    currentIdx = idx;
    const fname = fnameFor(frames[idx]);
    img.src = `/api/snapshots/${encodeURIComponent(fname)}`;
    img.alt = `Frame ${idx + 1}/${frames.length}`;

    // Update dot indicators
    const dots = indicator.querySelectorAll('.frame-dot');
    dots.forEach((d, i) => d.classList.toggle('active', i === idx));

    // Show canvas + boxes only on frame 0 (where YOLO detection happened)
    canvas.style.display = idx === 0 ? '' : 'none';
    if (idx === 0 && boxes.length > 0) {
      drawBoxes(canvas, img, boxes);
    }

    // Update nav button enabled state
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === frames.length - 1;

    resetAutoTimer();
  }

  function resetAutoTimer() {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(() => {
      showFrame((currentIdx + 1) % frames.length);
    }, CYCLE_MS);
  }

  function stopAutoTimer() {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = null;
  }

  // Nav button handlers
  prevBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (currentIdx > 0) showFrame(currentIdx - 1);
  });
  nextBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (currentIdx < frames.length - 1) showFrame(currentIdx + 1);
  });

  // Pause on hover
  snapshotArea.addEventListener('mouseenter', () => {
    stopAutoTimer();
  });
  snapshotArea.addEventListener('mouseleave', () => {
    resetAutoTimer();
  });

  // Keyboard nav
  snapshotArea.setAttribute('tabindex', '0');
  snapshotArea.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); if (currentIdx > 0) showFrame(currentIdx - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); if (currentIdx < frames.length - 1) showFrame(currentIdx + 1); }
  });

  // Start with first frame
  showFrame(0);

  // Draw boxes on first frame after image loads
  img.onload = () => { if (currentIdx === 0) drawBoxes(canvas, img, boxes); };
  if (img.complete && currentIdx === 0) drawBoxes(canvas, img, boxes);
}

function renderEvents(events) {
  const list = $('#eventList');
  const template = $('#eventCardTemplate');
  list.innerHTML = '';

  if (!events.length) {
    list.innerHTML = '<p style="color:#64748b">No detections yet. The pipeline polls automatically — events will appear here when detected.</p>';
    $('#eventCount').textContent = '';
    return;
  }

  $('#eventCount').textContent = `(${events.length})`;

  for (const evt of events) {
    const cardFragment = template.content.cloneNode(true);
    const card = cardFragment.querySelector('.event-card');

    // Collect frames: frame_burst from raw, fall back to single snapshot
    const frames = (evt.raw?.frame_burst && evt.raw.frame_burst.length > 0)
      ? evt.raw.frame_burst
      : [evt.snapshot_path];

    const img = card.querySelector('.snapshot-img');
    const canvas = card.querySelector('.overlay-canvas');
    const fname = frames[0].split('/').pop();
    img.src = `/api/snapshots/${encodeURIComponent(fname)}`;
    img.alt = `Snapshot #${evt.id}`;

    card.querySelector('.event-id').textContent = `#${evt.id}`;
    card.querySelector('.event-time').textContent = new Date(evt.created_at).toLocaleString();

    const detList = card.querySelector('.event-detections');
    if (evt.boxes && evt.boxes.length > 0) {
      evt.boxes.forEach((b) => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${b.label}</span><span class="det-score ${scoreClass(b.score)}">${(b.score || 0).toFixed(3)}</span>`;
        detList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'no detections';
      li.style.color = '#64748b';
      detList.appendChild(li);
    }

    // VLM classification rendering
    const vlm = evt.raw?.vlm;
    const eventMeta = card.querySelector('.event-meta');
    if (vlm && !vlm.error) {
      const vlmDiv = document.createElement('div');
      vlmDiv.className = 'vlm-classification';

      const suspClass = vlm.suspicious ? 'suspicious' : 'normal';
      const suspLabel = vlm.suspicious ? '⚠ SUSPICIOUS' : '✓ Normal';
      vlmDiv.innerHTML = '<span class="vlm-badge ' + suspClass + '">' + suspLabel + '</span>';

      if (vlm.activity) {
        const actSpan = document.createElement('span');
        actSpan.className = 'vlm-activity';
        actSpan.textContent = vlm.activity;
        vlmDiv.appendChild(actSpan);
      }
      if (vlm.person_present !== null && vlm.person_present !== undefined) {
        const personSpan = document.createElement('span');
        personSpan.className = 'vlm-person';
        personSpan.textContent = vlm.person_present ? '👤 Person' : '🚫 No person';
        vlmDiv.appendChild(personSpan);
      }
      if (vlm.carrying && vlm.carrying !== 'nothing' && vlm.carrying !== null) {
        const carrySpan = document.createElement('span');
        carrySpan.className = 'vlm-carrying';
        carrySpan.textContent = '📦 ' + vlm.carrying;
        vlmDiv.appendChild(carrySpan);
      }
      if (vlm.summary) {
        const summP = document.createElement('p');
        summP.className = 'vlm-summary';
        summP.textContent = vlm.summary;
        vlmDiv.appendChild(summP);
      }
      eventMeta.appendChild(vlmDiv);

      if (vlm.context_description) {
        const contextSection = document.createElement('section');
        contextSection.className = 'vlm-context-section';

        const contextTitle = document.createElement('h4');
        contextTitle.textContent = 'Context description';
        contextSection.appendChild(contextTitle);

        const contextText = document.createElement('p');
        contextText.textContent = vlm.context_description;
        contextSection.appendChild(contextText);

        eventMeta.appendChild(contextSection);
      }
    }
    if (vlm?.error) {
      const errDiv = document.createElement('div');
      errDiv.className = 'vlm-error';
      errDiv.textContent = '⚠ VLM: ' + vlm.error;
      eventMeta.appendChild(errDiv);
    }

    list.appendChild(card);

    // Set up frame cycling after DOM insertion
    setupFrameCycling(card, frames, evt.boxes || []);
  }
}

// ── settings form ────────────────────────────────────────────────────
function flattenSettings(obj, prefix = '') {
  const result = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      // Only recurse into known config sections (not prompts/zones arrays)
      if (['service', 'camera', 'notifications', 'face_recognition', 'pipeline'].includes(k)) {
        Object.assign(result, flattenSettings(v, key));
      }
    } else {
      result[key] = v;
    }
  }
  return result;
}

function populateSettingsForm(settings) {
  currentSettings = settings;
  const flat = flattenSettings(settings);
  for (const [key, value] of Object.entries(flat)) {
    const el = document.querySelector(`[name="${key}"]`);
    if (el) {
      if (el.type === 'checkbox') el.checked = !!value;
      else el.value = value ?? '';
    }
  }
  // Read-only classification prompt
  const promptEl = $('#classificationPrompt');
  if (promptEl && settings.vlm?.classification_prompt) {
    promptEl.textContent = settings.vlm.classification_prompt;
  }
}

function buildSettingsFromForm() {
  const flat = {};
  const els = $$('#settingsForm [name]');
  els.forEach((el) => {
    if (el.closest('template')) return; // skip template elements
    flat[el.name] = el.type === 'checkbox' ? el.checked : el.value;
  });

  // Rebuild nested structure from the last fetched settings so hidden config
  // sections (VLM, YOLO, filesystem paths, prompts, zones) are preserved.
  const settings = currentSettings ? JSON.parse(JSON.stringify(currentSettings)) : {
    service: {},
    camera: {},
    notifications: {},
    face_recognition: {},
    pipeline: {},
    zones: [],
  };

  for (const [key, value] of Object.entries(flat)) {
    const [section, field] = key.split('.');
    if (settings[section] !== undefined) {
      settings[section][field] = value;
    }
  }

  // Coerce numeric fields
  settings.camera.channel = parseInt(settings.camera.channel, 10) || 0;
  settings.camera.capture_timeout_seconds = parseInt(settings.camera.capture_timeout_seconds, 10) || 15;
  settings.pipeline.recent_context_seconds = Math.max(0, parseInt(settings.pipeline.recent_context_seconds, 10) || 0);

  return settings;
}

// ── refresh loop ─────────────────────────────────────────────────────
async function refresh() {
  const [health, settings, events, serverStatus] = await Promise.all([
    jsonFetch('/healthz'),
    jsonFetch('/api/settings'),
    jsonFetch('/api/events?limit=20'),
    jsonFetch('/api/servers/status').catch(() => null),
  ]);
  updateStatus(health);
  populateSettingsForm(settings);
  renderEvents(events);
  if (serverStatus) renderServerStatus(serverStatus);
}

// ── init ─────────────────────────────────────────────────────────────
$('#apiToken').value = localStorage.getItem(TOKEN_KEY) || '';

$('#saveToken').addEventListener('click', () => {
  const token = $('#apiToken').value.trim();
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
  refresh().catch(() => {});
});

$('#settingsForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const settings = buildSettingsFromForm();
    await jsonFetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    const msg = $('#settingsMsg');
    msg.textContent = '✓ Saved';
    msg.className = 'ok';
    setTimeout(() => { msg.textContent = ''; msg.className = ''; }, 2000);
    await refresh();
  } catch (err) {
    const msg = $('#settingsMsg');
    msg.textContent = '✗ ' + err.message;
    msg.className = 'err';
  }
});

$('#settingsToggle').addEventListener('click', () => {
  const body = $('#settingsBody');
  const toggle = $('#settingsToggle');
  if (body.hidden) {
    body.hidden = false;
    toggle.textContent = 'Settings ▴';
  } else {
    body.hidden = true;
    toggle.textContent = 'Settings ▾';
  }
});

// Auto-refresh
refresh().catch((err) => {
  $('#statusText').textContent = 'Error: ' + err.message;
  $('#statusDot').className = 'dot dot-err';
});

