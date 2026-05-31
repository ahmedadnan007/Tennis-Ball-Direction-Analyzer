const state = {
  jobId: null,
  pollTimer: null,
};

const el = {
  videoInput: document.getElementById('videoInput'),
  fileLabel: document.getElementById('fileLabel'),
  uploadBtn: document.getElementById('uploadBtn'),
  uploadSpinner: document.getElementById('uploadSpinner'),
  fileBox: document.getElementById('fileBox'),
  statusText: document.getElementById('statusText'),
  progressFill: document.getElementById('progressFill'),
  logBox: document.getElementById('logBox'),
  resultsEmpty: document.getElementById('resultsEmpty'),
  resultsArea: document.getElementById('resultsArea'),
  statsRow: document.getElementById('statsRow'),
  videoPlayer: document.getElementById('videoPlayer'),
  videoDownload: document.getElementById('videoDownload'),
  csvDownload: document.getElementById('csvDownload'),
  chartShotType: document.getElementById('chartShotType'),
  chartStroke: document.getElementById('chartStroke'),
  chartSpin: document.getElementById('chartSpin'),
  chartSpeed: document.getElementById('chartSpeed'),
  jobIdText: document.getElementById('jobId'),
  copyJobBtn: document.getElementById('copyJobBtn'),
};

function setStatus(text, progress = null) {
  el.statusText.textContent = text;
  if (progress !== null) {
    el.progressFill.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function setJobId(id) {
  el.jobIdText.textContent = id || '—';
}

function setLogs(lines) {
  if (!lines || !lines.length) {
    el.logBox.textContent = 'No logs yet.';
    return;
  }
  el.logBox.textContent = lines.join('\n');
  el.logBox.scrollTop = el.logBox.scrollHeight;
}

function showError(message) {
  setStatus('Error', 0);
  el.logBox.textContent = message;
}

function csvRowsToObjects(csvText) {
  const lines = csvText.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];

  const headers = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? '';
    });
    return row;
  });
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function renderStatCards(rows) {
  const count = rows.length;
  const avgSpeed = count ? rows.reduce((sum, row) => sum + toNumber(row.avg_speed_kmh), 0) / count : 0;
  const avgApex = count ? rows.reduce((sum, row) => sum + toNumber(row.apex_height_meters), 0) / count : 0;
  const avgConfidence = count ? rows.reduce((sum, row) => sum + toNumber(row.confidence), 0) / count : 0;

  el.statsRow.innerHTML = [
    { label: 'Shots', value: count.toString() },
    { label: 'Avg Speed', value: `${avgSpeed.toFixed(1)} km/h` },
    { label: 'Avg Apex', value: `${avgApex.toFixed(2)} m` },
    { label: 'Avg Confidence', value: avgConfidence.toFixed(2) },
  ].map((item) => `
    <div class="stat-card">
      <div class="label">${item.label}</div>
      <div class="value">${item.value}</div>
    </div>
  `).join('');
}

function countBy(rows, key) {
  const counts = {};
  rows.forEach((row) => {
    const value = (row[key] || 'unknown').toString() || 'unknown';
    counts[value] = (counts[value] || 0) + 1;
  });
  return counts;
}

function renderBarChart(containerId, title, rows, key, color) {
  if (!window.Plotly) return;
  const counts = countBy(rows, key);
  const labels = Object.keys(counts);
  const values = Object.values(counts);

  Plotly.newPlot(containerId, [{
    type: 'bar',
    x: labels,
    y: values,
    marker: { color },
  }], {
    title,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e8edf7' },
    margin: { l: 40, r: 20, t: 40, b: 60 },
    xaxis: { tickangle: -25, color: '#94a3b8' },
    yaxis: { color: '#94a3b8' },
  }, { responsive: true, displayModeBar: false });
}

function renderSpeedChart(rows) {
  if (!window.Plotly) return;
  const speeds = rows.map((row) => toNumber(row.avg_speed_kmh)).filter((value) => value > 0);

  Plotly.newPlot('chartSpeed', [{
    type: 'histogram',
    x: speeds,
    marker: { color: '#38bdf8' },
    nbinsx: 20,
  }], {
    title: 'Speed Distribution',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e8edf7' },
    margin: { l: 40, r: 20, t: 40, b: 40 },
    xaxis: { title: 'km/h', color: '#94a3b8' },
    yaxis: { title: 'Count', color: '#94a3b8' },
  }, { responsive: true, displayModeBar: false });
}

async function fetchCsvRows(jobId) {
  const response = await fetch(`/api/results/${jobId}/csv`);
  if (!response.ok) {
    return [];
  }
  const csvText = await response.text();
  return csvRowsToObjects(csvText);
}

async function renderResults(jobId) {
  const rows = await fetchCsvRows(jobId);
  renderStatCards(rows);

  const videoUrl = `/api/results/${jobId}/video`;
  const csvUrl = `/api/results/${jobId}/csv`;
  el.videoPlayer.src = videoUrl;
  el.videoPlayer.load();
  el.videoDownload.href = videoUrl;
  el.csvDownload.href = csvUrl;
  setJobId(jobId);

  if (!rows.length) {
    el.resultsEmpty.textContent = 'The analysis finished, but no shots were exported to CSV.';
    el.resultsEmpty.classList.remove('hidden');
    el.resultsArea.classList.remove('hidden');
    return;
  }

  el.resultsEmpty.classList.add('hidden');
  el.resultsArea.classList.remove('hidden');

  // Prefer friendly labels if present in CSV
  const shotTypeKey = rows.length && rows[0].type_label ? 'type_label' : 'shot_type';
  const directionKey = rows.length && rows[0].direction_label ? 'direction_label' : 'direction';
  const strokeKey = rows.length && rows[0].stroke_label ? 'stroke_label' : 'stroke';

  renderBarChart('chartShotType', 'Shot Types', rows, shotTypeKey, '#38bdf8');
  renderBarChart('chartStroke', 'Strokes', rows, strokeKey, '#22c55e');
  renderBarChart('chartSpin', 'Spin', rows, 'spin', '#f59e0b');
  renderSpeedChart(rows);
}

async function pollStatus(jobId) {
  const response = await fetch(`/api/status/${jobId}`);
  if (!response.ok) {
    throw new Error('Could not load job status');
  }

  const data = await response.json();
  setStatus(data.message || data.status, data.progress || 0);
  setLogs(data.logs || []);
  // update job id display if available
  setJobId(data.job_id || jobId);

  if (data.status === 'done') {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    await renderResults(jobId);
    setStatus('Done', 100);
  }

  if (data.status === 'error') {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    showError(data.message || 'Processing failed');
  }
}

async function startUpload() {
  const file = el.videoInput.files?.[0];
  if (!file) {
    alert('Please choose a video file first.');
    return;
  }

  state.jobId = null;
  el.resultsArea.classList.add('hidden');
  el.resultsEmpty.classList.remove('hidden');
  el.resultsEmpty.textContent = 'Results will appear after the job finishes.';
  el.statsRow.innerHTML = '';
  el.videoPlayer.removeAttribute('src');
  el.videoPlayer.load();
  setStatus('Uploading...', 5);
  setLogs(['Uploading file...']);
  el.uploadBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/analyze', { method: 'POST', body: formData });
    if (!response.ok) {
      throw new Error('Upload failed');
    }

    const data = await response.json();
    state.jobId = data.job_id;
    setStatus('Queued', 8);
    setLogs([`Job created: ${data.job_id}`, 'Waiting for the backend to start...']);
    setJobId(state.jobId);
    el.uploadSpinner.classList.remove('hidden');
    el.uploadBtn.classList.add('disabled');

    state.pollTimer = window.setInterval(() => {
      if (state.jobId) {
        pollStatus(state.jobId).catch((error) => showError(error.message));
      }
    }, 2000);

    await pollStatus(state.jobId);
  } catch (error) {
    showError(error.message || 'Something went wrong');
  } finally {
    el.uploadSpinner.classList.add('hidden');
    el.uploadBtn.classList.remove('disabled');
    el.uploadBtn.disabled = false;
  }
}

el.videoInput.addEventListener('change', () => {
  const file = el.videoInput.files?.[0];
  el.fileLabel.textContent = file ? file.name : 'Click to select a video file';
});

// Drag & drop support
if (el.fileBox) {
  el.fileBox.addEventListener('dragover', (e) => { e.preventDefault(); el.fileBox.classList.add('dragover'); });
  el.fileBox.addEventListener('dragleave', () => el.fileBox.classList.remove('dragover'));
  el.fileBox.addEventListener('drop', (e) => {
    e.preventDefault(); el.fileBox.classList.remove('dragover');
    const f = e.dataTransfer.files?.[0];
    if (f) {
      const dt = new DataTransfer(); dt.items.add(f); el.videoInput.files = dt.files; el.videoInput.dispatchEvent(new Event('change'));
    }
  });
}

// Copy Job ID
if (el.copyJobBtn) {
  el.copyJobBtn.addEventListener('click', async () => {
    const id = el.jobIdText.textContent || '';
    try { await navigator.clipboard.writeText(id); el.copyJobBtn.textContent = 'Copied'; setTimeout(()=> el.copyJobBtn.textContent = 'Copy',1200); } catch { alert('Copy failed'); }
  });
}

el.uploadBtn.addEventListener('click', startUpload);
