/**
 * RAD·LAB Chest X-ray Finding Analyzer — Application Logic
 * Clinical Vision-Language Zero-Shot Inference Console
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements: Upload & Control ---
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const dropzoneIdle = document.getElementById('dropzoneIdle');
  const dropzoneActive = document.getElementById('dropzoneActive');
  const selectedFileName = document.getElementById('selectedFileName');
  const imagePreview = document.getElementById('imagePreview');
  const btnClearFile = document.getElementById('btnClearFile');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnLoadDemoSample = document.getElementById('btnLoadDemoSample');
  const errorAlert = document.getElementById('errorAlert');
  const errorMessage = document.getElementById('errorMessage');

  // --- DOM Elements: Output & Findings ---
  const countAboveThreshold = document.getElementById('countAboveThreshold');
  const countPromptsEvaluated = document.getElementById('countPromptsEvaluated');
  const currentThresholdDisplay = document.getElementById('currentThresholdDisplay');
  const findingsTbody = document.getElementById('findingsTbody');
  const prototypeConfidence = document.getElementById('prototypeConfidence');
  const modelCheckpointLabel = document.getElementById('modelCheckpointLabel');
  const displayRunId = document.getElementById('displayRunId');

  // --- DOM Elements: Explainability & PACS Viewer ---
  const selectFindingExplain = document.getElementById('selectFindingExplain');
  const patchGrid7x7 = document.getElementById('patchGrid7x7');
  const realOverlayContainer = document.getElementById('realOverlayContainer');
  const realOverlayImg = document.getElementById('realOverlayImg');
  const btnModeGrid = document.getElementById('btnModeGrid');
  const btnModeOverlay = document.getElementById('btnModeOverlay');
  const attentionLocationText = document.getElementById('attentionLocationText');
  const explainDisclaimer = document.getElementById('explainDisclaimer');

  // --- DOM Elements: Modals & Navigation ---
  const navPromptLibrary = document.getElementById('navPromptLibrary');
  const promptLibraryModal = document.getElementById('promptLibraryModal');
  const btnClosePromptModal = document.getElementById('btnClosePromptModal');
  const btnMethodologyModal = document.getElementById('btnMethodologyModal');
  const btnTopNewAnalysis = document.getElementById('btnTopNewAnalysis');
  const navNewAnalysis = document.getElementById('navNewAnalysis');

  // --- State Variables ---
  let selectedFile = null;
  let currentFindings = [];
  let currentThreshold = 0.62;
  let activeFindingKey = 'pleural_effusion';
  let viewerMode = 'grid'; // 'grid' or 'overlay'

  // Anatomic clinical focus maps for simulated 7x7 spatial patch attention
  const FINDING_CLINICAL_META = {
    'pleural_effusion': {
      sub: 'Left costophrenic angle',
      location: 'HIGH ATTENTION · LEFT BASE',
      centerR: 5, centerC: 4, spread: 1.2
    },
    'atelectasis': {
      sub: 'Basilar opacity',
      location: 'HIGH ATTENTION · RIGHT LOWER LOBE',
      centerR: 4, centerC: 2, spread: 1.4
    },
    'cardiomegaly': {
      sub: 'Cardiothoracic ratio',
      location: 'HIGH ATTENTION · CARDIAC SILHOUETTE',
      centerR: 4, centerC: 3, spread: 1.8
    },
    'pneumonia': {
      sub: 'Alveolar consolidation',
      location: 'HIGH ATTENTION · MID-LOWER FIELD',
      centerR: 3, centerC: 4, spread: 1.5
    },
    'pneumothorax': {
      sub: 'Apical visceral pleural line',
      location: 'HIGH ATTENTION · APICAL / LATERAL',
      centerR: 1, centerC: 5, spread: 1.3
    },
    'pulmonary_edema': {
      sub: 'Perihilar batwing haze',
      location: 'HIGH ATTENTION · BILATERAL PERIHILAR',
      centerR: 3, centerC: 3, spread: 2.1
    },
    'consolidation': {
      sub: 'Dense air-space opacification',
      location: 'HIGH ATTENTION · RIGHT MIDDLE LOBE',
      centerR: 3, centerC: 2, spread: 1.4
    },
    'normal': {
      sub: 'Clear bronchovascular markings',
      location: 'HOMOGENEOUS ATTENTION · LUNG FIELDS',
      centerR: 3, centerC: 3, spread: 3.5
    }
  };

  // --- Initialize Default UI State ---
  init7x7PatchGrid(activeFindingKey);
  checkBackendHealth();

  // --- Drag and Drop Listeners ---
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  dropzone.addEventListener('click', (e) => {
    if (e.target !== btnClearFile && !e.target.closest('#btnClearFile')) {
      if (!selectedFile) {
        fileInput.click();
      }
    }
  });

  btnClearFile.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileSelection();
  });

  // Load Demo Sample Button
  btnLoadDemoSample.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
      // Create synthetic chest radiograph for seamless immediate demo
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext('2d');

      // Gradient representing thoracic cavity
      const grad = ctx.createRadialGradient(128, 128, 20, 128, 128, 120);
      grad.addColorStop(0, '#101720');
      grad.addColorStop(0.7, '#243346');
      grad.addColorStop(1, '#080d14');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 256, 256);

      // Rib cage silhouettes
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 4;
      for (let y = 60; y <= 200; y += 28) {
        ctx.beginPath();
        ctx.arc(60, y, 40, -Math.PI / 4, Math.PI / 4);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(196, y, 40, (3 * Math.PI) / 4, (5 * Math.PI) / 4);
        ctx.stroke();
      }

      // Cardiac silhouette
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.beginPath();
      ctx.arc(140, 145, 38, 0, Math.PI * 2);
      ctx.fill();

      canvas.toBlob((blob) => {
        const demoFile = new File([blob], 'sample_chest_xray.png', { type: 'image/png' });
        handleFileSelection(demoFile);
      }, 'image/png');
    } catch (err) {
      console.warn('Demo generation fallback:', err);
    }
  });

  function handleFileSelection(file) {
    hideError();
    const validExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
      showError(`Unsupported file extension '${ext}'. Please upload PNG, JPG, JPEG, BMP, or TIFF.`);
      return;
    }

    selectedFile = file;
    selectedFileName.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzoneIdle.classList.add('hidden');
      dropzoneActive.classList.remove('hidden');
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    selectedFileName.textContent = '';
    dropzoneActive.classList.add('hidden');
    dropzoneIdle.classList.remove('hidden');
    analyzeBtn.disabled = true;
    hideError();
  }

  // --- API Analysis Call ---
  analyzeBtn.addEventListener('click', runAnalysis);
  btnTopNewAnalysis.addEventListener('click', () => {
    resetFileSelection();
    fileInput.click();
  });
  navNewAnalysis.addEventListener('click', () => {
    resetFileSelection();
    fileInput.click();
  });

  async function runAnalysis() {
    if (!selectedFile) return;

    hideError();
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze X-ray image.');
      }

      displayRunId.textContent = 'RUN_' + Math.floor(1000 + Math.random() * 9000);
      renderResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      showError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  }

  // --- Render Findings ---
  function renderResults(data) {
    currentFindings = data.findings || [];
    
    // Update prototype confidence
    if (typeof data.prototype_confidence === 'number') {
      prototypeConfidence.textContent = data.prototype_confidence.toFixed(4);
    }
    
    if (data.threshold_used) {
      currentThreshold = data.threshold_used;
      currentThresholdDisplay.textContent = Number(data.threshold_used).toFixed(2);
    }

    if (data.model && data.model.name) {
      modelCheckpointLabel.textContent = `${data.model.name} (${data.model.checkpoint || 'ViT-B/32'})`;
    }

    // Calculate count above threshold
    let aboveCount = 0;
    findingsTbody.innerHTML = '';

    currentFindings.forEach((item, index) => {
      const isCandidate = item.status === 'candidate' || item.score >= currentThreshold;
      if (isCandidate) aboveCount++;

      const tr = document.createElement('tr');
      tr.className = 'finding-row' + (index === 0 ? ' active-row' : '');
      tr.setAttribute('data-finding', item.finding);

      // Rank styling
      let rankClass = 'rank-muted';
      if (item.rank === 1) rankClass = 'rank-1';
      else if (item.rank === 2) rankClass = 'rank-2';
      else if (item.rank === 3) rankClass = 'rank-3';

      // Bar fill color
      let fillClass = 'fill-muted';
      if (item.rank === 1) fillClass = 'fill-coral';
      else if (item.rank === 2) fillClass = 'fill-amber';
      else if (item.rank === 3) fillClass = 'fill-blue';

      // Status badge
      let statusClass = 'status-unlikely';
      let statusText = 'Below threshold';
      if (isCandidate) {
        statusClass = 'status-candidate';
        statusText = 'Candidate';
      } else if (item.score >= currentThreshold * 0.8) {
        statusClass = 'status-review';
        statusText = 'Review';
      }

      // Metadata lookup
      const meta = FINDING_CLINICAL_META[item.finding] || {
        sub: 'Thoracic region',
        location: 'ATTENTION DETECTED'
      };

      const scorePercent = Math.min(Math.max((item.score / 1.0) * 100, 4), 100).toFixed(1);

      tr.innerHTML = `
        <td class="finding-cell">
          <span class="rank-num ${rankClass}">${String(item.rank).padStart(2, '0')}</span>
          <div class="finding-details">
            <span class="finding-title">${formatFindingTitle(item.finding)}</span>
            <span class="finding-sub">${meta.sub}</span>
          </div>
        </td>
        <td class="score-cell">
          <div class="score-bar-wrapper">
            <span class="score-val">${item.score.toFixed(2)}</span>
            <div class="score-track">
              <div class="score-fill ${fillClass}" style="width: ${scorePercent}%;"></div>
            </div>
          </div>
        </td>
        <td class="status-cell">
          <span class="status-badge ${statusClass}">${statusText}</span>
        </td>
      `;

      tr.addEventListener('click', () => {
        selectFindingForExplainability(item.finding, tr);
      });

      findingsTbody.appendChild(tr);
    });

    countAboveThreshold.textContent = String(aboveCount).padStart(2, '0');
    countPromptsEvaluated.textContent = String(currentFindings.length).padStart(2, '0');

    if (data.explainability && data.explainability.disclaimer) {
      explainDisclaimer.textContent = data.explainability.disclaimer;
    }

    // Automatically highlight top finding
    if (currentFindings.length > 0) {
      selectFindingForExplainability(currentFindings[0].finding, findingsTbody.children[0]);
    }
  }

  function formatFindingTitle(key) {
    return key
      .split('_')
      .map(w => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
  }

  // Row selection & synchronizing with Explainability viewer
  function selectFindingForExplainability(findingKey, rowEl) {
    activeFindingKey = findingKey;

    // Highlight row
    document.querySelectorAll('.finding-row').forEach(r => r.classList.remove('active-row'));
    if (rowEl) rowEl.classList.add('active-row');

    // Sync dropdown
    selectFindingExplain.value = findingKey;

    // Update viewer
    updateExplainabilityView(findingKey);
  }

  selectFindingExplain.addEventListener('change', (e) => {
    const findingKey = e.target.value;
    const matchingRow = document.querySelector(`.finding-row[data-finding="${findingKey}"]`);
    selectFindingForExplainability(findingKey, matchingRow);
  });

  // --- Explainability Viewer Logic ---
  function updateExplainabilityView(findingKey) {
    const meta = FINDING_CLINICAL_META[findingKey] || {
      location: 'HIGH ATTENTION · DETECTED REGION',
      centerR: 3, centerC: 3, spread: 1.5
    };
    attentionLocationText.textContent = meta.location;

    // Render 7x7 grid
    init7x7PatchGrid(findingKey);

    // If in overlay mode or file is present, fetch overlay
    if (viewerMode === 'overlay' && selectedFile) {
      fetchRealOverlay(findingKey);
    }
  }

  function init7x7PatchGrid(findingKey) {
    patchGrid7x7.innerHTML = '';
    const meta = FINDING_CLINICAL_META[findingKey] || {
      centerR: 3, centerC: 3, spread: 1.5
    };

    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const cell = document.createElement('div');
        cell.className = 'patch-cell';

        // Calculate distance to finding attention focus
        const dist = Math.hypot(r - meta.centerR, c - meta.centerC);
        const intensity = Math.max(0, 1 - (dist / (meta.spread * 2.2)));

        if (intensity > 0.75) {
          // Peak coral attention
          cell.style.background = '#ff5c72';
          cell.style.boxShadow = '0 0 10px rgba(255, 92, 114, 0.45)';
          cell.style.border = '1px solid #ff7b8d';
        } else if (intensity > 0.45) {
          // Medium warm amber-coral
          cell.style.background = '#b45348';
          cell.style.border = '1px solid rgba(255, 255, 255, 0.08)';
        } else if (intensity > 0.25) {
          // Subtle warm brown/terracotta
          cell.style.background = '#543632';
          cell.style.border = '1px solid rgba(255, 255, 255, 0.04)';
        } else {
          // Baseline dark slate patch
          cell.style.background = '#121c29';
          cell.style.border = '1px solid rgba(255, 255, 255, 0.03)';
        }

        cell.title = `Patch (${r}, ${c}) — Relative Attention: ${(intensity * 100).toFixed(0)}%`;
        patchGrid7x7.appendChild(cell);
      }
    }
  }

  // Mode Switchers
  btnModeGrid.addEventListener('click', () => {
    viewerMode = 'grid';
    btnModeGrid.classList.add('active');
    btnModeOverlay.classList.remove('active');
    patchGrid7x7.classList.remove('hidden');
    realOverlayContainer.classList.add('hidden');
  });

  btnModeOverlay.addEventListener('click', () => {
    viewerMode = 'overlay';
    btnModeOverlay.classList.add('active');
    btnModeGrid.classList.remove('active');
    patchGrid7x7.classList.add('hidden');
    realOverlayContainer.classList.remove('hidden');

    if (selectedFile) {
      fetchRealOverlay(activeFindingKey);
    } else {
      realOverlayContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Analyze an image to view Grad-CAM overlay</span>';
    }
  });

  async function fetchRealOverlay(findingKey) {
    if (!selectedFile) return;

    realOverlayContainer.innerHTML = '<div class="spinner" style="border-top-color: #2dd4bf;"></div>';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('finding', findingKey);

    try {
      const response = await fetch('/explain', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to generate overlay');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      realOverlayContainer.innerHTML = `<img src="${url}" alt="Attribution Heatmap Overlay for ${findingKey}">`;
    } catch (err) {
      realOverlayContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Overlay unavailable</span>';
    }
  }

  // --- Threshold Sweep Bar Interactions ---
  document.querySelectorAll('.sweep-bar-col').forEach(col => {
    col.addEventListener('click', () => {
      const t = parseFloat(col.getAttribute('data-thresh'));
      currentThreshold = t;
      currentThresholdDisplay.textContent = t.toFixed(2);
      
      // Update counts if findings loaded
      if (currentFindings.length > 0) {
        let count = currentFindings.filter(f => f.score >= t).length;
        countAboveThreshold.textContent = String(count).padStart(2, '0');
      }
    });
  });

  // --- Modal Open / Close ---
  navPromptLibrary.addEventListener('click', () => {
    promptLibraryModal.classList.remove('hidden');
  });
  btnMethodologyModal.addEventListener('click', () => {
    promptLibraryModal.classList.remove('hidden');
  });
  btnClosePromptModal.addEventListener('click', () => {
    promptLibraryModal.classList.add('hidden');
  });
  promptLibraryModal.addEventListener('click', (e) => {
    if (e.target === promptLibraryModal) {
      promptLibraryModal.classList.add('hidden');
    }
  });

  // --- Backend Health Check ---
  async function checkBackendHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const health = await res.json();
        if (health.model && health.model.name) {
          document.getElementById('sidebarModelTitle').textContent = health.model.name;
        }
      }
    } catch (err) {
      console.warn('Backend health check delayed:', err);
    }
  }

  // --- Utility Helpers ---
  function setLoading(isLoading) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      btnSpinner.classList.remove('hidden');
      analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
    } else {
      analyzeBtn.disabled = false;
      btnSpinner.classList.add('hidden');
      analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Chest X-Ray';
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove('hidden');
  }

  function hideError() {
    errorAlert.classList.add('hidden');
  }
});
