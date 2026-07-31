document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileMeta = document.getElementById('fileMeta');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const removeFileBtn = document.getElementById('removeFileBtn');
  
  const mediaPlayer = document.getElementById('mediaPlayer');
  const mediaPlaceholder = document.getElementById('mediaPlaceholder');
  const activeScriptText = document.getElementById('activeScriptText');
  const sceneContextInput = document.getElementById('sceneContextInput');
  const autoAnalyzeBtn = document.getElementById('autoAnalyzeBtn');
  
  const transcribeBtn = document.getElementById('transcribeBtn');
  const translateBtn = document.getElementById('translateBtn');
  const exportMainBtn = document.getElementById('exportMainBtn');
  const exportMenu = document.getElementById('exportMenu');
  
  const progressCard = document.getElementById('progressCard');
  const progressStepLabel = document.getElementById('progressStepLabel');
  const progressPercent = document.getElementById('progressPercent');
  const progressBarFill = document.getElementById('progressBarFill');
  
  const transcriptBody = document.getElementById('transcriptBody');
  const rowCountBadge = document.getElementById('rowCountBadge');
  const searchInput = document.getElementById('searchInput');
  const vieColHeader = document.getElementById('vieColHeader');
  const vieSylColHeader = document.getElementById('vieSylColHeader');
  
  const settingsBtn = document.getElementById('settingsBtn');
  const settingsModal = document.getElementById('settingsModal');
  const closeSettingsBtn = document.getElementById('closeSettingsBtn');
  const saveSettingsBtn = document.getElementById('saveSettingsBtn');

  const modelSizeSelect = document.getElementById('modelSizeSelect');
  const speakerModeSelect = document.getElementById('speakerModeSelect');
  const hfTokenInput = document.getElementById('hfTokenInput');
  const ollamaModelInput = document.getElementById('ollamaModelInput');

  let selectedFile = null;
  let currentSegments = [];
  let hasVietnamese = false;
  let detectedSpeakerMap = {};

  // File Upload
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFileSelected(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
  });
  removeFileBtn.addEventListener('click', (e) => { e.stopPropagation(); resetFileState(); });

  function handleFileSelected(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    fileMeta.classList.remove('hidden');
    dropZone.classList.add('hidden');
    transcribeBtn.disabled = false;

    const objectUrl = URL.createObjectURL(file);
    mediaPlayer.src = objectUrl;
    mediaPlayer.load();
    mediaPlaceholder.classList.add('hidden');
  }

  function resetFileState() {
    selectedFile = null;
    fileInput.value = '';
    fileMeta.classList.add('hidden');
    dropZone.classList.remove('hidden');
    transcribeBtn.disabled = true;
    translateBtn.disabled = true;
    exportMainBtn.disabled = true;
    mediaPlayer.pause();
    mediaPlayer.removeAttribute('src');
    mediaPlaceholder.classList.remove('hidden');
    currentSegments = [];
    sceneContextInput.value = '';
    detectedSpeakerMap = {};
    renderTable();
  }

  // Settings Modal
  settingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
  closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
  saveSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));

  // Export Menu
  exportMainBtn.addEventListener('click', (e) => { e.stopPropagation(); exportMenu.classList.toggle('hidden'); });
  document.addEventListener('click', () => exportMenu.classList.add('hidden'));
  exportMenu.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const fmt = btn.getAttribute('data-fmt');
      exportMenu.classList.add('hidden');
      await downloadExport(fmt);
    });
  });

  async function downloadExport(format) {
    if (currentSegments.length === 0) return;
    try {
      const response = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segments: currentSegments,
          format: format,
          include_vietnamese: hasVietnamese
        })
      });
      if (!response.ok) throw new Error('Export failed');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ttal_transcript_${hasVietnamese ? 'dub' : 'eng'}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    }
  }

  // Auto Analyze Context Button
  autoAnalyzeBtn.addEventListener('click', async () => {
    if (currentSegments.length === 0) {
      alert("Please transcribe a video first to analyze dialogue context.");
      return;
    }
    showProgress("AI is inferring scene context & character roles...", 50);
    const modelName = ollamaModelInput ? ollamaModelInput.value.trim() : "qwen2.5:3b";
    try {
      const res = await fetch('/api/analyze-context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segments: currentSegments,
          ollama_model: modelName
        })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.segments) {
          currentSegments = data.segments;
        }
        if (data.auto_scene_context) {
          sceneContextInput.value = data.auto_scene_context;
        }
        renderTable();
      }
      showProgress("Context auto-analyzed!", 100);
      setTimeout(() => hideProgress(), 800);
    } catch (e) {
      hideProgress();
      alert("Failed to analyze context: " + e.message);
    }
  });

  // Transcribe Action (Function 1)
  transcribeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    transcribeBtn.disabled = true;
    showProgress("Uploading media & extracting 16kHz audio track...", 15);

    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const hfToken = hfTokenInput ? hfTokenInput.value.trim() : '';
    if (hfToken) formData.append('hf_token', hfToken);

    const modelSize = modelSizeSelect ? modelSizeSelect.value : 'small.en';
    formData.append('model_size', modelSize);

    const spkMode = speakerModeSelect ? speakerModeSelect.value : 'auto';
    if (spkMode !== 'auto' && spkMode !== 'pyannote') {
      formData.append('num_speakers', parseInt(spkMode, 10));
    }

    try {
      showProgress(`Running Faster-Whisper (${modelSize})...`, 40);
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Transcription failed');
      }

      showProgress("Extracting dialogue & inferring character context...", 85);
      const data = await response.json();
      currentSegments = data.segments || [];
      hasVietnamese = false;

      if (data.auto_scene_context) {
        sceneContextInput.value = data.auto_scene_context;
      }

      showProgress("Complete!", 100);
      setTimeout(() => hideProgress(), 800);

      renderTable();
      translateBtn.disabled = false;
      exportMainBtn.disabled = false;

    } catch (err) {
      hideProgress();
      alert(`Transcription Error: ${err.message}`);
      transcribeBtn.disabled = false;
    }
  });

  // Translate Action (Function 2)
  translateBtn.addEventListener('click', async () => {
    if (currentSegments.length === 0) return;

    translateBtn.disabled = true;
    const modelName = ollamaModelInput ? ollamaModelInput.value.trim() : "qwen2.5:3b";
    const contextText = sceneContextInput ? sceneContextInput.value.trim() : "";

    showProgress(`Translating scene dialogue with Local LLM (${modelName})...`, 30);

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segments: currentSegments,
          ollama_enabled: true,
          ollama_url: "http://localhost:11434",
          ollama_model: modelName,
          scene_context: contextText
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Translation failed');
      }

      showProgress("Evaluating natural spoken dubbing register & timing fit...", 85);
      const data = await response.json();
      currentSegments = data.segments || [];
      hasVietnamese = true;

      showProgress("Dubbing translation complete!", 100);
      setTimeout(() => hideProgress(), 800);

      renderTable();
      translateBtn.disabled = false;

    } catch (err) {
      hideProgress();
      alert(`Translation Error: ${err.message}`);
      translateBtn.disabled = false;
    }
  });

  // Render Table
  function renderTable() {
    transcriptBody.innerHTML = '';
    rowCountBadge.textContent = `${currentSegments.length} Segments`;

    if (hasVietnamese) {
      vieColHeader.classList.remove('hidden');
      vieSylColHeader.classList.remove('hidden');
    } else {
      vieColHeader.classList.add('hidden');
      vieSylColHeader.classList.add('hidden');
    }

    if (currentSegments.length === 0) {
      transcriptBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="13">
            <div class="empty-state">
              <i class="ri-draft-line"></i>
              <p>No transcript generated yet. Upload a video and click <strong>1. Transcribe</strong> to begin.</p>
            </div>
          </td>
        </tr>`;
      return;
    }

    currentSegments.forEach((seg, index) => {
      const tr = document.createElement('tr');
      tr.setAttribute('data-id', seg.id || index + 1);
      tr.setAttribute('data-start', seg.start);
      tr.setAttribute('data-end', seg.end);

      const timingBadgeClass = seg.timing_badge_class || 'badge-neutral';
      const timingLabel = seg.timing_label || 'Optimal';
      const activeSpeaker = seg.speaker_label || 'SPEAKER_01';

      tr.innerHTML = `
        <td>${index + 1}</td>
        <td>
          <button class="btn-icon play-seg-btn" title="Seek & Play ${seg.time_start}">
            <i class="ri-play-fill"></i>
          </button>
        </td>
        <td><input class="cell-editable time-start-input" value="${seg.time_start}" /></td>
        <td><input class="cell-editable time-end-input" value="${seg.time_end}" /></td>
        <td class="duration-cell">${seg.duration}s</td>
        <td>
          <input class="cell-editable speaker-input" value="${activeSpeaker}" />
        </td>
        <td><textarea class="cell-editable eng-script-input" rows="2">${seg.eng_script}</textarea></td>
        <td class="eng-word-cnt">${seg.word_count}</td>
        <td class="eng-syl-cnt">${seg.syllable_count}</td>
        ${hasVietnamese ? `
          <td class="vie-col"><textarea class="cell-editable vie-script-input" rows="2">${seg.vie_script || ''}</textarea></td>
          <td class="vie-col vie-syl-cnt">${seg.vie_syllable_count || 0}</td>
        ` : ''}
        <td>
          <span class="badge ${timingBadgeClass}" title="${seg.timing_recommendation || ''}">
            ${timingLabel} (${hasVietnamese ? (seg.vie_sps || seg.sps) : seg.sps} SPS)
          </span>
        </td>
        <td>
          <div class="row-actions">
            <button class="btn-icon split-row-btn" title="Split Segment"><i class="ri-scissors-cut-line"></i></button>
            ${index < currentSegments.length - 1 ? `<button class="btn-icon merge-row-btn" title="Merge with Next"><i class="ri-merge-cells-horizontal"></i></button>` : ''}
          </div>
        </td>
      `;

      tr.querySelector('.play-seg-btn').addEventListener('click', () => {
        mediaPlayer.currentTime = parseFloat(seg.start);
        mediaPlayer.play();
        highlightActiveRow(tr, seg.eng_script);
      });

      const spkInput = tr.querySelector('.speaker-input');
      if (spkInput) {
        spkInput.addEventListener('change', (e) => {
          seg.speaker_label = e.target.value;
        });
      }

      tr.querySelector('.split-row-btn').addEventListener('click', () => splitRow(index));
      const mergeBtn = tr.querySelector('.merge-row-btn');
      if (mergeBtn) mergeBtn.addEventListener('click', () => mergeRow(index));

      const engInput = tr.querySelector('.eng-script-input');
      const vieInput = tr.querySelector('.vie-script-input');
      
      const onRowEdited = async () => {
        const newEng = engInput ? engInput.value : seg.eng_script;
        const newVie = vieInput ? vieInput.value : (seg.vie_script || '');
        
        seg.eng_script = newEng;
        if (hasVietnamese) seg.vie_script = newVie;

        try {
          const res = await fetch('/api/recalculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              start: seg.start,
              end: seg.end,
              eng_script: newEng,
              vie_script: newVie
            })
          });
          if (res.ok) {
            const rData = await res.json();
            seg.word_count = rData.eng_word_count;
            seg.syllable_count = rData.eng_syllable_count;
            seg.vie_syllable_count = rData.vie_syllable_count;
            seg.sps = rData.eng_sps;
            seg.vie_sps = rData.vie_sps;
            seg.timing_badge_class = rData.timing_badge_class;
            seg.timing_label = rData.timing_label;
            seg.timing_recommendation = rData.timing_recommendation;

            tr.querySelector('.eng-word-cnt').textContent = rData.eng_word_count;
            tr.querySelector('.eng-syl-cnt').textContent = rData.eng_syllable_count;
            if (hasVietnamese) {
              tr.querySelector('.vie-syl-cnt').textContent = rData.vie_syllable_count;
            }
            const badgeSpan = tr.querySelector('.badge');
            if (badgeSpan) {
              badgeSpan.className = `badge ${rData.timing_badge_class}`;
              badgeSpan.textContent = `${rData.timing_label} (${hasVietnamese ? rData.vie_sps : rData.eng_sps} SPS)`;
              badgeSpan.title = rData.timing_recommendation;
            }
          }
        } catch (e) {}
      };

      if (engInput) engInput.addEventListener('change', onRowEdited);
      if (vieInput) vieInput.addEventListener('change', onRowEdited);

      transcriptBody.appendChild(tr);
    });
  }

  function splitRow(index) {
    const seg = currentSegments[index];
    const words = seg.eng_script.split(' ');
    if (words.length <= 1) return;

    const mid = Math.floor(words.length / 2);
    const text1 = words.slice(0, mid).join(' ');
    const text2 = words.slice(mid).join(' ');
    const duration = parseFloat(seg.end) - parseFloat(seg.start);
    const midTime = roundVal(parseFloat(seg.start) + duration / 2);

    const seg1 = {
      ...seg,
      end: midTime,
      duration: roundVal(midTime - seg.start),
      eng_script: text1,
      vie_script: '',
      word_count: text1.split(' ').length,
      syllable_count: Math.ceil(seg.syllable_count / 2)
    };

    const seg2 = {
      ...seg,
      id: seg.id + 0.1,
      start: midTime,
      duration: roundVal(parseFloat(seg.end) - midTime),
      speaker_label: seg.speaker_label.includes('SPEAKER_01') ? 'SPEAKER_02' : 'SPEAKER_01',
      eng_script: text2,
      vie_script: '',
      word_count: text2.split(' ').length,
      syllable_count: Math.floor(seg.syllable_count / 2)
    };

    currentSegments.splice(index, 1, seg1, seg2);
    renderTable();
  }

  function mergeRow(index) {
    if (index >= currentSegments.length - 1) return;
    const seg1 = currentSegments[index];
    const seg2 = currentSegments[index + 1];

    const merged = {
      ...seg1,
      end: seg2.end,
      duration: roundVal(parseFloat(seg2.end) - parseFloat(seg1.start)),
      eng_script: `${seg1.eng_script} ${seg2.eng_script}`.trim(),
      vie_script: seg1.vie_script && seg2.vie_script ? `${seg1.vie_script} ${seg2.vie_script}` : (seg1.vie_script || seg2.vie_script || ''),
      word_count: seg1.word_count + seg2.word_count,
      syllable_count: seg1.syllable_count + seg2.syllable_count
    };

    currentSegments.splice(index, 2, merged);
    renderTable();
  }

  function roundVal(v) { return Math.round(v * 1000) / 1000; }

  function highlightActiveRow(tr, scriptText) {
    document.querySelectorAll('.ttal-table tr').forEach(r => r.classList.remove('active-playing'));
    tr.classList.add('active-playing');
    activeScriptText.textContent = scriptText;
  }

  mediaPlayer.addEventListener('timeupdate', () => {
    const currentTime = mediaPlayer.currentTime;
    const rows = transcriptBody.querySelectorAll('tr[data-start]');
    rows.forEach(tr => {
      const start = parseFloat(tr.getAttribute('data-start'));
      const end = parseFloat(tr.getAttribute('data-end'));
      if (currentTime >= start && currentTime <= end) {
        if (!tr.classList.contains('active-playing')) {
          const engText = tr.querySelector('.eng-script-input').value;
          highlightActiveRow(tr, engText);
        }
      }
    });
  });

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const rows = transcriptBody.querySelectorAll('tr[data-id]');
    rows.forEach(tr => {
      const text = tr.textContent.toLowerCase();
      if (text.includes(query)) tr.classList.remove('hidden');
      else tr.classList.add('hidden');
    });
  });

  function showProgress(stepLabel, percent) {
    progressCard.classList.remove('hidden');
    progressStepLabel.textContent = stepLabel;
    progressPercent.textContent = `${percent}%`;
    progressBarFill.style.width = `${percent}%`;
  }

  function hideProgress() {
    progressCard.classList.add('hidden');
    progressBarFill.style.width = '0%';
  }
});
