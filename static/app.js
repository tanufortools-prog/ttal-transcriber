document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const mediaContainer = document.getElementById('media-container');
  const videoPlayer = document.getElementById('video-player');
  const statusCard = document.getElementById('status-card');
  const transcribeBtn = document.getElementById('transcribe-btn');
  const exportBtn = document.getElementById('export-btn');
  const exportFormat = document.getElementById('export-format');
  const headerStatus = document.getElementById('header-status');
  const dialogueTbody = document.getElementById('dialogue-tbody');

  let currentSegments = [];

  // Trigger file dialog on click
  dropZone.addEventListener('click', () => fileInput.click());

  // Drag and Drop listeners
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    statusCard.textContent = `Uploading "${file.name}" & extracting 16kHz audio...`;
    headerStatus.textContent = 'Processing Audio...';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      videoPlayer.src = data.video_url;
      mediaContainer.style.display = 'block';
      
      statusCard.textContent = `Video loaded: ${file.name} | Audio Extracted: session_audio.wav`;
      headerStatus.textContent = 'Step 1 Complete: Ready for Step 2';
      transcribeBtn.disabled = false;

    } catch (err) {
      statusCard.textContent = `Error: ${err.message}`;
      headerStatus.textContent = 'Extraction Failed';
    }
  }

  // Step 2: Transcribe & Diarize Trigger
  transcribeBtn.addEventListener('click', async () => {
    statusCard.textContent = 'Running ASR & Speaker Diarization...';
    headerStatus.textContent = 'Transcribing Spoken Lines...';
    transcribeBtn.disabled = true;

    try {
      const response = await fetch('/api/transcribe', { method: 'POST' });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Transcription failed');
      }

      currentSegments = data.segments;
      renderDialogueTable(currentSegments);

      statusCard.textContent = `Transcription complete: ${currentSegments.length} spoken dialogue lines extracted.`;
      headerStatus.textContent = 'Step 2 Complete: Ready to Edit & Export';
      transcribeBtn.disabled = false;
      exportBtn.disabled = false;

    } catch (err) {
      statusCard.textContent = `Error: ${err.message}`;
      headerStatus.textContent = 'Transcription Failed';
      transcribeBtn.disabled = false;
    }
  });

  function renderDialogueTable(segments) {
    if (!segments || segments.length === 0) {
      dialogueTbody.innerHTML = `
        <tr>
          <td colspan="6">
            <div class="empty-state">No spoken lines detected.</div>
          </td>
        </tr>`;
      return;
    }

    dialogueTbody.innerHTML = segments.map((seg, idx) => `
      <tr data-index="${idx}">
        <td>
          <span class="speaker-badge editable-cell" contenteditable="true" data-field="speaker">${seg.speaker}</span>
        </td>
        <td class="editable-cell" contenteditable="true" data-field="start">${seg.start.toFixed(3)}</td>
        <td class="editable-cell" contenteditable="true" data-field="end">${seg.end.toFixed(3)}</td>
        <td data-field="duration">${seg.duration.toFixed(3)}s</td>
        <td class="editable-cell" contenteditable="true" data-field="text">${seg.text}</td>
        <td>
          <button class="action-btn delete-btn" data-index="${idx}">Delete</button>
        </td>
      </tr>
    `).join('');

    // Attach row click video seek sync
    dialogueTbody.querySelectorAll('tr').forEach(tr => {
      tr.addEventListener('click', (e) => {
        if (e.target.isContentEditable || e.target.classList.contains('action-btn')) return;
        const idx = parseInt(tr.dataset.index);
        const seg = currentSegments[idx];
        if (seg && videoPlayer) {
          videoPlayer.currentTime = seg.start;
          videoPlayer.play();
          
          dialogueTbody.querySelectorAll('tr').forEach(r => r.classList.remove('active-row'));
          tr.classList.add('active-row');
        }
      });
    });

    // Attach contenteditable listeners for live updates & duration recalculation
    dialogueTbody.querySelectorAll('.editable-cell').forEach(cell => {
      cell.addEventListener('blur', (e) => {
        const tr = e.target.closest('tr');
        const idx = parseInt(tr.dataset.index);
        const field = e.target.dataset.field;
        const value = e.target.innerText.trim();

        if (field === 'start' || field === 'end') {
          const numVal = parseFloat(value) || 0;
          currentSegments[idx][field] = numVal;
          
          // Recalculate duration
          const start = currentSegments[idx].start;
          const end = currentSegments[idx].end;
          const newDuration = Math.max(0, end - start);
          currentSegments[idx].duration = newDuration;

          const durationCell = tr.querySelector('[data-field="duration"]');
          if (durationCell) {
            durationCell.textContent = `${newDuration.toFixed(3)}s`;
          }
        } else if (field === 'speaker') {
          currentSegments[idx].speaker = value;
        } else if (field === 'text') {
          currentSegments[idx].text = value;
        }
      });
    });

    // Delete row listeners
    dialogueTbody.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.dataset.index);
        currentSegments.splice(idx, 1);
        renderDialogueTable(currentSegments);
      });
    });
  }

  // Export handler
  exportBtn.addEventListener('click', async () => {
    const fmt = exportFormat.value;
    try {
      const response = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: fmt, segments: currentSegments })
      });

      if (!response.ok) throw new Error('Export failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dubbing_transcript.${fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

    } catch (err) {
      alert(`Export Error: ${err.message}`);
    }
  });
});
