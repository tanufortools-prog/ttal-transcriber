import os
import re
import numpy as np
from typing import List, Dict, Any, Optional
from logger import get_logger

logger = get_logger("DIARIZATION")

# Monkeypatch huggingface_hub.hf_hub_download to strip deprecated 'use_auth_token'
# which causes pyannote.audio 3.3.2 to crash on newer huggingface_hub versions.
try:
    import huggingface_hub
    _orig_hf_hub_download = huggingface_hub.hf_hub_download

    def _patched_hf_hub_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            token_val = kwargs.pop('use_auth_token')
            if 'token' not in kwargs and isinstance(token_val, str):
                kwargs['token'] = token_val
        return _orig_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _patched_hf_hub_download
except Exception as patch_err:
    logger.warning("MONKEYPATCH_WARN", f"Could not patch hf_hub_download: {patch_err}")

def run_pyannote_diarization(audio_path: str, hf_token: str) -> List[Dict[str, Any]]:
    """
    Runs official Pyannote 3.1 Speaker Diarization using user's HuggingFace Token.
    Returns list of speaker turn time windows: [{'start': 0.0, 'end': 2.8, 'speaker': 'SPEAKER_01'}, ...]
    """
    logger.start("PYANNOTE_INIT", f"Audio Path: {audio_path}")
    try:
        import torch
        from huggingface_hub import login
        from pyannote.audio import Pipeline
        
        token = hf_token.strip()
        logger.progress("PYANNOTE_LOGIN", "Authenticating with HuggingFace Hub...")
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            logger.warning("HF_LOGIN_WARN", f"HF Hub login warning: {e}")
        
        logger.progress("PYANNOTE_LOAD", "Loading Pyannote 3.1 Pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        logger.progress("PYANNOTE_INFERENCE", "Running Pyannote neural diarization on audio track...")
        diarization = pipeline(audio_path)
        
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            spk_name = f"SPEAKER_{int(speaker.split('_')[-1]):02d}" if '_' in str(speaker) and str(speaker).split('_')[-1].isdigit() else str(speaker).upper()
            turns.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": spk_name
            })
            
        logger.complete("PYANNOTE_SUCCESS", f"Detected {len(turns)} speaker turns")
        return turns
    except Exception as e:
        logger.error("PYANNOTE_FAILURE", e, "Falling back to Local Acoustic Feature Diarization")
        return []

def extract_mfcc_and_pitch_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts 13-band Mel-Filterbank Energies, MFCCs, Spectral Centroid, and Zero Crossing Rate
    for high-precision acoustic speaker clustering.
    """
    if len(y) < int(sr * 0.1):
        return np.zeros(28, dtype=np.float32)

    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    frame_len = int(sr * 0.04)
    hop_len = int(sr * 0.02)
    
    num_frames = max(1, (len(y_pre) - frame_len) // hop_len + 1)
    window = np.hanning(frame_len)
    
    fft_size = 512
    num_filters = 13
    mel_points = np.linspace(0, np.log1p(sr / 2.0), num_filters + 2)
    hz_points = np.expm1(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((num_filters, int(fft_size / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / max(1, f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / max(1, f_m_plus - f_m)

    features = []
    frequencies = np.linspace(0, sr / 2, int(fft_size / 2 + 1))
    
    for i in range(num_frames):
        start_idx = i * hop_len
        end_idx = start_idx + frame_len
        frame = y_pre[start_idx:end_idx]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
            
        w_frame = frame * window
        spectrum = np.abs(np.fft.rfft(w_frame, n=fft_size))
        power_spectrum = (spectrum ** 2) / fft_size
        
        mel_energies = np.dot(fbank, power_spectrum)
        log_mel_energies = np.log1p(mel_energies)
        
        spec_centroid = np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-8)
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        energy = np.log1p(np.sum(frame ** 2))
        
        feat = list(log_mel_energies) + [spec_centroid / 1000.0, zcr, energy]
        features.append(feat)

    feat_arr = np.array(features)
    mean_vec = np.mean(feat_arr, axis=0)
    std_vec = np.std(feat_arr, axis=0)
    return np.concatenate([mean_vec, std_vec])

def acoustic_fallback_diarization(segments: List[Dict[str, Any]], audio_path: Optional[str] = None, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Local MFCC + Spectral Acoustic Diarization.
    Extracts 28-dimensional Mel-Filterbank + Timbre vectors for each segment
    and performs cosine Agglomerative Clustering.
    """
    if not segments:
        return segments

    logger.start("LOCAL_ACOUSTIC_INIT", f"Segments: {len(segments)}, Audio Path: {audio_path}, Requested Speakers: {num_speakers}")

    if not audio_path or not os.path.exists(audio_path):
        logger.warning("AUDIO_MISSING", "Audio path invalid or missing. Using dialog turn fallback.")
        return _dialog_turn_fallback(segments)

    try:
        import soundfile as sf
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        feature_matrix = []
        valid_indices = []

        for idx, seg in enumerate(segments):
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", 0.0))
            
            start_sample = max(0, int(start_sec * sr))
            end_sample = min(len(audio_data), int(end_sec * sr))
            
            y_slice = audio_data[start_sample:end_sample]
            
            if len(y_slice) > int(sr * 0.1):
                feat = extract_mfcc_and_pitch_features(y_slice, sr)
                feature_matrix.append(feat)
                valid_indices.append(idx)

        if not feature_matrix:
            logger.warning("INSUFFICIENT_FEATURES", "Could not extract audio features. Using dialog fallback.")
            return _dialog_turn_fallback(segments)

        X = np.array(feature_matrix)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = num_speakers if num_speakers and num_speakers > 1 else (2 if len(segments) >= 2 else 1)
        k = min(k, len(feature_matrix))

        logger.progress("CLUSTERING", f"Clustering {len(feature_matrix)} MFCC timbre vectors into {k} speakers...")
        clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = clusterer.fit_predict(X_scaled)

        for pos, seg_idx in enumerate(valid_indices):
            spk_num = labels[pos] + 1
            segments[seg_idx]["speaker_label"] = f"SPEAKER_{spk_num:02d}"

        for seg in segments:
            if "speaker_label" not in seg:
                seg["speaker_label"] = "SPEAKER_01"

        segments = _smooth_speaker_labels(segments)
        logger.complete("LOCAL_ACOUSTIC_SUCCESS", f"Clustered into {k} speakers across {len(segments)} segments")
        return segments

    except Exception as e:
        logger.error("LOCAL_ACOUSTIC_ERROR", e, "Fallback to dialog turn heuristic")
        return _dialog_turn_fallback(segments)

def _dialog_turn_fallback(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dialog-turn acoustic heuristic if waveform reading fails."""
    current_speaker = 1
    prev_end = 0.0
    prev_text = ""

    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        text = seg.get("eng_script", seg.get("text", "")).strip()
        gap = start - prev_end
        
        is_response = bool(re.match(r"^(yeah|yes|no|well|sure|okay|hey|look|oh|that's|can't|too)\b", text, re.I))
        prev_question = prev_text.endswith("?")
        
        if idx > 0 and (prev_question or gap >= 0.5 or is_response):
            current_speaker = 2 if current_speaker == 1 else 1
            
        seg["speaker_label"] = f"SPEAKER_{current_speaker:02d}"
        prev_end = float(seg.get("end", start))
        prev_text = text

    return segments

def _smooth_speaker_labels(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Smoothes isolated 1-segment speaker label glitches [A, B, A] -> [A, A, A]."""
    if len(segments) < 3:
        return segments

    for i in range(1, len(segments) - 1):
        prev_spk = segments[i - 1].get("speaker_label")
        curr_spk = segments[i].get("speaker_label")
        next_spk = segments[i + 1].get("speaker_label")

        dur = float(segments[i].get("end", 0)) - float(segments[i].get("start", 0))
        if prev_spk == next_spk and curr_spk != prev_spk and dur < 1.2:
            segments[i]["speaker_label"] = prev_spk

    return segments

def assign_speaker_labels(segments: List[Dict[str, Any]], speaker_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assigns Pyannote speaker turns to transcript segments based on maximum time overlap."""
    if not speaker_turns:
        return segments

    logger.start("ASSIGN_PYANNOTE_TURNS", f"Mapping {len(speaker_turns)} turns to {len(segments)} segments")
    for seg in segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        
        best_speaker = "SPEAKER_01"
        max_overlap = -1.0
        
        for turn in speaker_turns:
            t_start = turn["start"]
            t_end = turn["end"]
            
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > max_overlap and overlap > 0.05:
                max_overlap = overlap
                best_speaker = turn["speaker"]
                
        seg["speaker_label"] = best_speaker
        
    logger.complete("ASSIGN_PYANNOTE_TURNS", f"Mapped speaker labels successfully")
    return segments
