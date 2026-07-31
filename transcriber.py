import os
import sys
import re
import subprocess
from typing import List, Dict, Any, Callable, Optional
from syllable_counter import count_words_english, count_syllables_english, calculate_sps, evaluate_dub_timing
from exporter import format_timestamp
from logger import get_logger

logger = get_logger("TRANSCRIBE")

def ensure_ffmpeg_path() -> str:
    """Ensures ffmpeg executable is available in PATH or imageio_ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_bin)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ.get("PATH", "")
        return ffmpeg_bin
    except Exception:
        return "ffmpeg"

def extract_audio_from_video(video_path: str, output_wav_path: str) -> str:
    """Extracts 16kHz mono WAV audio from input video file."""
    logger.start("AUDIO_EXTRACTION", f"Input: {video_path} -> Output: {output_wav_path}")
    ffmpeg_cmd = ensure_ffmpeg_path()
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err_msg = result.stderr.decode('utf-8', errors='ignore')
        logger.error("AUDIO_EXTRACTION_FAILED", RuntimeError(err_msg))
        raise RuntimeError(f"Audio extraction failed: {err_msg}")
        
    logger.complete("AUDIO_EXTRACTION", f"WAV extracted successfully ({os.path.getsize(output_wav_path)} bytes)")
    return output_wav_path

class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "small.en"):
        self.model_size = model_size
        self.model = None

    def _load_model(self, requested_model_size: Optional[str] = None):
        target_size = requested_model_size or self.model_size
        if self.model is not None and self.model_size == target_size:
            return

        self.model_size = target_size
        from faster_whisper import WhisperModel
        
        device = "cuda"
        compute_type = "float32"
        
        logger.start("MODEL_LOAD", f"Loading Faster-Whisper '{target_size}' on {device} ({compute_type})...")
        try:
            self.model = WhisperModel(target_size, device=device, compute_type=compute_type)
            logger.complete("MODEL_LOAD", f"Loaded '{target_size}' on CUDA GPU")
        except Exception as e:
            logger.warning("CUDA_LOAD_FALLBACK", f"CUDA init failed ({e}), falling back to CPU int8...")
            self.model = WhisperModel(target_size, device="cpu", compute_type="int8")
            logger.complete("MODEL_LOAD", f"Loaded '{target_size}' on CPU int8")

    def transcribe(
        self,
        audio_path: str,
        model_size: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        self._load_model(model_size)
        
        logger.start("WHISPER_INFERENCE", f"Transcribing audio: {audio_path} using model {self.model_size}")
        if progress_callback:
            progress_callback(0.2, "Transcribing full audio dialogue sentences...")

        segments_raw, info = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                speech_pad_ms=200
            )
        )

        raw_list = list(segments_raw)
        logger.progress("SEGMENT_PARSING", f"Raw Whisper segments received: {len(raw_list)}")

        sentence_segments = []
        for seg in raw_list:
            if not hasattr(seg, 'words') or not seg.words:
                text = seg.text.strip()
                if text:
                    sentence_segments.append({
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "text": text
                    })
                continue

            words = seg.words
            current_sentence = []
            
            for i, w in enumerate(words):
                current_sentence.append(w)
                word_str = w.word.strip()
                
                is_last = (i == len(words) - 1)
                ends_sentence = bool(re.search(r'[.?!]', word_str))
                
                if not is_last:
                    next_w = words[i+1]
                    gap = next_w.start - w.end
                    pause_split = gap >= 0.45
                else:
                    pause_split = True

                if (ends_sentence and (is_last or pause_split)) or (pause_split and len(current_sentence) >= 3):
                    text_sentence = " ".join([item.word.strip() for item in current_sentence]).strip()
                    if text_sentence:
                        sentence_segments.append({
                            "start": round(current_sentence[0].start, 3),
                            "end": round(current_sentence[-1].end, 3),
                            "text": text_sentence
                        })
                    current_sentence = []

            if current_sentence:
                text_sentence = " ".join([item.word.strip() for item in current_sentence]).strip()
                if text_sentence:
                    sentence_segments.append({
                        "start": round(current_sentence[0].start, 3),
                        "end": round(current_sentence[-1].end, 3),
                        "text": text_sentence
                    })

        results = []
        for idx, item in enumerate(sentence_segments):
            start = item["start"]
            end = item["end"]
            eng_text = item["text"]

            duration = round(max(0.1, end - start), 3)
            word_cnt = count_words_english(eng_text)
            syl_cnt = count_syllables_english(eng_text)
            sps = calculate_sps(syl_cnt, duration)
            timing_info = evaluate_dub_timing(sps)

            results.append({
                "id": idx + 1,
                "start": start,
                "end": end,
                "time_start": format_timestamp(start),
                "time_end": format_timestamp(end),
                "duration": duration,
                "speaker_label": "SPEAKER_01",
                "eng_script": eng_text,
                "word_count": word_cnt,
                "syllable_count": syl_cnt,
                "sps": sps,
                "timing_status": timing_info["status"],
                "timing_label": timing_info["label"],
                "timing_badge_class": timing_info["badge_class"],
                "timing_recommendation": timing_info["recommendation"]
            })

        logger.complete("WHISPER_INFERENCE", f"Produced {len(results)} clean dialogue segments")
        return results

transcriber_engine = FasterWhisperTranscriber(model_size="small.en")
