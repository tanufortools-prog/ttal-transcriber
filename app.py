import os
import re
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transcriber import transcriber_engine, extract_audio_from_video
from diarizer import run_pyannote_diarization, assign_speaker_labels, acoustic_fallback_diarization
from translator import translator_engine, analyze_scene_and_speakers
from syllable_counter import count_words_english, count_syllables_english, count_words_vietnamese, count_syllables_vietnamese, calculate_sps, evaluate_dub_timing
from exporter import export_to_csv, export_to_json, export_to_txt, export_to_xlsx_bytes, export_to_srt, export_to_vtt
from logger import get_logger

logger = get_logger("SERVER")

app = FastAPI(title="TTAL Video Transcriber & Dubbing Studio")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class RecalculateRequest(BaseModel):
    start: float
    end: float
    eng_script: str
    vie_script: Optional[str] = ""

class TranslateRequest(BaseModel):
    segments: List[Dict[str, Any]]
    ollama_enabled: Optional[bool] = True
    ollama_url: Optional[str] = "http://localhost:11434"
    ollama_model: Optional[str] = "qwen2.5:3b"
    scene_context: Optional[str] = ""

class AnalyzeContextRequest(BaseModel):
    segments: List[Dict[str, Any]]
    ollama_url: Optional[str] = "http://localhost:11434"
    ollama_model: Optional[str] = "qwen2.5:3b"

class DiarizeRequest(BaseModel):
    segments: List[Dict[str, Any]]
    audio_filename: Optional[str] = ""
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None

class ExportRequest(BaseModel):
    segments: List[Dict[str, Any]]
    format: str
    include_vietnamese: Optional[bool] = True

def apply_speaker_mappings_to_segments(segments: List[Dict[str, Any]], speaker_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Applies inferred character roles to segment speaker labels in Python."""
    if not speaker_map or not segments:
        return segments

    # Normalize map keys (e.g. handle SPEKER_02 or SPEAKER_02)
    norm_map = {}
    for k, v in speaker_map.items():
        clean_k = k.upper().replace("SPEKER", "SPEAKER")
        norm_map[clean_k] = v

    for seg in segments:
        orig = seg.get("speaker_label", "SPEAKER_01")
        # Extract base tag like SPEAKER_01 from SPEAKER_01 (Nadine)
        base_tag = orig.split()[0].upper().replace("SPEKER", "SPEAKER")
        
        if base_tag in norm_map:
            char_desc = norm_map[base_tag]
            if char_desc and not char_desc in orig:
                seg["speaker_label"] = f"{base_tag} ({char_desc})"
                
    return segments

@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/"):
        logger.start("API_REQUEST", f"{request.method} {path}")
    try:
        response = await call_next(request)
        if path.startswith("/api/"):
            logger.complete("API_REQUEST", f"{request.method} {path} -> Status {response.status_code}")
        return response
    except Exception as e:
        logger.error("API_REQUEST_ERROR", e, f"Method: {request.method}, Path: {path}")
        raise

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>TTAL Transcriber App Loaded</h1>")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "TTAL Transcriber & Dubbing Studio"}

@app.post("/api/transcribe")
async def transcribe_video(
    file: UploadFile = File(...),
    hf_token: Optional[str] = Form(None),
    model_size: Optional[str] = Form("small.en"),
    num_speakers: Optional[int] = Form(None)
):
    logger.start("TRANSCRIBE_ENDPOINT", f"Filename: {file.filename}, Model: {model_size}, HF Token Provided: {bool(hf_token and hf_token.strip())}")
    temp_video_path = os.path.join(TEMP_DIR, f"input_{file.filename}")
    temp_wav_path = os.path.join(TEMP_DIR, f"session_audio.wav")
    
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if temp_video_path.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv")):
            audio_file = extract_audio_from_video(temp_video_path, temp_wav_path)
        else:
            audio_file = temp_video_path

        segments = transcriber_engine.transcribe(audio_file, model_size=model_size)

        if hf_token and hf_token.strip():
            logger.progress("DIARIZATION_SELECTION", "Running Pyannote 3.1 Diarization Pipeline...")
            speaker_turns = run_pyannote_diarization(audio_file, hf_token)
            if speaker_turns:
                segments = assign_speaker_labels(segments, speaker_turns)
            else:
                logger.warning("PYANNOTE_EMPTY", "Pyannote returned empty turns. Switching to Local Acoustic Diarization.")
                segments = acoustic_fallback_diarization(segments, audio_file, num_speakers=num_speakers)
        else:
            logger.progress("DIARIZATION_SELECTION", "Running Local Feature Acoustic Diarization (No HF Token)...")
            segments = acoustic_fallback_diarization(segments, audio_file, num_speakers=num_speakers)

        # Automatic AI Scene & Speaker Context Extraction
        logger.progress("AUTO_CONTEXT", "Extracting automated scene summary & character roles from transcript...")
        auto_summary, speaker_map = analyze_scene_and_speakers(segments)
        
        # Apply inferred character roles directly to segment speaker labels in Python
        segments = apply_speaker_mappings_to_segments(segments, speaker_map)

        context_parts = []
        if auto_summary:
            context_parts.append(f"Scene Setting: {auto_summary}")
        if speaker_map:
            spk_desc = ", ".join([f"{k} = {v}" for k, v in speaker_map.items()])
            context_parts.append(f"Character Roles: {spk_desc}")
            
        auto_context_text = "\n".join(context_parts)

        logger.complete("TRANSCRIBE_ENDPOINT", f"Successfully transcribed {len(segments)} segments with auto-context")
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "total_segments": len(segments),
            "segments": segments,
            "auto_scene_context": auto_context_text,
            "speaker_mappings": speaker_map
        })

    except Exception as e:
        logger.error("TRANSCRIBE_ENDPOINT_FAIL", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            try: os.remove(temp_video_path)
            except Exception: pass

@app.post("/api/analyze-context")
def analyze_context(req: AnalyzeContextRequest):
    logger.start("ANALYZE_CONTEXT_ENDPOINT", f"Segments: {len(req.segments)}")
    try:
        summary, speaker_map = analyze_scene_and_speakers(
            req.segments,
            ollama_url=req.ollama_url or "http://localhost:11434",
            ollama_model=req.ollama_model or "qwen2.5:3b"
        )
        updated_segments = apply_speaker_mappings_to_segments(req.segments, speaker_map)
        
        context_parts = []
        if summary:
            context_parts.append(f"Scene Setting: {summary}")
        if speaker_map:
            spk_desc = ", ".join([f"{k} = {v}" for k, v in speaker_map.items()])
            context_parts.append(f"Character Roles: {spk_desc}")
            
        auto_context_text = "\n".join(context_parts)
        
        logger.complete("ANALYZE_CONTEXT_ENDPOINT", "Successfully analyzed scene context")
        return JSONResponse(content={
            "status": "success",
            "segments": updated_segments,
            "auto_scene_context": auto_context_text,
            "speaker_mappings": speaker_map
        })
    except Exception as e:
        logger.error("ANALYZE_CONTEXT_FAIL", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diarize")
def re_diarize_transcript(req: DiarizeRequest):
    logger.start("DIARIZE_ENDPOINT", f"Segments: {len(req.segments)}, Num Speakers: {req.num_speakers}")
    audio_path = os.path.join(TEMP_DIR, "session_audio.wav")
    
    try:
        segments = req.segments
        if req.hf_token and req.hf_token.strip():
            speaker_turns = run_pyannote_diarization(audio_path, req.hf_token)
            if speaker_turns:
                segments = assign_speaker_labels(segments, speaker_turns)
            else:
                segments = acoustic_fallback_diarization(segments, audio_path, num_speakers=req.num_speakers)
        else:
            segments = acoustic_fallback_diarization(segments, audio_path, num_speakers=req.num_speakers)

        logger.complete("DIARIZE_ENDPOINT", f"Re-diarized {len(segments)} segments")
        return JSONResponse(content={
            "status": "success",
            "segments": segments
        })
    except Exception as e:
        logger.error("DIARIZE_ENDPOINT_FAIL", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
def translate_transcript(req: TranslateRequest):
    logger.start("TRANSLATE_ENDPOINT", f"Segments: {len(req.segments)}, Model: {req.ollama_model}, Scene Context: {req.scene_context or 'None'}")
    try:
        updated_segments = translator_engine.translate_batch(
            req.segments,
            ollama_enabled=req.ollama_enabled,
            ollama_url=req.ollama_url or "http://localhost:11434",
            ollama_model=req.ollama_model or "qwen2.5:3b",
            scene_context=req.scene_context
        )
        
        for seg in updated_segments:
            duration = float(seg.get("duration", 1.0))
            vie_text = seg.get("vie_script", "")
            vie_syl = count_syllables_vietnamese(vie_text)
            vie_sps = calculate_sps(vie_syl, duration)
            timing_info = evaluate_dub_timing(vie_sps)
            
            seg["vie_syllable_count"] = vie_syl
            seg["vie_sps"] = vie_sps
            seg["timing_status"] = timing_info["status"]
            seg["timing_label"] = timing_info["label"]
            seg["timing_badge_class"] = timing_info["badge_class"]
            seg["timing_recommendation"] = timing_info["recommendation"]

        logger.complete("TRANSLATE_ENDPOINT", f"Successfully translated {len(updated_segments)} segments")
        return JSONResponse(content={
            "status": "success",
            "total_segments": len(updated_segments),
            "segments": updated_segments
        })
    except Exception as e:
        logger.error("TRANSLATE_ENDPOINT_FAIL", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recalculate")
def recalculate_row(req: RecalculateRequest):
    duration = round(max(0.1, req.end - req.start), 3)
    eng_words = count_words_english(req.eng_script)
    eng_syls = count_syllables_english(req.eng_script)
    eng_sps = calculate_sps(eng_syls, duration)

    vie_words = count_words_vietnamese(req.vie_script) if req.vie_script else 0
    vie_syls = count_syllables_vietnamese(req.vie_script) if req.vie_script else 0
    vie_sps = calculate_sps(vie_syls, duration)

    active_sps = vie_sps if req.vie_script else eng_sps
    timing_info = evaluate_dub_timing(active_sps)

    return {
        "duration": duration,
        "eng_word_count": eng_words,
        "eng_syllable_count": eng_syls,
        "eng_sps": eng_sps,
        "vie_word_count": vie_words,
        "vie_syllable_count": vie_syls,
        "vie_sps": vie_sps,
        "timing_status": timing_info["status"],
        "timing_label": timing_info["label"],
        "timing_badge_class": timing_info["badge_class"],
        "timing_recommendation": timing_info["recommendation"]
    }

@app.post("/api/export")
def export_file(req: ExportRequest):
    fmt = req.format.lower()
    include_vie = req.include_vietnamese
    filename_base = "ttal_transcript_dub" if include_vie else "ttal_transcript_eng"
    logger.start("EXPORT_ENDPOINT", f"Format: {fmt}, Include Vietnamese: {include_vie}")

    if fmt == "csv":
        content = export_to_csv(req.segments, include_vie)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.csv"}
        )
    elif fmt == "json":
        content = export_to_json(req.segments, include_vie)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.json"}
        )
    elif fmt == "txt":
        content = export_to_txt(req.segments, include_vie)
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.txt"}
        )
    elif fmt == "xlsx":
        xlsx_bytes = export_to_xlsx_bytes(req.segments, include_vietnamese=include_vie)
        return Response(
            content=xlsx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.xlsx"}
        )
    elif fmt == "srt":
        content = export_to_srt(req.segments, include_vietnamese=include_vie)
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.srt"}
        )
    elif fmt == "vtt":
        content = export_to_vtt(req.segments, include_vietnamese=include_vie)
        return Response(
            content=content,
            media_type="text/vtt",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.vtt"}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
