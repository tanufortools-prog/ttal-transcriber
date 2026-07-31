import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from audio_extractor import extract_audio
from transcriber import transcribe_audio
from diarizer import diarize_segments
from exporter import export_transcript

app = FastAPI(title="TTAL Localhost Dubbing Transcriber")

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

SESSION_DATA = {
    "segments": []
}

app.mount("/static", StaticFiles(directory="static"), name="static")

class DialogueSegment(BaseModel):
    start: float
    end: float
    duration: float
    text: str
    speaker: str

class ExportRequest(BaseModel):
    format: str
    segments: Optional[List[DialogueSegment]] = None

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(TEMP_DIR, "session_video.mp4")
    audio_path = os.path.join(TEMP_DIR, "session_audio.wav")
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        extract_audio(video_path, audio_path)
        SESSION_DATA["segments"] = []
        
        return {
            "status": "success",
            "message": "Video uploaded and audio extracted successfully.",
            "video_url": "/api/video",
            "audio_path": audio_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video")
async def get_video():
    video_path = os.path.join(TEMP_DIR, "session_video.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video file not found")

@app.post("/api/transcribe")
async def process_transcription():
    """
    Step 2: Combined ASR & Diarization Pass.
    Extracts spoken dialogue lines, identifies unique speakers, and calculates line duration.
    """
    audio_path = os.path.join(TEMP_DIR, "session_audio.wav")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="Audio file not found. Upload video first.")
        
    try:
        raw_segments = transcribe_audio(audio_path)
        diarized_segments = diarize_segments(raw_segments, audio_path)
        SESSION_DATA["segments"] = diarized_segments
        
        return {
            "status": "success",
            "count": len(diarized_segments),
            "segments": diarized_segments
        }
    except Exception as e:
        print(f"Transcription API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
async def export_data(req: ExportRequest):
    segments = req.segments if req.segments is not None else SESSION_DATA.get("segments", [])
    if not segments:
        raise HTTPException(status_code=400, detail="No segments available to export.")
        
    if req.segments:
        segments = [s.dict() for s in req.segments]
        
    try:
        content = export_transcript(segments, req.format)
        
        media_types = {
            "srt": "application/x-subrip",
            "vtt": "text/vtt",
            "csv": "text/csv",
            "ttal": "application/json",
            "json": "application/json",
            "txt": "text/plain"
        }
        
        media_type = media_types.get(req.format.lower(), "text/plain")
        filename = f"dubbing_transcript.{req.format.lower()}"
        
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
