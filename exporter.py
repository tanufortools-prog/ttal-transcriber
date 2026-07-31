import io
import json
import csv
from typing import List, Dict, Any
import pandas as pd
from logger import get_logger

logger = get_logger("EXPORTER")

def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm string."""
    millis = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds into SRT HH:MM:SS,mmm string."""
    millis = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def prepare_export_rows(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> List[Dict[str, Any]]:
    """
    Transforms segments list into exact required TTAL dict columns:
    time_start, time_end, duration, speaker_label, eng_script, word-count, syllable count, [vie_script]
    """
    formatted = []
    for seg in segments:
        duration = round(float(seg.get("end", 0)) - float(seg.get("start", 0)), 3)
        row = {
            "time_start": seg.get("time_start", format_timestamp(float(seg.get("start", 0)))),
            "time_end": seg.get("time_end", format_timestamp(float(seg.get("end", 0)))),
            "duration": duration,
            "speaker_label": seg.get("speaker_label", "SPEAKER_01"),
            "eng_script": seg.get("eng_script", seg.get("text", "")),
            "word-count": seg.get("word_count", 0),
            "syllable count": seg.get("syllable_count", 0),
        }
        if include_vietnamese or "vie_script" in seg:
            row["vie_script"] = seg.get("vie_script", "")
        formatted.append(row)
    return formatted

def export_to_csv(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> str:
    """Generates CSV string content."""
    logger.start("EXPORT_CSV", f"Segments: {len(segments)}, Include Vietnamese: {include_vietnamese}")
    rows = prepare_export_rows(segments, include_vietnamese)
    if not rows:
        return ""
    
    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    logger.complete("EXPORT_CSV", "CSV generated successfully")
    return output.getvalue()

def export_to_json(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> str:
    """Generates formatted JSON string content."""
    logger.start("EXPORT_JSON", f"Segments: {len(segments)}")
    rows = prepare_export_rows(segments, include_vietnamese)
    logger.complete("EXPORT_JSON", "JSON generated successfully")
    return json.dumps(rows, indent=2, ensure_ascii=False)

def export_to_txt(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> str:
    """Generates clean human-readable text script with timestamps and speaker labels."""
    logger.start("EXPORT_TXT", f"Segments: {len(segments)}")
    lines = []
    rows = prepare_export_rows(segments, include_vietnamese)
    for r in rows:
        header = f"[{r['time_start']} --> {r['time_end']}] {r['speaker_label']}"
        lines.append(header)
        lines.append(f"ENG: {r['eng_script']} (words: {r['word-count']}, syllables: {r['syllable count']})")
        if include_vietnamese and "vie_script" in r and r["vie_script"]:
            lines.append(f"VIE: {r['vie_script']}")
        lines.append("")  # Empty separator line
    logger.complete("EXPORT_TXT", "TXT script generated successfully")
    return "\n".join(lines)

def export_to_xlsx_bytes(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> bytes:
    """Generates Excel (.xlsx) file bytes."""
    logger.start("EXPORT_XLSX", f"Segments: {len(segments)}")
    rows = prepare_export_rows(segments, include_vietnamese)
    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Transcript')
    logger.complete("EXPORT_XLSX", "XLSX file generated successfully")
    return output.getvalue()

def export_to_srt(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> str:
    """Generates SubRip Subtitle (.srt) format."""
    logger.start("EXPORT_SRT", f"Segments: {len(segments)}")
    srt_blocks = []
    for idx, seg in enumerate(segments, 1):
        s_sec = float(seg.get("start", 0))
        e_sec = float(seg.get("end", 0))
        spk = seg.get("speaker_label", "SPEAKER_01")
        eng = seg.get("eng_script", seg.get("text", ""))
        vie = seg.get("vie_script", "") if include_vietnamese else ""

        t_start = format_timestamp_srt(s_sec)
        t_end = format_timestamp_srt(e_sec)

        block = f"{idx}\n{t_start} --> {t_end}\n[{spk}] {eng}"
        if vie:
            block += f"\n{vie}"
        srt_blocks.append(block)

    logger.complete("EXPORT_SRT", "SRT file generated successfully")
    return "\n\n".join(srt_blocks)

def export_to_vtt(segments: List[Dict[str, Any]], include_vietnamese: bool = True) -> str:
    """Generates WebVTT Subtitle (.vtt) format."""
    logger.start("EXPORT_VTT", f"Segments: {len(segments)}")
    lines = ["WEBVTT - TTAL Video Dubbing Subtitles\n"]
    for idx, seg in enumerate(segments, 1):
        s_sec = float(seg.get("start", 0))
        e_sec = float(seg.get("end", 0))
        spk = seg.get("speaker_label", "SPEAKER_01")
        eng = seg.get("eng_script", seg.get("text", ""))
        vie = seg.get("vie_script", "") if include_vietnamese else ""

        t_start = format_timestamp(s_sec)
        t_end = format_timestamp(e_sec)

        block = f"{idx}\n{t_start} --> {t_end}\n<v {spk}>{eng}"
        if vie:
            block += f"\n{vie}"
        lines.append(block)

    logger.complete("EXPORT_VTT", "VTT file generated successfully")
    return "\n\n".join(lines)
