import json
import csv
import io

def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_vtt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def export_transcript(segments: list, format_type: str) -> str:
    """
    Export dialogue segments to specified format: srt, vtt, csv, ttal, json, txt.
    """
    format_type = format_type.lower()
    
    if format_type == "srt":
        output = []
        for idx, seg in enumerate(segments, start=1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            speaker = seg.get("speaker", "SPEAKER_01")
            text = seg.get("text", "")
            output.append(f"{idx}\n{start} --> {end}\n[{speaker}] {text}\n")
        return "\n".join(output)

    elif format_type == "vtt":
        output = ["WEBVTT\n"]
        for idx, seg in enumerate(segments, start=1):
            start = format_vtt_time(seg["start"])
            end = format_vtt_time(seg["end"])
            speaker = seg.get("speaker", "SPEAKER_01")
            text = seg.get("text", "")
            output.append(f"{idx}\n{start} --> {end}\n<v {speaker}>{text}\n")
        return "\n".join(output)

    elif format_type == "csv":
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)
        writer.writerow(["Index", "Speaker", "Start", "End", "Duration", "Spoken Text"])
        for idx, seg in enumerate(segments, start=1):
            writer.writerow([
                idx,
                seg.get("speaker", "SPEAKER_01"),
                f"{seg['start']:.3f}",
                f"{seg['end']:.3f}",
                f"{seg['duration']:.3f}",
                seg.get("text", "")
            ])
        return output_buffer.getvalue()

    elif format_type == "ttal":
        ttal_data = {
            "version": "1.0",
            "type": "dubbing_transcript",
            "segments": [
                {
                    "id": idx,
                    "speaker": seg.get("speaker", "SPEAKER_01"),
                    "start_time": seg["start"],
                    "end_time": seg["end"],
                    "duration": seg["duration"],
                    "text": seg.get("text", "")
                }
                for idx, seg in enumerate(segments, start=1)
            ]
        }
        return json.dumps(ttal_data, indent=2)

    elif format_type == "json":
        return json.dumps(segments, indent=2)

    elif format_type == "txt":
        output = []
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER_01")
            text = seg.get("text", "")
            output.append(f"[{speaker}] {text}")
        return "\n".join(output)

    else:
        raise ValueError(f"Unsupported export format: {format_type}")
