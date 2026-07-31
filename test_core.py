import sys
import json
sys.stdout.reconfigure(encoding='utf-8')

from logger import get_logger
from syllable_counter import count_words_english, count_syllables_english, count_words_vietnamese, count_syllables_vietnamese, calculate_sps, evaluate_dub_timing
from exporter import export_to_csv, export_to_json, export_to_txt, export_to_xlsx_bytes, export_to_srt, export_to_vtt, prepare_export_rows
from diarizer import acoustic_fallback_diarization, assign_speaker_labels
from translator import translator_engine

logger = get_logger("TEST_CORE")

def test_all():
    logger.start("TEST_SUITE", "Starting unit tests for TTAL Transcriber core functions...")

    print("\n--- 1. Testing Syllable & Word Counting ---")
    eng_text = "Welcome to the video transcribing and dubbing application."
    eng_words = count_words_english(eng_text)
    eng_syls = count_syllables_english(eng_text)
    print(f"English: '{eng_text}'")
    print(f"  Words: {eng_words}, Syllables: {eng_syls}")
    assert eng_words == 8, f"Expected 8 words, got {eng_words}"
    assert eng_syls > 10, f"Expected >10 syllables, got {eng_syls}"

    vie_text = "Chào mừng bạn đến với ứng dụng chuyển đổi phụ đề và lồng tiếng."
    vie_words = count_words_vietnamese(vie_text)
    vie_syls = count_syllables_vietnamese(vie_text)
    print(f"Vietnamese: '{vie_text}'")
    print(f"  Words: {vie_words}, Syllables: {vie_syls}")
    assert vie_words == 14, f"Expected 14 words, got {vie_words}"
    assert vie_syls == 14, f"Expected 14 syllables, got {vie_syls}"

    print("\n--- 2. Testing Dubbing Timing (SPS) ---")
    duration = 3.0
    sps_eng = calculate_sps(eng_syls, duration)
    timing_eng = evaluate_dub_timing(sps_eng)
    print(f"Duration: {duration}s, Eng SPS: {sps_eng}, Status: {timing_eng['label']}")
    assert sps_eng > 0, "SPS calculation error"

    print("\n--- 3. Testing Local Acoustic Diarization Fallback ---")
    dummy_segments = [
        {"id": 1, "start": 0.0, "end": 2.5, "eng_script": "Looking sharp, by the way."},
        {"id": 2, "start": 2.8, "end": 4.5, "eng_script": "Not too bad yourself."},
        {"id": 3, "start": 4.8, "end": 7.0, "eng_script": "You're so out of place here."}
    ]
    diarized = acoustic_fallback_diarization(dummy_segments)
    for seg in diarized:
        print(f"  Segment {seg['id']}: Speaker = {seg.get('speaker_label')}")
        assert "speaker_label" in seg, "Diarization missing speaker_label"

    print("\n--- 4. Testing Translation Engine with Context ---")
    translated = translator_engine.translate_batch(
        dummy_segments,
        ollama_enabled=True,
        scene_context="A man and a woman meeting at an evening party."
    )
    for seg in translated:
        print(f"  Segment {seg['id']} [{seg.get('speaker_label')}]: ENG='{seg.get('eng_script')}' -> VIE='{seg.get('vie_script')}'")
        assert "vie_script" in seg, "Translation missing vie_script"

    print("\n--- 5. Testing Exporter Formats (CSV, JSON, TXT, SRT, VTT, XLSX) ---")
    export_sample = [
        {
            "id": 1,
            "start": 1.5,
            "end": 4.2,
            "time_start": "00:00:01.500",
            "time_end": "00:00:04.200",
            "duration": 2.7,
            "speaker_label": "SPEAKER_01",
            "eng_script": "Hello and welcome to our channel.",
            "word_count": 6,
            "syllable_count": 9,
            "vie_script": "Xin chào và chào mừng đến kênh của chúng tôi.",
            "vie_syllable_count": 10
        }
    ]

    csv_out = export_to_csv(export_sample, include_vietnamese=True)
    assert "time_start,time_end,duration,speaker_label,eng_script,word-count,syllable count,vie_script" in csv_out, "CSV Header Mismatch!"

    json_out = export_to_json(export_sample, include_vietnamese=True)
    parsed_json = json.loads(json_out)
    assert "word-count" in parsed_json[0], "JSON missing 'word-count' key!"

    txt_out = export_to_txt(export_sample, include_vietnamese=True)
    assert "[00:00:01.500 --> 00:00:04.200] SPEAKER_01" in txt_out, "TXT format mismatch"

    srt_out = export_to_srt(export_sample, include_vietnamese=True)
    assert "00:00:01,500 --> 00:00:04,200" in srt_out, "SRT format mismatch"

    vtt_out = export_to_vtt(export_sample, include_vietnamese=True)
    assert "WEBVTT - TTAL Video Dubbing Subtitles" in vtt_out, "VTT header mismatch"

    xlsx_bytes = export_to_xlsx_bytes(export_sample, include_vietnamese=True)
    assert len(xlsx_bytes) > 100, "XLSX export failed"

    logger.complete("TEST_SUITE", "ALL CORE TESTS PASSED SUCCESSFULLY! ✅")
    print("\nALL CORE TESTS PASSED SUCCESSFULLY! ✅")

if __name__ == "__main__":
    test_all()
