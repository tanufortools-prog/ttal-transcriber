import os
import json
import re
import time
import subprocess
import requests
from typing import List, Dict, Any, Optional, Tuple
from syllable_counter import count_syllables_vietnamese, calculate_sps
from logger import get_logger

logger = get_logger("TRANSLATION")

UNIVERSAL_DUBBING_SYSTEM_PROMPT = """You are an elite audiovisual film dubbing director translating English movie dialogue into natural, emotionally compelling spoken Vietnamese for professional voice actors.

Core Audiovisual Film Dubbing Rules:
1. CONTEXTUAL PRONOUN CONSISTENCY (CRITICAL):
   - Examine the scene context and character descriptions carefully.
   - Maintain strict, consistent interpersonal pronouns (e.g., Victor [Male] = "anh", Nadine [Female] = "cô" or "em") across all lines of dialogue.
   - Never switch pronouns arbitrarily between consecutive lines for the same speaker!

2. AUTHENTIC SPOKEN MOVIE DIALOGUE:
   - Use natural spoken film Vietnamese register. Incorporate authentic conversational particles (đấy, nhé, thôi, hả, chứ, à, nào, cơ mà) to make dialogue feel alive in a movie dub.
   - Avoid mechanical, written prose or literal Google Translate phrasing.

3. ISOCHRONISM & TEMPO FIT:
   - Keep the target Vietnamese sentence length naturally speakable within the specified segment duration window.

FEW-SHOT EXAMPLES:

Example Input:
Scene Context: Victor (Male) and Nadine (Female) meet unexpectedly at an evening party.
Dialogue:
[
  {"id": 1, "speaker": "SPEAKER_01 (Nadine - Female)", "duration_sec": 1.8, "eng_script": "Looking sharp, by the way."},
  {"id": 2, "speaker": "SPEAKER_02 (Victor - Male)", "duration_sec": 1.6, "eng_script": "Not too bad yourself."},
  {"id": 3, "speaker": "SPEAKER_01 (Nadine - Female)", "duration_sec": 2.2, "eng_script": "You're so out of place here."}
]

Example Output:
[
  {"id": 1, "vie_script": "Trông anh bảnh bao đấy chứ."},
  {"id": 2, "vie_script": "Cô trông cũng không tệ đâu."},
  {"id": 3, "vie_script": "Anh hoàn toàn không thuộc về nơi này."}
]

Return ONLY a strictly valid JSON array of objects matching this format."""

CONTEXT_ANALYZER_SYSTEM_PROMPT = """You are an expert film script supervisor and character analyst.
Analyze the provided dialogue transcript and determine:
1. OVERALL SCENE SUMMARY: A 1-2 sentence description of the scene setting and situation.
2. SPEAKER IDENTITIES & GENDERS: Infer character names, roles, and genders for each speaker label (e.g., SPEAKER_01 = Female (Nadine), SPEAKER_02 = Male (Victor)).

Return ONLY a JSON object formatted as:
{
  "scene_summary": "Short 1-2 sentence scene summary...",
  "speakers": {
    "SPEAKER_01": "Nadine (Female)",
    "SPEAKER_02": "Victor (Male)"
  }
}
Do not include any markdown wrappers outside the JSON."""

def ensure_ollama_running(url: str = "http://localhost:11434") -> bool:
    """Ensures Ollama service is running on local port 11434. Auto-starts if needed."""
    try:
        res = requests.get(f"{url}/api/tags", timeout=3)
        if res.status_code == 200:
            logger.progress("OLLAMA_CHECK", f"Ollama daemon is active at {url}")
            return True
    except Exception:
        pass

    exe_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if os.path.exists(exe_path):
        try:
            logger.progress("OLLAMA_START", f"Auto-starting local Ollama daemon from {exe_path}...")
            subprocess.Popen([exe_path, "serve"], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            time.sleep(3)
            res = requests.get(f"{url}/api/tags", timeout=3)
            if res.status_code == 200:
                logger.complete("OLLAMA_START", "Ollama started successfully")
                return True
        except Exception as e:
            logger.warning("OLLAMA_START_FAIL", f"Failed to start Ollama daemon: {e}")

    logger.warning("OLLAMA_UNAVAILABLE", f"Ollama not reachable at {url}")
    return False

def clean_spoken_vietnamese(text: str) -> str:
    """Cleans up raw translation text, removing duplicate repeated words."""
    if not text:
        return ""
    
    cleaned = text.strip()
    words = cleaned.split()
    dedup = []
    for w in words:
        if not dedup or dedup[-1].lower() != w.lower():
            dedup.append(w)
    return " ".join(dedup)

def analyze_scene_and_speakers(
    segments: List[Dict[str, Any]],
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:3b"
) -> Tuple[str, Dict[str, str]]:
    """
    Analyzes English transcript dialogue with local LLM to automatically infer:
    1. Scene context & narrative summary.
    2. Speaker names, roles, and genders (e.g. SPEAKER_01 -> Nadine (Female)).
    """
    if not segments:
        return "", {}

    logger.start("AUTO_CONTEXT_ANALYSIS", f"Analyzing {len(segments)} segments with {ollama_model}...")

    if not ensure_ollama_running(ollama_url):
        logger.warning("AUTO_CONTEXT_FAIL", "Ollama not running. Returning default context.")
        return "Movie dialogue scene.", {}

    dialogue_block = []
    for seg in segments:
        dialogue_block.append(f"{seg.get('speaker_label', 'SPEAKER_01')}: {seg.get('eng_script', seg.get('text', ''))}")

    user_prompt = "Transcript Dialogue to Analyze:\n" + "\n".join(dialogue_block)

    try:
        payload = {
            "model": ollama_model,
            "system": CONTEXT_ANALYZER_SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        res = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=30)
        if res.status_code == 200:
            raw_response = res.json().get("response", "").strip()
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                summary = data.get("scene_summary", "").strip()
                speakers = data.get("speakers", {})
                logger.complete("AUTO_CONTEXT_ANALYSIS", f"Inferred Summary: '{summary}', Speakers: {speakers}")
                return summary, speakers
    except Exception as e:
        logger.error("AUTO_CONTEXT_ERROR", e)

    return "Movie scene dialogue.", {}

class UniversalOllamaDubTranslator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.default_model = "qwen2.5:3b"

    def get_available_models(self, url: Optional[str] = None) -> List[str]:
        target_url = url or self.ollama_url
        try:
            res = requests.get(f"{target_url}/api/tags", timeout=3)
            if res.status_code == 200:
                models = res.json().get("models", [])
                names = [m["name"] for m in models]
                logger.progress("MODELS_DETECTED", f"Ollama models available: {names}")
                return names
        except Exception as e:
            logger.warning("MODELS_DETECT_FAIL", f"Could not fetch Ollama models: {e}")
        return []

    def translate_scene_block_with_ollama(
        self,
        segments: List[Dict[str, Any]],
        model_name: str = "qwen2.5:3b",
        url: Optional[str] = None,
        scene_context: Optional[str] = None
    ) -> Dict[int, str]:
        target_url = url or self.ollama_url
        logger.start("OLLAMA_BLOCK_TRANSLATE", f"Model: {model_name}, Segments: {len(segments)}, Context: {scene_context or 'None'}")
        
        prompt_data = []
        for seg in segments:
            spk_label = seg.get("speaker_label", "SPEAKER_01")
            prompt_data.append({
                "id": seg.get("id"),
                "speaker": spk_label,
                "duration_sec": seg.get("duration", 2.0),
                "eng_script": seg.get("eng_script", seg.get("text", ""))
            })

        user_prompt = ""
        if scene_context and scene_context.strip():
            user_prompt += f"Scene Context & Character Descriptions:\n{scene_context.strip()}\n\n"
            
        user_prompt += f"Scene Dialogue Block to Translate into Spoken Vietnamese Dub Script:\n{json.dumps(prompt_data, indent=2, ensure_ascii=False)}"

        try:
            payload = {
                "model": model_name,
                "system": UNIVERSAL_DUBBING_SYSTEM_PROMPT,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.95
                }
            }
            res = requests.post(f"{target_url}/api/generate", json=payload, timeout=60)
            if res.status_code == 200:
                raw_response = res.json().get("response", "").strip()
                logger.debug("OLLAMA_RAW_RESPONSE", raw_response[:300] + "..." if len(raw_response) > 300 else raw_response)
                
                json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
                if json_match:
                    parsed_array = json.loads(json_match.group(0))
                    result_map = {}
                    for item in parsed_array:
                        sid = item.get("id")
                        script = item.get("vie_script", "").strip()
                        if sid is not None:
                            result_map[int(sid)] = script
                    logger.complete("OLLAMA_BLOCK_TRANSLATE", f"Successfully translated {len(result_map)} dialogue lines")
                    return result_map
                else:
                    logger.warning("OLLAMA_JSON_PARSING_FAIL", f"Could not find JSON array in response: {raw_response[:200]}")
            else:
                logger.warning("OLLAMA_HTTP_ERROR", f"HTTP {res.status_code}: {res.text}")
        except Exception as e:
            logger.error("OLLAMA_BLOCK_TRANSLATE_ERROR", e, f"Target URL: {target_url}")
            
        return {}

    def fallback_block_translation(self, segments: List[Dict[str, Any]]) -> Dict[int, str]:
        logger.start("FALLBACK_TRANSLATE", f"Translating {len(segments)} lines with GoogleTranslator Spoken Post-processor")
        result_map = {}
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='en', target='vi')
            
            for seg in segments:
                sid = seg.get("id")
                eng_text = seg.get("eng_script", seg.get("text", ""))
                if eng_text:
                    raw_vi = translator.translate(eng_text)
                    spoken_vi = clean_spoken_vietnamese(raw_vi)
                    result_map[sid] = spoken_vi
                else:
                    result_map[sid] = ""
            logger.complete("FALLBACK_TRANSLATE", f"Fallback translation complete for {len(result_map)} lines")
        except Exception as e:
            logger.error("FALLBACK_TRANSLATE_ERROR", e)
            for seg in segments:
                sid = seg.get("id")
                result_map[sid] = seg.get("eng_script", "")
                
        return result_map

    def translate_batch(
        self,
        segments: List[Dict[str, Any]],
        ollama_enabled: bool = True,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:3b",
        scene_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not segments:
            return segments

        logger.start("BATCH_TRANSLATE", f"Translating {len(segments)} segments. Ollama Enabled: {ollama_enabled}")
        target_url = ollama_url or self.ollama_url
        translated_map = {}

        ollama_up = ensure_ollama_running(target_url) if ollama_enabled else False

        if ollama_enabled and ollama_up:
            available = self.get_available_models(target_url)
            chosen_model = ollama_model
            if available:
                matched = [m for m in available if "qwen" in m.lower() or "llama" in m.lower()]
                chosen_model = matched[0] if matched else available[0]
            
            logger.progress("MODEL_CHOSEN", f"Selected local LLM model '{chosen_model}'")
            translated_map = self.translate_scene_block_with_ollama(segments, chosen_model, target_url, scene_context)

        if not translated_map:
            logger.warning("SWITCHING_FALLBACK", "Ollama translation unavailable or failed. Switching to DeepTranslator engine.")
            translated_map = self.fallback_block_translation(segments)

        for seg in segments:
            sid = seg.get("id")
            duration = float(seg.get("duration", 2.0))
            if duration <= 0:
                duration = 2.0
            
            vie_text = translated_map.get(sid, seg.get("eng_script", ""))
            vie_text = clean_spoken_vietnamese(vie_text)
            
            seg["vie_script"] = vie_text
            syl_count = count_syllables_vietnamese(vie_text)
            sps = calculate_sps(syl_count, duration)
            seg["vie_syllable_count"] = syl_count
            seg["vie_sps"] = sps
            
        logger.complete("BATCH_TRANSLATE", f"Batch translation complete for {len(segments)} segments")
        return segments

translator_engine = UniversalOllamaDubTranslator()
