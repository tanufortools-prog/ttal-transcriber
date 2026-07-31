import sys
import json
import requests
sys.stdout.reconfigure(encoding='utf-8')

system_prompt = """You are an expert film dubbing scriptwriter translating English dialogue into natural spoken Vietnamese for voice actors.

Key Dubbing Rules:
1. Translate into natural, emotional spoken conversational Vietnamese used in dubbed movies.
2. Use consistent character pronouns (anh, cô, tôi, em, chị, chú) inferred from context.
3. Incorporate natural spoken particles (đấy, nhé, thôi, hả, chứ, à).
4. Maintain speech tempo and line duration without truncating sentence meaning.

Output JSON format ONLY:
[
  {"id": 1, "vie_script": "..."},
  {"id": 2, "vie_script": "..."}
]"""

prompt = """Scene Dialogue to Translate:
Line 1 [SPEAKER_01 - Female]: Looking sharp, by the way.
Line 2 [SPEAKER_02 - Male]: Not too bad yourself.
Line 3 [SPEAKER_01 - Female]: You're so out of place here.
Line 4 [SPEAKER_02 - Male]: Can't tell you what a relief it is to run into another English speaker. Even if you are American.
Line 5 [SPEAKER_01 - Female]: I don't have to blame my parents for that one."""

res = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:3b",
    "system": system_prompt,
    "prompt": prompt,
    "stream": False
}, timeout=30)

print("Ollama Response:")
print(res.json().get("response"))
