import re

def count_words_english(text: str) -> int:
    """Counts the number of words in an English text string."""
    if not text:
        return 0
    words = re.findall(r"\b[a-zA-Z0-9']+\b", text)
    return len(words)

def _count_word_syllables_en(word: str) -> int:
    """Rule-based syllable counter for a single English word."""
    word = word.lower().strip()
    if not word:
        return 0
    
    # Strip non-alphabet characters
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    
    # Exceptions/short-cuts for common patterns
    word = re.sub(r'(?:[^laeiouy]es|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    
    # Count vowel clusters
    vowel_clusters = re.findall(r'[aeiouy]{1,2}', word)
    count = len(vowel_clusters)
    
    # Fallback: at least 1 syllable for any non-empty word
    return max(1, count)

def count_syllables_english(text: str) -> int:
    """Counts total syllables in an English text string."""
    if not text:
        return 0
    
    # Try pyphen if installed, else fallback to rule-based algorithm
    try:
        import pyphen
        dic = pyphen.Pyphen(lang='en_US')
        words = re.findall(r"\b[a-zA-Z']+\b", text)
        total = 0
        for w in words:
            hyphenated = dic.inserted(w)
            parts = [p for p in hyphenated.split('-') if p]
            total += max(1, len(parts))
        return total
    except ImportError:
        words = re.findall(r"\b[a-zA-Z']+\b", text)
        return sum(_count_word_syllables_en(w) for w in words)

def count_words_vietnamese(text: str) -> int:
    """Counts space-separated words in Vietnamese text."""
    if not text:
        return 0
    tokens = text.strip().split()
    return len(tokens)

def count_syllables_vietnamese(text: str) -> int:
    """
    In Vietnamese, each written space-separated word corresponds to 1 spoken syllable.
    Example: 'xin chào bạn' -> 3 syllables.
    """
    return count_words_vietnamese(text)

def calculate_sps(syllable_count: int, duration_sec: float) -> float:
    """Calculates Syllables Per Second (SPS)."""
    if duration_sec <= 0:
        return 0.0
    return round(syllable_count / duration_sec, 2)

def evaluate_dub_timing(sps: float) -> dict:
    """
    Evaluates timing fit for dubbing:
    - optimal (2.0 - 4.5 SPS): Natural speech pace, ideal lip sync window.
    - overpacked (>4.5 SPS): Too fast, actor must rush, needs text condensation.
    - underpacked (<2.0 SPS): Too slow, speech gap, needs slight padding or rest.
    """
    if sps <= 0:
        return {"status": "empty", "label": "No Text", "badge_class": "badge-neutral", "recommendation": "Empty segment"}
    elif sps < 2.0:
        return {"status": "underpacked", "label": "Gap (Slow)", "badge_class": "badge-warning", "recommendation": "Underpacked: Speech ends early. Consider expanding dialogue or adding pause markers."}
    elif sps <= 4.5:
        return {"status": "optimal", "label": "Optimal Sync", "badge_class": "badge-success", "recommendation": "Optimal: Natural dubbing speech pace."}
    else:
        return {"status": "overpacked", "label": "Rush (Fast)", "badge_class": "badge-danger", "recommendation": "Overpacked: Too fast to speak in target duration. Condense text."}
