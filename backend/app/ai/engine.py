import re
from functools import lru_cache
from pathlib import Path
from app.utils.config import settings

@lru_cache(maxsize=1)
def whisper_model():
    if not settings.enable_ai_models: return None
    import whisper
    return whisper.load_model(settings.whisper_model)

@lru_cache(maxsize=1)
def summary_model():
    if not settings.enable_ai_models: return None
    from transformers import pipeline
    return pipeline("summarization", model=settings.summarizer_model)

def transcribe(path: Path) -> str:
    model = whisper_model()
    if model is None: raise RuntimeError("AI models are disabled")
    text = model.transcribe(str(path), fp16=False).get("text", "").strip()
    if not text: raise ValueError("No speech recognized")
    return text

def summarize(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(text.split()) < 80 or summary_model() is None: return " ".join(sentences[:5])[:1600]
    words = text.split(); parts = []
    for i in range(0, len(words), 700):
        out = summary_model()(" ".join(words[i:i+700]), max_length=180, min_length=35, do_sample=False)
        parts.append(out[0]["summary_text"])
    return " ".join(parts)

def extract_actions(text: str) -> list[dict]:
    trigger = re.compile(r"\b(will|must|should|needs? to|action item|follow up|todo|assigned to)\b", re.I)
    owner = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will|must|should|needs? to)\b")
    due = re.compile(r"\b(?:by|before|due|deadline:?|on)\s+((?:today|tomorrow|next week|end of (?:day|week|month)|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?))\b", re.I)
    result=[]
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s=s.strip(" -•\t")
        if len(s)<8 or not trigger.search(s): continue
        low=s.lower(); o=owner.search(s); d=due.search(s)
        priority="High" if any(x in low for x in ("urgent","critical","asap")) else "Low" if "low priority" in low else "Medium"
        result.append({"task":s[:1000],"owner":o.group(1) if o else None,"deadline":d.group(1) if d else None,"priority":priority,"completed":False})
    return result[:50]
