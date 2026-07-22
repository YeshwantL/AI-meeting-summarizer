import re
from functools import lru_cache
from pathlib import Path
from app.utils.config import settings

@lru_cache(maxsize=1)
def whisper_model():
    if not settings.enable_ai_models: return None
    import torch
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(settings.whisper_model, device=device)

@lru_cache(maxsize=1)
def summary_model():
    if not settings.enable_ai_models: return None
    import torch
    from transformers import pipeline
    if torch.cuda.is_available():
        return pipeline(
            "summarization",
            model=settings.summarizer_model,
            device=0,
            model_kwargs={"torch_dtype": torch.float16},
        )
    return pipeline("summarization", model=settings.summarizer_model, device=-1)

def transcribe(path: Path) -> str:
    model = whisper_model()
    if model is None: raise RuntimeError("AI models are disabled")
    text = model.transcribe(str(path), fp16=model.device.type == "cuda").get("text", "").strip()
    if not text: raise ValueError("No speech recognized")
    return text

def summarize(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(text.split()) < 80 or summary_model() is None:
        selected = sentences[:2]
    else:
        words = text.split(); parts = []
        for i in range(0, len(words), 700):
            out = summary_model()(" ".join(words[i:i+700]), max_length=90, min_length=25, do_sample=False)
            parts.append(out[0]["summary_text"])
        combined = " ".join(parts)
        if len(combined.split()) > 140:
            combined = summary_model()(combined, max_length=110, min_length=35, do_sample=False)[0]["summary_text"]
        selected = re.split(r"(?<=[.!?])\s+", combined)[:5]

    points = []
    word_count = 0
    for sentence in selected:
        sentence = sentence.strip()
        if not sentence:
            continue
        remaining = 120 - word_count
        if remaining <= 0:
            break
        sentence_words = sentence.split()[:remaining]
        points.append("• " + " ".join(sentence_words))
        word_count += len(sentence_words)
    return "\n".join(points)
# Backward-compatible alias for callers that imported extraction from this module.
from app.ai.action_items import extract_action_items as extract_actions