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
def extract_actions(text: str) -> list[dict]:
    speaker_commitment = re.compile(
        r"^(?P<speaker>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?):\s*"
        r"(?:(?:I|we)\s+(?:will|must|need to|have to|am going to)|(?:I'll|we'll))\s+(?P<task>.+)$",
        re.I,
    )
    named_commitment = re.compile(
        r"^(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
        r"(?:will|must|needs? to|has to|is going to|agreed to|committed to)\s+(?P<task>.+)$"
    )
    first_person_commitment = re.compile(
        r"^(?:(?:I|we)\s+(?:will|must|need to|have to|am going to)|(?:I'm going to|I'll|we'll))\s+(?P<task>.+)$",
        re.I,
    )
    informal_intention = re.compile(
        r"^(?:I'm going to|I'll)\s+(?P<task>.+)$",
        re.I,
    )
    team_requirement = re.compile(
        r"^(?:we|the team)\s+(?:need to|have to|must)\s+(?P<task>.+)$",
        re.I,
    )
    polite_request = re.compile(
        r"^(?:(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+)?"
        r"please\s+(?:do\s+)?(?P<task>.+)$",
        re.I,
    )
    labelled_action = re.compile(
        r"^(?:action item|todo|follow[- ]?up)\s*:?\s*(?:(?P<owner>[A-Z][a-z]+)\s+(?:will|to)\s+)?(?P<task>.+)$",
        re.I,
    )
    assignment = re.compile(
        r"^(?P<task>.+?)\s+(?:is\s+)?assigned to\s+(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$",
        re.I,
    )
    due = re.compile(
        r"\b(?:by|before|due(?:\s+on)?|deadline:?|on)\s+"
        r"((?:the\s+)?(?:end of (?:the\s+)?(?:day|week|month)|today|tomorrow|next week|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?))\b",
        re.I,
    )
    hypothetical = re.compile(
        r"\b(?:if|would|could|might|maybe|perhaps|probably|hopefully|for example|let's say)\b",
        re.I,
    )

    result = []
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip(" -•\t")
        sentence = re.sub(r"^(?:so|okay|all right|and)\b[,:\s-]*", "", sentence, flags=re.I)
        sentence = re.sub(r"^I\s+think\s+(?=I'm going to)", "", sentence, flags=re.I)
        sentence = re.sub(r"^I\s+will\s+I'll\s+", "I'll ", sentence, flags=re.I)
        please_match = re.search(r"\bplease\b", sentence, flags=re.I)
        if please_match:
            sentence = sentence[please_match.start():]
            sentence = re.sub(r"^(?:please\s+){2,}", "please ", sentence, flags=re.I)
        if len(sentence) < 8 or sentence.endswith("?"):
            continue

        owner = None
        task = None
        commitment_kind = None
        for kind, pattern in (
            ("speaker", speaker_commitment),
            ("named", named_commitment),
            ("request", polite_request),
            ("requirement", team_requirement),
            ("intention", informal_intention),
            ("first_person", first_person_commitment),
            ("labelled", labelled_action),
            ("assignment", assignment),
        ):
            match = pattern.match(sentence)
            if match:
                values = match.groupdict()
                owner = values.get("speaker") or values.get("owner")
                task = values.get("task")
                commitment_kind = kind
                break
        if not task:
            continue
        if commitment_kind == "request" and re.match(r"(?:don't|do not)\b", task, flags=re.I):
            continue
        if hypothetical.search(sentence) and commitment_kind != "request":
            continue
        if owner and owner.lower() in {"i", "we", "you", "he", "she", "it", "they", "this", "that", "someone", "everyone"}:
            continue

        deadline_match = due.search(task)
        deadline = deadline_match.group(1).strip() if deadline_match else None
        low = sentence.lower()
        priority = "High" if any(x in low for x in ("urgent", "critical", "asap")) else "Low" if "low priority" in low else "Medium"
        if commitment_kind == "first_person" and not deadline and priority == "Medium":
            continue
        task = due.sub("", task)
        task = re.sub(r"\b(?:and\s+)?(?:this is\s+)?(?:urgent|critical|asap|low priority)\b", "", task, flags=re.I)
        task = re.sub(r"\s+", " ", task).strip(" ,.;:-")
        if len(task.split()) < 2:
            continue

        key = re.sub(r"[^a-z0-9]+", " ", task.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        result.append({"task": task[:1000], "owner": owner, "deadline": deadline, "priority": priority, "completed": False})
    return result[:25]
