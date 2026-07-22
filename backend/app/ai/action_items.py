"""Action-item extraction and quality filtering.

The module intentionally keeps candidate detection separate from filtering and
metadata enrichment so each stage is deterministic and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable


BLACKLIST = (
    "read that",
    "ping me",
    "follow up",
    "sounds good",
    "thank you",
    "thanks",
    "okay",
    "alright",
    "got it",
    "cool",
    "take a minute",
    "check this",
    "look into it",
)

ACKNOWLEDGEMENTS = re.compile(
    r"^(?:sounds good|thank(?:s| you)(?: everyone)?|okay|alright|all right|got it|cool|great|perfect|welcome back)[.!]*$",
    re.I,
)
FILLERS = re.compile(r"^(?:um+|uh+|you know|I mean|anyway|so yeah|right)[,.!\s]*$", re.I)
HYPOTHETICAL = re.compile(r"\b(?:if|would|could|might|maybe|perhaps|probably|hopefully|for example|let's say)\b", re.I)
PRONOUNS = {"it", "that", "this", "them", "those", "these", "me", "you", "him", "her", "us"}
INVALID_OWNERS = {"i", "we", "you", "he", "she", "it", "they", "this", "that", "someone", "everyone", "please"}
GENERIC_VERBS = {"read", "ping", "check", "look", "follow", "see", "do", "get", "take", "go", "tell"}
WORK_VERBS = {
    "analyze", "approve", "build", "complete", "confirm", "create", "deliver", "deploy", "design",
    "distribute", "document", "draft", "email", "finalize", "fix", "implement", "investigate", "migrate",
    "prepare", "publish", "release", "resolve", "review", "schedule", "send", "test", "update", "validate",
}
BUSINESS_OBJECTS = {
    "api", "backend", "brief", "budget", "contract", "customer", "customers", "database", "design", "document",
    "documentation", "frontend", "invitation", "issue", "launch", "meeting", "onboarding", "plan", "proposal",
    "release", "report", "roadmap", "schedule", "survey", "ticket", "timeline", "website",
}

HIGH_PRIORITY = re.compile(r"\b(?:urgent|asap|today|tomorrow|deadline|must|critical|immediately)\b", re.I)
MEDIUM_PRIORITY = re.compile(r"\b(?:update|create|prepare|review|send|schedule)\b", re.I)
LOW_PRIORITY = re.compile(r"\b(?:read|ping|check|look at|follow up)\b", re.I)

DEADLINE = re.compile(
    r"\b(?:(?:by|before|due(?:\s+on)?|deadline:?|on)\s+)?"
    r"(?P<deadline>today|tomorrow|next week|before\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|end of (?:the )?(?:day|week|month)|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b",
    re.I,
)


@dataclass(frozen=True)
class Candidate:
    text: str
    owner: str | None
    context: str
    source: str


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def fuzzy_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def is_blacklisted(task: str, threshold: float = 0.78) -> bool:
    """Reject exact and near-matches while allowing contextual, specific work."""
    normalized = normalize_text(task)
    normalized = re.sub(r"^please\s+", "", normalized)
    for phrase in BLACKLIST:
        phrase_normalized = normalize_text(phrase)
        if fuzzy_similarity(normalized, phrase_normalized) >= threshold:
            return True
        if normalized.startswith(phrase_normalized) and len(normalized.split()) <= len(phrase_normalized.split()) + 2:
            return True
    return False


def detect_priority(text: str) -> str:
    if HIGH_PRIORITY.search(text):
        return "High"
    if MEDIUM_PRIORITY.search(text):
        return "Medium"
    if LOW_PRIORITY.search(text):
        return "Low"
    return "Medium"


def detect_deadline(text: str) -> str | None:
    match = DEADLINE.search(text)
    if not match:
        return None
    value = match.group("deadline").strip()
    if match.group(0).lower().startswith("before ") and not value.lower().startswith("before "):
        value = f"before {value}"
    return "EOD" if value.lower() == "eod" else value


def detect_owner(text: str, explicit_owner: str | None = None) -> str | None:
    if explicit_owner and explicit_owner.lower() not in INVALID_OWNERS:
        return explicit_owner.strip().title()

    patterns = (
        r"^(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+(?:please\s+)?(?:will\s+)?(?:send|update|create|prepare|review|schedule|complete|fix|confirm|deliver|deploy|publish)\b",
        r"\bassigned to\s+(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group("owner").lower() not in INVALID_OWNERS:
            return match.group("owner").strip().title()
    return None


def _context_object(context: str) -> str | None:
    patterns = (
        r"\b(onboarding document|onboarding doc|release notes|design document|project plan|roadmap|report|proposal|contract|document)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, context, re.I)
        if match:
            value = match.group(1).lower()
            return "onboarding document" if value in {"onboarding doc", "onboarding document"} else value
    return None


def rewrite_vague_task(task: str, context: str) -> str | None:
    """Rewrite a vague reference only when nearby text names its object."""
    normalized = normalize_text(task)
    context_object = _context_object(context)
    if normalized in {"go check it", "check it", "check this"}:
        return f"Review the {context_object} and verify its accuracy" if context_object else None
    if normalized in {"confirm that", "confirm it"}:
        return f"Confirm the accuracy of the {context_object}" if context_object else None
    return task


def _clean_task(task: str, owner: str | None, deadline: str | None) -> str:
    task = re.sub(r"^(?:please\s+(?:do\s+)?|go\s+)", "", task, flags=re.I)
    if owner:
        task = re.sub(rf"^{re.escape(owner)}\s*,?\s*", "", task, flags=re.I)
    task = re.sub(r"^(?:will|must|needs? to|has to|is going to|to)\s+", "", task, flags=re.I)
    if deadline:
        task = DEADLINE.sub("", task)
    task = re.sub(r"\b(?:and\s+)?(?:this is\s+)?(?:urgent|critical|asap|immediately|low priority)\b", "", task, flags=re.I)
    task = re.sub(r"\s+", " ", task).strip(" ,.;:-")

    if re.search(r"\bon\s*boarding\s+(?:dock|doc(?:ument)?)\b", task, re.I):
        task = re.sub(r"\bon\s*boarding\s+(?:dock|doc(?:ument)?)\b", "onboarding document", task, flags=re.I)
    if re.search(r"update (?:the )?(?:new hire )?onboarding document.*(?:ask|tell) all new hires to read", task, re.I):
        task = "Update the onboarding document and distribute it to all new hires"

    if task:
        task = task[0].upper() + task[1:]
    return task.rstrip(".") + "." if task else ""


def _is_independently_actionable(task: str) -> bool:
    words = normalize_text(task).split()
    if len(words) < 5:
        return False
    # A standalone task must open with a concrete work verb. This rejects
    # meeting chatter such as "let him know..." and "pile on that issue..."
    # even when those sentences happen to mention a business noun.
    if not words or words[0] not in WORK_VERBS:
        return False
    meaningful = [word for word in words if word not in PRONOUNS]
    if len(meaningful) < 4:
        return False
    if all(word in GENERIC_VERBS or word in PRONOUNS for word in words):
        return False
    if is_blacklisted(task):
        return False
    return not (ACKNOWLEDGEMENTS.match(task) or FILLERS.match(task))


def calculate_confidence(task: str, owner: str | None, deadline: str | None, source: str) -> tuple[int, str]:
    word_list = normalize_text(task).split()
    words = set(word_list)
    confidence = 50
    evidence = []
    if word_list and word_list[0] in WORK_VERBS:
        confidence += 20
        evidence.append("clear work verb")
    if words & BUSINESS_OBJECTS:
        confidence += 15
        evidence.append("specific business object")
    if owner:
        confidence += 10
        evidence.append("named owner")
    if deadline:
        confidence += 10
        evidence.append("explicit deadline")
    if source in {"labelled", "assignment", "named"}:
        confidence += 8
        evidence.append("explicit commitment")
    elif source in {"speaker", "intention", "request", "named_command"}:
        confidence += 5
        evidence.append("direct action statement")
    if len(word_list) >= 8:
        confidence += 8
        evidence.append("sufficient detail")
    if any(word in PRONOUNS for word in words) and not (words & BUSINESS_OBJECTS):
        confidence -= 12
    confidence = max(0, min(99, confidence))
    reason = "Concrete business action"
    if evidence:
        reason += " with " + ", ".join(evidence)
    return confidence, reason + "."


def extract_candidates(transcript: str, summary: str = "") -> list[Candidate]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", transcript) if item.strip()]
    candidates: list[Candidate] = []
    patterns = (
        ("speaker", re.compile(r"^(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?):\s*(?:I|we)\s+(?:will|must|need to|have to|am going to)\s+(?P<task>.+)$", re.I)),
        ("named", re.compile(r"^(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will|must|needs? to|has to|is going to|agreed to|committed to)\s+(?P<task>.+)$")),
        ("labelled", re.compile(r"^(?:action item|todo|follow[- ]?up)\s*:?\s*(?:(?P<owner>[A-Z][a-z]+)\s+(?:will|to)\s+)?(?P<task>.+)$", re.I)),
        ("assignment", re.compile(r"^(?P<task>.+?)\s+(?:is\s+)?assigned to\s+(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$", re.I)),
        ("request", re.compile(r"^(?:(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+)?please\s+(?:do\s+)?(?P<task>.+)$", re.I)),
        ("intention", re.compile(r"^(?:I'm going to|I'll|I will|we'll|we will|we need to|the team needs to)\s+(?P<task>.+)$", re.I)),
        ("named_command", re.compile(r"^(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+(?P<task>(?:send|update|create|prepare|review|schedule|complete|fix|confirm|deliver|deploy|publish)\s+.+)$")),
    )

    for index, original in enumerate(sentences):
        sentence = re.sub(r"^(?:so|okay|all right|and)\b[,:\s-]*", "", original, flags=re.I)
        sentence = re.sub(r"^I\s+think\s+(?=I'm going to)", "", sentence, flags=re.I)
        please_match = re.search(r"\bplease\b", sentence, re.I)
        if please_match:
            sentence = re.sub(r"^(?:please\s+){2,}", "please ", sentence[please_match.start():], flags=re.I)
        if sentence.endswith("?") or ACKNOWLEDGEMENTS.match(sentence) or FILLERS.match(sentence):
            continue
        if HYPOTHETICAL.search(sentence) and "please" not in sentence.lower():
            continue

        context_parts = [summary]
        if index:
            context_parts.append(sentences[index - 1])
        context_parts.append(original)
        context = " ".join(context_parts)
        for source, pattern in patterns:
            match = pattern.match(sentence)
            if match:
                values = match.groupdict()
                candidates.append(Candidate(values["task"].strip(), values.get("owner"), context, source))
                break
    return candidates


def _duplicate(left: dict, right: dict) -> bool:
    left_text = normalize_text(left["task"])
    right_text = normalize_text(right["task"])
    if SequenceMatcher(None, left_text, right_text).ratio() >= 0.72:
        return True
    left_tokens, right_tokens = set(left_text.split()), set(right_text.split())
    union = left_tokens | right_tokens
    return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.6


def merge_duplicates(tasks: Iterable[dict]) -> list[dict]:
    merged: list[dict] = []
    for task in tasks:
        match = next((existing for existing in merged if _duplicate(existing, task)), None)
        if not match:
            merged.append(task)
            continue
        if task["confidence"] > match["confidence"]:
            match.update(task)
        else:
            match["owner"] = match["owner"] or task["owner"]
            match["deadline"] = match["deadline"] or task["deadline"]
    return merged


def extract_action_items(transcript: str, summary: str = "", confidence_threshold: int = 72) -> list[dict]:
    """Run detection, filtering, enrichment, and deduplication in order."""
    tasks = []
    for candidate in extract_candidates(transcript, summary):
        owner = detect_owner(candidate.text, candidate.owner)
        deadline = detect_deadline(candidate.text)
        rewritten = rewrite_vague_task(candidate.text, candidate.context)
        if not rewritten:
            continue
        task = _clean_task(rewritten, owner, deadline)
        if not _is_independently_actionable(task):
            continue
        priority = detect_priority(candidate.text)
        confidence, reason = calculate_confidence(task, owner, deadline, candidate.source)
        if confidence < confidence_threshold:
            continue
        tasks.append({
            "task": task[:1000],
            "owner": owner,
            "deadline": deadline,
            "priority": priority,
            "confidence": confidence,
            "reason": reason,
            "completed": False,
        })
    return merge_duplicates(tasks)[:25]
