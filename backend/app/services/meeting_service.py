from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.ai.action_items import extract_action_items
from app.ai.engine import summarize
from app.models.meeting import Meeting
from app.models.task import Task
from app.utils.config import settings

def create_meeting(db: Session, title: str, transcript: str) -> Meeting:
    summary = summarize(transcript)
    meeting=Meeting(title=title, transcript=transcript, summary=summary)
    meeting.tasks=[Task(**item) for item in extract_action_items(transcript, summary, settings.action_confidence_threshold)]
    db.add(meeting); db.commit(); db.refresh(meeting); return meeting
def list_meetings(db: Session, search: str | None=None):
    query=select(Meeting).order_by(Meeting.upload_date.desc())
    if search: query=query.where(Meeting.title.ilike(f"%{search}%"))
    return list(db.scalars(query).unique())
def get_meeting(db: Session, meeting_id: int): return db.get(Meeting, meeting_id)
