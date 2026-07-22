import shutil, uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.ai.engine import transcribe
from app.database.session import get_db
from app.models.task import Task
from app.schemas.entities import MeetingCreate, MeetingDetail, TaskRead, TaskUpdate
from app.services.meeting_service import create_meeting, get_meeting, list_meetings
from app.utils.config import settings

router=APIRouter()
ALLOWED={".mp3",".wav",".mp4",".mov",".mkv",".webm",".m4v",".txt"}
@router.post("/upload", response_model=MeetingDetail, status_code=201)
def upload(title: str=Form(...), file: UploadFile=File(...), db: Session=Depends(get_db)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED: raise HTTPException(400,"Supported files: MP3, WAV, MP4, MOV, MKV, WEBM, M4V, TXT")
    settings.upload_dir.mkdir(parents=True,exist_ok=True); path=settings.upload_dir/f"{uuid.uuid4()}{suffix}"
    with path.open("wb") as target: shutil.copyfileobj(file.file,target)
    try:
        if path.stat().st_size > settings.max_upload_mb*1024*1024: raise HTTPException(413,"File too large")
        transcript=path.read_text(encoding="utf-8",errors="replace") if suffix==".txt" else transcribe(path)
        return create_meeting(db,title,transcript)
    finally: path.unlink(missing_ok=True)
@router.post("/summarize", response_model=MeetingDetail, status_code=201)
def summarize_text(payload: MeetingCreate, db: Session=Depends(get_db)): return create_meeting(db,payload.title,payload.transcript)
@router.get("/meetings")
def meetings(search: str|None=None, db: Session=Depends(get_db)):
    return [{"id":m.id,"title":m.title,"upload_date":m.upload_date,"summary":m.summary,"task_count":len(m.tasks)} for m in list_meetings(db,search)]
@router.get("/meeting/{meeting_id}", response_model=MeetingDetail)
def meeting(meeting_id:int, db:Session=Depends(get_db)):
    item=get_meeting(db,meeting_id)
    if not item: raise HTTPException(404,"Meeting not found")
    return item
@router.delete("/meeting/{meeting_id}", status_code=204)
def delete_meeting(meeting_id:int, db:Session=Depends(get_db)):
    item=get_meeting(db,meeting_id)
    if not item: raise HTTPException(404,"Meeting not found")
    db.delete(item); db.commit()
@router.patch("/tasks/{task_id}", response_model=TaskRead)
def update_task(task_id:int,payload:TaskUpdate,db:Session=Depends(get_db)):
    task=db.get(Task,task_id)
    if not task: raise HTTPException(404,"Task not found")
    task.completed=payload.completed; db.commit(); db.refresh(task); return task
