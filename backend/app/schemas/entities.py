from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
class TaskRead(BaseModel):
    id: int; meeting_id: int; task: str; owner: str | None = None; deadline: str | None = None
    priority: str = "Medium"; completed: bool = False
    model_config = ConfigDict(from_attributes=True)
class TaskUpdate(BaseModel): completed: bool
class MeetingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    transcript: str = Field(min_length=1)
class MeetingDetail(BaseModel):
    id: int; title: str; upload_date: datetime; summary: str; transcript: str; tasks: list[TaskRead]
    model_config = ConfigDict(from_attributes=True)
