from datetime import datetime, timezone
from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.session import Base

class Meeting(Base):
    __tablename__ = "meetings"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    upload_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    summary: Mapped[str] = mapped_column(Text, default="")
    transcript: Mapped[str] = mapped_column(Text)
    tasks: Mapped[list["Task"]] = relationship(back_populates="meeting", cascade="all, delete-orphan", lazy="selectin")

from app.models.task import Task  # noqa: E402
