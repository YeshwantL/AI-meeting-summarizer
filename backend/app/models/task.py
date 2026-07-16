from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.session import Base
class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    task: Mapped[str] = mapped_column(Text)
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deadline: Mapped[str | None] = mapped_column(String(100), nullable=True)
    priority: Mapped[str] = mapped_column(String(20), default="Medium")
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    meeting: Mapped["Meeting"] = relationship(back_populates="tasks")
from app.models.meeting import Meeting  # noqa: E402
