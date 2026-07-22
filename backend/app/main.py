from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text
from app.database.session import Base, engine
from app.routes.api import router
from app.utils.config import settings

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    # Existing SQLite databases need an in-place additive migration because
    # SQLAlchemy's create_all only creates missing tables.
    if engine.dialect.name == "sqlite":
        columns = {column["name"] for column in inspect(engine).get_columns("tasks")}
        with engine.begin() as connection:
            if "confidence" not in columns:
                connection.execute(text("ALTER TABLE tasks ADD COLUMN confidence INTEGER NOT NULL DEFAULT 0"))
            if "reason" not in columns:
                connection.execute(text("ALTER TABLE tasks ADD COLUMN reason TEXT"))
    yield
app=FastAPI(title=settings.app_name,version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router,prefix="/api")
@app.get("/health")
def health(): return {"status":"ok"}
