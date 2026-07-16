from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database.session import Base, engine
from app.routes.api import router
from app.utils.config import settings

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine); yield
app=FastAPI(title=settings.app_name,version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router,prefix="/api")
@app.get("/health")
def health(): return {"status":"ok"}
