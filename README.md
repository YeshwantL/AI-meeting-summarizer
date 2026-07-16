# MeetingMind — AI Meeting Summarizer

A full-stack FastAPI + React application that accepts MP3, WAV, MP4, and TXT meetings, creates a concise BART summary, extracts action items, and stores meeting history in SQLite.

## Architecture

```text
backend/app/
├── ai/          # Whisper, Transformers, action extraction
├── database/    # SQLAlchemy engine and sessions
├── models/      # Meeting and Task ORM models
├── routes/      # REST controllers
├── schemas/     # Pydantic API contracts
├── services/    # Business/application logic
├── utils/       # Configuration
└── main.py
frontend/src/
├── api/         # Axios client
├── App.jsx      # Routed screens and UI components
├── main.jsx
└── styles.css
```

## Prerequisites

- Python 3.11 recommended
- Node.js 20+
- FFmpeg on `PATH` (required by Whisper for audio/video)
- About 2–5 GB of disk space for Torch and downloaded AI models

## Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`; Swagger documentation is at `http://localhost:8000/docs`. Database tables are created on startup. Models load lazily on their first use.

For quick UI/API development without downloading AI models, set `ENABLE_AI_MODELS=false`. TXT uploads still work and use the built-in extractive summary fallback; audio transcription is disabled in that mode.

## Frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`.

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/upload` | Multipart upload (`title`, `file`) and analyze |
| POST | `/api/summarize` | Analyze JSON transcript (`title`, `transcript`) |
| GET | `/api/meetings?search=` | List/search meeting history |
| GET | `/api/meeting/{id}` | Get summary, transcript, and tasks |
| DELETE | `/api/meeting/{id}` | Delete a meeting and its tasks |
| PATCH | `/api/tasks/{id}` | Set `{ "completed": true/false }` |

Example transcript request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/summarize `
  -ContentType application/json `
  -Body '{"title":"Product sync","transcript":"Maya will send the launch brief by tomorrow. This is urgent."}'
```

## Notes

- Uploaded files are temporary and deleted after processing; transcripts and results persist in SQLite.
- Action extraction is deterministic and best-effort. It detects commitment language, named owners, relative/date-like deadlines, and priority cues.
- Generated PDFs are created client-side, so exporting does not upload data again.
- Production deployments should add authentication, object storage, background jobs for long recordings, Alembic migrations, antivirus scanning, and a reverse proxy upload limit.
