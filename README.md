# Medora AI — Flask Clone

Full replica of https://www.medinsight.in/ landing UI + all functionality, built with Flask.

## Features (parity with original)
- Landing `/` — same sections: hero + Analysis Progress 84% card, features (6), dark preview, How It Works, stats (10k+/95%/<30s/50+), testimonials, FAQ, CTA, footer
- Auth `/login` — sign-in / sign-up (Flask-Login + SQLite, replaces Supabase)
- `/dashboard` — report grid, counts
- `/upload` — drag&drop PDF/JPG/PNG/TIFF/DICOM, patient name + relationship, async AI pipeline
- `/analysis/<id>` — live progress polling (`/api/report_status`)
- `/report/<id>` — executive summary, findings, diagnosis+confidence+differential, pathophysiology, treatment, lifestyle, follow-up, risk bars, emergency triggers, current status, chat, doctor search, translate, share, PDF download, delete
- `/history`, `/trends` (Chart.js), `/settings`
- `/shared/<token>` — public doctor link (no login)
- `/privacy`, `/terms`, `/security`, maintenance mode
- Admin `/adminpanel/*` — dashboard, reports, users, settings (landing headline + maintenance), errors, login
- APIs: `GET /api/health`, `GET /api/report_status/<id>`, `POST /api/translate`, `POST /api/doctors`, `POST /api/chat`

## AI — Ollama + RAG (local, semantic) + Gemini fallback
- Generation: `gemma3:1b` via Ollama (`OLLAMA_MODEL`).
- Confidence is never 0.0 without reason: robust parsing (numbers, `85%`, words like
  "high"), a model-rated repair pass from the evidence, and impression phrasing when the
  summary exists but the label is empty. Non-medical uploads honestly return
  "No medical content detected" instead of a fake diagnosis.
- Embeddings: `nomic-embed-text` 768-d (`OLLAMA_EMBED_MODEL`); TF-IDF fallback if Ollama is down.
- RAG: every upload is chunked (~500 chars, 120 overlap) into `document_chunk` rows with embeddings.
  Hybrid retrieval (0.7 embedding cosine + 0.3 keyword overlap). Analysis prompt receives top-6
  retrieved excerpts; chat receives top-5 excerpts with `[S1]` citations + source previews.
- Ingestion: PDF (PyMuPDF + raw fallback), images (pytesseract OCR + vision), DOC/DOCX (python-docx), TXT, DICOM.
- Gemini fallback (`GEMINI_MODEL`, default `gemini-2.5-flash`, auto-tries `gemini-flash-latest`,
  `gemini-2.5-flash-lite`) for analysis, chat, translate, explain, and referrals.
- `GEMINI_API_KEY` still supported as secondary fallback. No mock responses anywhere: if neither
  backend is reachable, analysis marks the report `failed` and chat/translate/explain/doctors
  return HTTP 503 with a clear error instead of fabricated data.
- Endpoints: `GET /api/rag_status` (models, reachability, chunk counts), `POST /api/reindex` (admin backfill).

## Run
```powershell
ollama serve                                  # in a separate terminal
ollama pull gemma3:1b                         # you already have this
ollama pull nomic-embed-text                  # semantic embeddings (274 MB)
pip install -r requirements.txt
copy .env.example .env   # OLLAMA_HOST/MODEL/EMBED_MODEL + ADMIN_EMAIL/PASSWORD
python app.py            # http://127.0.0.1:5000
```

Admin seed: set `ADMIN_EMAIL`/`ADMIN_PASSWORD` before first run, then login at `/adminpanel/login`.
