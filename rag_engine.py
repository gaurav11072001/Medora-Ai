"""RAG engine: Ollama (gemma3:1b generation + nomic-embed-text embeddings)
with TF-IDF fallback. Stores chunks in SQLAlchemy model defined in app.py."""
import os
import re
import json
import math

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEN_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Single-model setup (e.g. only gemma3:1b installed): cache `ollama list`
# so we can skip missing embed/vision models without error spam.
_models_cache = {"names": None}

def _available_models():
    if _models_cache["names"] is not None:
        return _models_cache["names"]
    try:
        import ollama
        c = ollama.Client(host=OLLAMA_HOST)
        data = c.list()
        models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
        names = []
        for m in models:
            if isinstance(m, dict):
                names.append(m.get("name", ""))
            else:
                names.append(getattr(m, "model", "") or "")
        _models_cache["names"] = [n for n in names if n]
        return _models_cache["names"]
    except Exception:
        return []

# ---------- text utils ----------
def chunk_text(text, size=500, overlap=120):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    # sentence-aware split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            # hard-split very long sentences
            while len(s) > size:
                chunks.append(s[:size])
                s = s[size - overlap:]
            cur = s
    if cur:
        chunks.append(cur)
    # add overlap
    overlapped = []
    for i, c in enumerate(chunks):
        if i and overlap:
            prev_tail = chunks[i - 1][-overlap:]
            c = prev_tail + " … " + c
        overlapped.append(c)
    return overlapped

def _cos(a, b):
    try:
        import numpy as np
        a = np.array(a, dtype=float); b = np.array(b, dtype=float)
        n = float((a @ b) / (math.sqrt(float(a @ a)) * math.sqrt(float(b @ b)) + 1e-9))
        return n
    except Exception:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-9
        nb = math.sqrt(sum(y * y for y in b)) + 1e-9
        return dot / (na * nb)

# ---------- ollama ----------
def ollama_ok():
    try:
        import ollama
        c = ollama.Client(host=OLLAMA_HOST)
        c.list()
        return True
    except Exception:
        return False

def embed_texts(texts):
    """Return list of vectors. Prefers Ollama embeddings, falls back to TF-IDF.
    Single-model config (only gemma3:1b): set OLLAMA_EMBED_MODEL=tfidf/disabled/empty
    or leave it pointing at a model you haven't pulled — we skip Ollama and use TF-IDF.
    INSTANT_ANALYSIS=1 skips Ollama embeddings entirely (TF-IDF, <0.1s) because
    a cold nomic-embed-text takes ~19s for the first call and blocks analysis."""
    import os as _os
    texts = [t[:2000] for t in texts]
    if not texts:
        return []
    instant = _os.environ.get("INSTANT_ANALYSIS", "1") != "0"
    if instant:
        # Fast path: TF-IDF is instant and good enough for 2-3k char docs.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=512, stop_words="english").fit(texts)
            return [row.toarray()[0].tolist() for row in vec.transform(texts)]
        except Exception:
            pass  # fall through to hashed vectors below
    want_ollama = (EMBED_MODEL or "").strip().lower() not in ("", "none", "disabled", "tfidf", "tf-idf", "local")
    if want_ollama:
        installed = _available_models()
        # exact or base-name match (e.g. "nomic-embed-text" vs "nomic-embed-text:latest")
        base = EMBED_MODEL.split(":")[0]
        found = any(n == EMBED_MODEL or n.startswith(base + ":") or n == base for n in installed)
        if installed and not found:
            print(f"Ollama embed model '{EMBED_MODEL}' not installed (have: {installed}). Using TF-IDF fallback — no pull needed.")
            want_ollama = False
    if want_ollama:
        try:
            import ollama
            c = ollama.Client(host=OLLAMA_HOST)
            vecs = []
            for t in texts:
                r = c.embeddings(model=EMBED_MODEL, prompt=t)
                v = r.get("embedding") if isinstance(r, dict) else r.embedding
                vecs.append(v)
            if vecs and vecs[0]:
                return vecs
        except Exception as e:
            print("Ollama embed failed, TF-IDF fallback:", str(e)[:160])
    # TF-IDF fallback (semantic-ish keyword retrieval)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=512, stop_words="english").fit(texts)
        return [row.toarray()[0].tolist() for row in vec.transform(texts)]
    except Exception:
        # last-resort hashed vectors
        import hashlib
        out = []
        for t in texts:
            v = [0.0] * 128
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16) % 128
                v[h] += 1.0
            out.append(v)
        return out

def generate(prompt, system="", temperature=0.25, max_tokens=1800, json_mode=False):
    import ollama
    # Local CPU models (qwen3:4b) need minutes, not seconds. Short timeouts made
    # the Ollama fallback fail outright whenever Gemini was unavailable.
    try:
        from ai_engine import gemini_available
        _tmo = 300 if not gemini_available() else 90
    except Exception:
        _tmo = 120
    c = ollama.Client(host=OLLAMA_HOST, timeout=_tmo)
    model = GEN_MODEL
    try:
        installed = _available_models()
        if installed:
            base = (GEN_MODEL or "").split(":")[0]
            if not any(n == GEN_MODEL or n.startswith(base + ":") or n == base for n in installed):
                # configured model not pulled — use what's actually there (e.g. gemma3:1b)
                model = installed[0]
                print(f"Ollama model '{GEN_MODEL}' not installed, using '{model}' instead.")
    except Exception:
        pass
    full = (system + "\n\n" + prompt) if system else prompt
    # gemma3:1b is small — keep context tight for speed
    full = full[:6000]
    import os as _os
    try:
        _threads = max(4, (_os.cpu_count() or 4))
    except Exception:
        _threads = 4
    kw = {"model": model, "prompt": full, "keep_alive": "10m",
          "options": {"temperature": temperature, "num_predict": max_tokens,
                      "num_ctx": 2048, "num_thread": _threads}}
    if json_mode:
        kw["format"] = "json"
    r = c.generate(**kw)
    txt = r.get("response", "") if isinstance(r, dict) else r.response
    return (txt or "").strip()

def _vision_candidates():
    """Installed Ollama models with vision capability, medgemma first.
    Single-model config (only gemma3:1b): returns [] so callers skip
    straight to Gemini/OCR instead of pulling a vision model."""
    import os as _os
    pref = (_os.environ.get("OLLAMA_VISION_MODEL", "") or "").strip()
    if pref.lower() in ("", "none", "disabled", "skip"):
        return []
    if not pref:
        pref = "medgemma:4b"
    try:
        names = _available_models()
        if not names:
            return [pref]
    except Exception:
        return [pref]
    cands = []
    for n in names:
        if n.startswith(pref.split(":")[0]):
            cands.append(n)
    try:
        import ollama as _o
        c2 = _o.Client(host=OLLAMA_HOST)
        for n in names:
            if n in cands:
                continue
            try:
                info = c2.show(n)
                caps = info.get("capabilities", []) if isinstance(info, dict) else getattr(info, "capabilities", [])
                if "vision" in [str(x).lower() for x in caps]:
                    cands.append(n)
            except Exception:
                continue
    except Exception:
        pass
    # Don't append `pref` if it isn't installed — single-model setups
    # (only gemma3:1b) should skip local vision, not error on a missing pull.
    return cands

def _gemini_vision(image_path, question):
    from ai_engine import GEMINI_KEY, GEMINI_MODELS, gemini_available, _is_quota_error, _gemini_block
    if not gemini_available():
        raise RuntimeError("Gemini vision skipped (cooldown/quota)")
    import google.generativeai as genai
    import os as _os
    genai.configure(api_key=GEMINI_KEY or _os.environ.get("GEMINI_API_KEY", ""))
    from PIL import Image
    img = Image.open(image_path)
    last = None
    for name in GEMINI_MODELS:
        try:
            resp = genai.GenerativeModel(name).generate_content([question, img])
            txt = (resp.text or "").strip()
            if txt:
                return txt
            last = ValueError(f"{name}: empty reply")
        except Exception as e:
            last = e
            if _is_quota_error(e):
                _gemini_block(str(e))
                break
    raise RuntimeError(f"Gemini vision failed: {last}")

def describe_image(image_path, question=None, modality_hint=""):
    """Medical image understanding chain (CT/X-ray/MRI/photos):
    1) local open-source vision model via Ollama (MedGemma preferred),
    2) Gemini Flash vision fallback. Returns text or ''."""
    if question is None:
        question = ("You are a medical imaging assistant. Analyze this clinical image"
                    + (f" (suspected modality: {modality_hint})" if modality_hint else "") + ". "
                    "Report: 1) likely modality and body region, 2) key visible findings with locations, "
                    "3) overall impression and urgency, 4) what a radiologist should verify next. "
                    "Be specific and grounded in what you actually see.")
    import ollama
    c = ollama.Client(host=OLLAMA_HOST)
    for m in _vision_candidates():
        try:
            r = c.generate(model=m, prompt=question, images=[image_path],
                           options={"temperature": 0.2, "num_predict": 900})
            txt = (r.get("response", "") if isinstance(r, dict) else r.response or "").strip()
            if txt:
                print(f"Vision OK via {m}")
                return txt
        except Exception as e:
            print(f"Vision via {m} failed:", str(e)[:150])
            continue
    try:
        return _gemini_vision(image_path, question)
    except Exception as e:
        print("Gemini vision failed:", str(e)[:150])
        return ""

# ---------- RAG ops (DB access injected from app) ----------
def ingest_chunks(db, DocumentChunk, report_id, user_id, text, filename=""):
    chunks = chunk_text(text)
    if not chunks:
        chunks = [f"Clinical document: {filename or report_id}. Limited machine-readable text; analyze by document type and available metadata."]
    # clear old
    DocumentChunk.query.filter_by(report_id=report_id).delete()
    db.session.commit()
    vecs = embed_texts(chunks)
    for i, (ch, v) in enumerate(zip(chunks, vecs)):
        db.session.add(DocumentChunk(report_id=report_id, user_id=user_id,
                                     chunk_index=i, text=ch,
                                     embedding=json.dumps(v)))
    db.session.commit()
    return len(chunks)

def retrieve(db, DocumentChunk, query, report_id=None, user_id=None, top_k=5):
    """Hybrid semantic retrieval: 0.7 cosine similarity + 0.3 keyword overlap,
    so paraphrased questions still match while exact terms get a boost."""
    import re as _re
    q = DocumentChunk.query
    if report_id is not None:
        q = q.filter_by(report_id=report_id)
    elif user_id is not None:
        q = q.filter_by(user_id=user_id)
    rows = q.all()
    if not rows:
        return []
    qv = embed_texts([query])[0]
    qwords = set(w for w in _re.findall(r"\w+", query.lower()) if len(w) > 2)
    scored = []
    for r in rows:
        cos_s, lex_s = None, 0.0
        try:
            v = json.loads(r.embedding or "[]")
            if v and len(v) == len(qv):
                cos_s = _cos(qv, v)
        except Exception:
            pass
        words = set(w for w in _re.findall(r"\w+", (r.text or "").lower()) if len(w) > 2)
        if qwords:
            lex_s = len(qwords & words) / len(qwords)
        if cos_s is None:
            score = lex_s  # embedding dim mismatch (TF-IDF fallback) → lexical only
        else:
            score = 0.7 * ((cos_s + 1.0) / 2.0) + 0.3 * lex_s
        scored.append((score, r))
    # TF-IDF fallback dim mismatch: use keyword overlap
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, r in scored[:top_k]:
        out.append({"score": round(float(s), 4), "text": r.text,
                    "report_id": r.report_id, "chunk": r.chunk_index})
    return out

def build_context(snippets):
    parts = []
    for i, s in enumerate(snippets, 1):
        parts.append(f"[S{i} | report {s['report_id']} chunk {s['chunk']} | score {s['score']}]\n{s['text']}")
    return "\n\n".join(parts)
