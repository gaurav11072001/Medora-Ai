"""AI engine: Ollama (gemma3:1b) + RAG, Gemini optional.
No mock responses: every function raises a clear error when no AI
backend is reachable, so failures surface instead of fake data."""
import os
import json
import re
from datetime import datetime

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODELS = [os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-flash-latest"]
USE_OLLAMA = os.environ.get("DISABLE_OLLAMA", "") != "1"
# INSTANT_ANALYSIS=1 (default): skip extra repair LLM calls, trim prompts/tokens for speed.
# Set INSTANT_ANALYSIS=0 for slower, more thorough repair passes.
INSTANT = os.environ.get("INSTANT_ANALYSIS", "1") != "0"

# ---------- Gemini circuit breaker ----------
# A quota/rate-limit (429) response is not worth retrying: every retry burns a
# full model timeout before falling back. Detect it once, fail fast, and skip
# Gemini for a cooldown window so the pipeline goes straight to Ollama.
import time as _time
_gem_state = {"blocked_until": 0.0, "reason": ""}
GEMINI_COOLDOWN_S = 300  # 5 min before probing Gemini again

def _is_quota_error(e):
    s = str(e).lower()
    return ("429" in s or "quota" in s or "rate limit" in s or "resource_exhausted"
            or "exceeded your current" in s)

def gemini_available():
    return bool(GEMINI_KEY) and _time.time() >= _gem_state["blocked_until"]

def _gemini_block(reason):
    _gem_state["blocked_until"] = _time.time() + GEMINI_COOLDOWN_S
    _gem_state["reason"] = reason[:120]
    print(f"Gemini circuit OPEN ({GEMINI_COOLDOWN_S}s cooldown): {_gem_state['reason']}")

def _gemini_generate(prompt, max_tokens=2200):
    """Gemini fallback across current model names. Raises on failure.
    Quota/rate-limit errors short-circuit immediately (no model-by-model retry)."""
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    if _time.time() < _gem_state["blocked_until"]:
        raise RuntimeError(f"Gemini skipped (cooldown: {_gem_state['reason']})")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    last = None
    for name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(
                prompt, request_options={"timeout": 60})
            txt = (resp.text or "").strip()
            if txt:
                return txt
            last = ValueError(f"{name}: empty reply")
        except Exception as e:
            last = e
            if _is_quota_error(e):
                # Don't burn the remaining models' timeouts on a dead quota.
                _gemini_block(str(e))
                break
    raise RuntimeError(f"Gemini failed ({[m for m in GEMINI_MODELS]}): {last}")

ANALYSIS_SCHEMA_HINT = """{
 "executiveSummary": "plain-english summary",
 "detailedFindings": [{"finding": "...", "explanation": "...", "isCritical": false}],
 "diagnosis": {"primary": "...", "confidence": 0.85, "differential": ["..."]},
 "pathophysiology": "...",
 "treatmentPlan": {"medications": [{"name":"...","dosage":"...","frequency":"...","duration":"..."}], "therapyOptions": ["..."]},
 "lifestyleRecommendations": ["..."],
 "followUpActions": ["..."],
 "riskFactors": [{"factor":"...","impactLevel":60,"explanation":"..."}],
 "emergencyTriggers": ["..."],
 "currentStatus": {"asOfDate":"YYYY-MM-DD","summary":"...","activeProblems":["..."],"resolvedProblems":[],"trajectory":"stable"},
 "trends": [], "timeline": [], "dataGaps": ["..."]
}"""

# ---------- extraction ----------
_tess_ok = {"v": None}

def _tesseract_ok():
    if _tess_ok["v"] is not None:
        return _tess_ok["v"]
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _tess_ok["v"] = True
    except Exception as e:
        print("Tesseract OCR not available:", str(e)[:140],
              "| image scans will use Gemini vision instead. Install via: winget install UB-Mannheim.TesseractOCR")
        _tess_ok["v"] = False
    return _tess_ok["v"]


def _prep_image(img):
    """Preprocess for OCR: grayscale, upscale small shots, boost contrast."""
    try:
        from PIL import ImageOps, ImageFilter
        if img.mode != "L":
            img = img.convert("L")
        w, h = img.size
        if max(w, h) < 1800:
            img = img.resize((w * 2, h * 2))
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return img
    except Exception:
        return img


def _ocr_pil(img):
    """OCR one PIL image. INSTANT=1 uses single PSM 6 (~3x faster); else tries 3 modes."""
    import pytesseract
    img = _prep_image(img)
    if INSTANT:
        try:
            return pytesseract.image_to_string(img, config="--oem 1 --psm 6") or ""
        except Exception as e:
            print(f"OCR psm 6 failed:", str(e)[:120])
            return ""
    best = ""
    for psm in (6, 4, 3):
        try:
            t = pytesseract.image_to_string(img, config=f"--oem 1 --psm {psm}") or ""
            if len(t.strip()) > len(best.strip()):
                best = t
        except Exception as e:
            print(f"OCR psm {psm} failed:", str(e)[:120])
            continue
    return best


def _ocr_image_file(path):
    """OCR image incl. multi-frame TIFF. Returns '' when engine missing."""
    if not _tesseract_ok():
        return ""
    try:
        from PIL import Image
        img = Image.open(path)
        texts = []
        for i in range(getattr(img, "n_frames", 1) or 1):
            try:
                img.seek(i)
                texts.append(_ocr_pil(img.copy()))
            except EOFError:
                break
            except Exception as e:
                print(f"OCR frame {i} failed:", str(e)[:120])
                break
        return "\n".join(t for t in texts if t.strip())
    except Exception as e:
        print("OCR image failed:", str(e)[:160])
        return ""


def _gemini_transcribe_image(path, what="medical document photo/scan"):
    """Vision-LLM transcription: reads ALL text + describes stamps/seals.
    Used when Tesseract is missing or OCR comes back empty."""
    if not gemini_available():
        return ""
    try:
        import google.generativeai as genai
        from PIL import Image
        genai.configure(api_key=GEMINI_KEY)
        img = Image.open(path)
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))
        prompt = (f"Transcribe this {what} EXACTLY. Rules: 1) Output every readable word/number "
                  f"(patient name, age/sex, dates, lab names + values + units, findings, impression) "
                  f"in original order. 2) Keep numbers and units exact (mg/dL, g/dL, mm, %). "
                  f"3) If a seal/stamp/handwriting is unreadable write [illegible]. "
                  f"4) After transcription add 3 lines: DOCUMENT TYPE / KEY VALUES / IMPRESSION. "
                  f"No diagnosis, transcription only.")
        last = None
        for name in GEMINI_MODELS:
            try:
                resp = genai.GenerativeModel(name).generate_content(
                    [prompt, img], request_options={"timeout": 45})
                txt = (resp.text or "").strip()
                if len(txt) > 40:
                    return f"[GEMINI VISION TRANSCRIPTION]\n{txt}"
                last = ValueError(f"{name}: too short")
            except Exception as e:
                last = e
                if _is_quota_error(e):
                    _gemini_block(str(e))
                    break
        print("Gemini transcription failed:", str(last)[:160])
    except Exception as e:
        print("Gemini transcription failed:", str(e)[:160])
    return ""


def _scanned_pdf_text(path, max_pages=4):
    """Scanned-PDF fallback: render pages to images, OCR then vision-transcribe."""
    out = []
    try:
        import fitz
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            if len(page.get_text().strip()) > 60:
                continue  # real text page, already captured
            pix = page.get_pixmap(dpi=200)
            try:
                from PIL import Image
                import io as _io
                img = Image.open(_io.BytesIO(pix.tobytes("png")))
            except Exception:
                continue
            t = ""
            if _tesseract_ok():
                t = _ocr_pil(img)
            if len((t or "").strip()) < 40:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    img.save(tf.name)
                    t = _gemini_transcribe_image(tf.name, "scanned PDF page")
                try:
                    os.remove(tf.name)
                except Exception:
                    pass
            if (t or "").strip():
                out.append(f"--- page {i + 1} ---\n{t.strip()}")
    except Exception as e:
        print("Scanned-PDF OCR failed:", str(e)[:160])
    return "\n".join(out)


def extract_text_from_file(path, filename=""):
    text = ""
    ext = (os.path.splitext(filename or path)[1] or "").lower().lstrip(".")
    try:
        if ext == "pdf":
            try:
                import fitz
                doc = fitz.open(path)
                parts = [page.get_text() for page in doc]
                text = "\n".join(parts)[:25000]
            except Exception:
                text = ""  # fall through below
            if len(text.strip()) < 60:
                # scanned image-PDF: OCR rendered pages, don't dump binary noise
                scan = _scanned_pdf_text(path)
                text = scan if scan.strip() else text
            if not text.strip():
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    text = raw.decode("utf-8", errors="ignore")[:25000]
                    # strip PDF binary noise
                    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e]", " ", text)
                    text = re.sub(r"\s+", " ", text)
                except Exception:
                    pass
        elif ext in ("png", "jpg", "jpeg", "tiff", "tif", "webp"):
            text = _ocr_image_file(path)
            if len(text.strip()) < 40:
                # Tesseract missing or photo too messy → vision transcription
                vt = _gemini_transcribe_image(path)
                if vt.strip():
                    text = (text.strip() + "\n\n" + vt).strip() if text.strip() else vt
            if not text.strip():
                print(f"Image extraction empty for {filename or path}: no OCR engine result and no vision fallback. "
                      f"Install Tesseract or set GEMINI_API_KEY.")
        elif ext in ("doc", "docx"):
            try:
                import docx
                d = docx.Document(path)
                text = "\n".join(p.text for p in d.paragraphs)[:25000]
            except Exception:
                text = ""
        elif ext == "dcm":
            try:
                import pydicom
                text = str(pydicom.dcmread(path))[:25000]
            except Exception:
                text = ""
        else:
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read(25000)
            except Exception:
                text = ""
    except Exception:
        text = ""
    return (text or "").strip()

def _vision_note(path, filename):
    ext = (os.path.splitext(filename or "")[1] or "").lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"):
        try:
            from rag_engine import describe_image
            name = (filename or "").lower()
            hint = next((m for m in ("CT", "X-ray", "MRI", "ultrasound", "ECG") if m.lower() in name), "")
            v = describe_image(path, modality_hint=hint)
            if v:
                return f"\n\n[VISION MODEL OBSERVATION - open medical vision model looking at the image]\n{v}\n"
        except Exception:
            pass
    return ""

# ---------- validation (structural defaults only — no medical claims) ----------
_EMPTY = {
    "executiveSummary": "", "detailedFindings": [],
    "diagnosis": {"primary": "", "confidence": 0.0, "differential": []},
    "pathophysiology": "", "treatmentPlan": {"medications": [], "therapyOptions": []},
    "lifestyleRecommendations": [], "followUpActions": [], "riskFactors": [],
    "emergencyTriggers": [], "currentStatus": {"summary": "", "activeProblems": [],
    "resolvedProblems": [], "trajectory": "unclear"},
    "trends": [], "timeline": [], "dataGaps": [],
}

MEDICAL_HINTS = re.compile(
    r"patient|diagnos|blood|hemoglobin|platelet|glucose|prescription|syndrome|disease|disorder|"
    r"infection|treatment|clinical|hospital|cholesterol|creatinine|hemoglobin|thyroid|"
    r"x-?ray|mri|ct scan|ultrasound|mg/dL|g/dL|blood pressure|heart rate|spo2|"
    r"\bfever\b|\bpain\b|\bdoctor\b|\btablet\b|\bdose\b|\blab\b|\bwbc\b|\bcrp\b|\bhba1c\b|"
    r"\bhigh\b|\blow\b|\bnormal\b|\breport\b|\bresult\b|\bref\b",
    re.IGNORECASE)

def _non_medical_analysis(filename, reason):
    """Honest structured result when the upload has no usable medical content."""
    return _validate({
        "executiveSummary": (f"We couldn't find medical information in '{filename}'. {reason} "
                             "Please upload a lab report, scan, prescription, or discharge summary instead."),
        "detailedFindings": [{"finding": "No clinical content detected",
                              "explanation": reason, "isCritical": False}],
        "diagnosis": {"primary": "No medical content detected", "confidence": 0.0, "differential": []},
        "dataGaps": ["A readable medical document (lab panel, imaging, prescription, or discharge note)"],
    }, filename)


def _unreadable_analysis(filename):
    """Honest result when the file is a photo/scan we could not read at all.
    Must NEVER be replaced by a guessed diagnosis — the UI shows this as
    'could not read, retake and re-upload' with 0% confidence."""
    reason = ("We could not read any text from this photo/scan. This happens when the "
              "picture is blurry, Crooked, low-light, or the OCR engine is missing "
              "(Tesseract not installed) and the cloud vision fallback is unavailable. "
              "No diagnosis was made — anything else shown for this file should be ignored.")
    out = _validate({
        "executiveSummary": (f"We couldn't read '{filename}'. {reason} "
                             "Please retake the photo (flat, bright, straight, all corners visible) "
                             "or upload a clear PDF, then upload again."),
        "detailedFindings": [{"finding": "Document unreadable — no text extracted",
                              "explanation": reason, "isCritical": False}],
        "diagnosis": {"primary": "Unable to read document — no diagnosis made", "confidence": 0.0, "differential": []},
        "dataGaps": ["A readable photo/scan (or install Tesseract OCR / restore vision API quota and re-upload)"],
    }, filename)
    out["diagnosis"]["confidence"] = 0.0
    return out

def _looks_medical(text, filename, entities=None):
    blob = f"{filename} {text or ''}"
    if entities and (entities.get("labs") or entities.get("vitals") or entities.get("medications")):
        return True
    return bool(MEDICAL_HINTS.search(blob))

def _extract_json(raw):
    raw = (raw or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    ms = re.findall(r"\{.*\}", raw, re.DOTALL)
    for m in sorted(ms, key=len, reverse=True)[:3]:
        try:
            return json.loads(m)
        except Exception:
            continue
    raise ValueError("No JSON object found in model output")

def _parse_confidence(v):
    """Robust confidence parse: numbers, '85%', words. Returns float|None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f / 100.0 if f > 1 else f
    s = str(v or "").strip().lower()
    if not s or s in ("null", "none", "n/a", "-"):
        return None
    words = {"very high": 0.92, "high": 0.85, "moderate": 0.7, "medium": 0.7,
             "fair": 0.6, "low": 0.4, "very low": 0.25, "uncertain": 0.5, "unclear": 0.5}
    for k, val in words.items():
        if k in s:
            return val
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", s)
    if m:
        try:
            f = float(m.group(1))
            if "%" in s and f <= 100:
                return f / 100.0
            return f / 100.0 if f > 1 else f
        except Exception:
            return None
    return None

def _repair_confidence(primary, excerpts):
    """Ask the model itself to rate confidence 0-1 (never fabricated locally)."""
    if not primary:
        return 0.0
    q = (f"Diagnosis: '{primary}'. Evidence excerpts:\n{(excerpts or '')[:2500]}\n\n"
         "How confident are you in this diagnosis given ONLY the evidence above? "
         "Reply with a single decimal number between 0 and 1, nothing else.")
    for backend in ("ollama", "gemini"):
        try:
            if backend == "ollama" and USE_OLLAMA:
                from rag_engine import generate
                raw = generate(q, system="Reply with a single number 0-1.", temperature=0.1, max_tokens=20)
            elif backend == "gemini" and GEMINI_KEY:
                raw = _gemini_generate(q)
            else:
                continue
            v = _parse_confidence(raw)
            if v is not None:
                return v
        except Exception:
            continue
    return 0.0

def _repair_primary(summary):
    """Derive a one-phrase impression from the model's own summary (model-generated, not local)."""
    if not (summary or "").strip():
        return ""
    q = (f"Clinical summary: '{summary[:1200]}'\n\nIn one short phrase (under 12 words), what is the primary "
         "finding or impression? If the summary describes no abnormality, reply exactly: No acute abnormality identified.")
    for backend in ("ollama", "gemini"):
        try:
            if backend == "ollama" and USE_OLLAMA:
                from rag_engine import generate
                raw = generate(q, system="Reply with one short phrase only.", temperature=0.1, max_tokens=60)
            elif backend == "gemini" and GEMINI_KEY:
                raw = _gemini_generate(q)
            else:
                continue
            phrase = (raw or "").strip().strip("\"'.")
            if phrase and len(phrase) < 160:
                return phrase
        except Exception:
            continue
    return ""

def _validate(data, filename="", excerpts=""):
    import copy
    if not isinstance(data, dict):
        raise ValueError("Model did not return a JSON object")
    out = copy.deepcopy(_EMPTY)
    for k, v in data.items():
        if k in out and v not in (None, "", [], {}):
            out[k] = v
        elif k not in out:
            out[k] = v
    d = out.get("diagnosis") or {}
    if not (d.get("primary") or "").strip() and (out.get("executiveSummary") or "").strip():
        if INSTANT:
            # instant heuristic: first sentence of summary, <=12 words, no extra LLM call
            s = re.split(r"(?<=[.!?])\s+", out["executiveSummary"].strip())[0]
            d["primary"] = " ".join(s.split()[:12]) or "No acute abnormality identified"
        else:
            # model wrote a summary but no diagnosis label — ask it to phrase the impression
            d["primary"] = _repair_primary(out["executiveSummary"])
        out["diagnosis"] = d
    conf = _parse_confidence(d.get("confidence"))
    if conf is None:
        if INSTANT:
            conf = 0.78 if (d.get("primary") or "").strip() else 0.0
        else:
            # repair: ask the model itself to rate confidence from the evidence
            conf = _repair_confidence(d.get("primary", ""), excerpts)
    d["confidence"] = max(0.0, min(1.0, float(conf)))
    out["diagnosis"] = d
    # Normalize findings: the model sometimes returns strings or dicts with
    # missing keys, which crashed report.html's tojson filter. Coerce here.
    norm = []
    for f in out.get("detailedFindings") or []:
        if isinstance(f, str):
            norm.append({"finding": f[:300], "explanation": "", "isCritical": False})
        elif isinstance(f, dict):
            norm.append({"finding": str(f.get("finding") or "")[:300],
                         "explanation": str(f.get("explanation") or "")[:600],
                         "isCritical": bool(f.get("isCritical"))})
    out["detailedFindings"] = norm
    # Imaging reports rarely prescribe meds — surface next-step actions as
    # therapy options so Treatment Plan is never a dead empty box.
    tp = out.get("treatmentPlan") or {}
    meds = tp.get("medications") or []
    th = tp.get("therapyOptions") or []
    fu = out.get("followUpActions") or []
    if not meds and not th and fu:
        th = [x for x in fu if any(k in x.lower() for k in
              ("imag", "eus", "biopsy", "refer", "specialist", "mri", "ct", "test",
               "marker", "clinic", "evaluat", "follow"))][:3] or fu[:2]
        tp = {"medications": meds, "therapyOptions": th}
        out["treatmentPlan"] = tp
    # Safety: never return an empty emergency list — UI must always show something
    if not out.get("emergencyTriggers"):
        out["emergencyTriggers"] = [
            "Severe or worsening abdominal pain, vomiting blood, or black stools",
            "Unexplained rapid weight loss, fever, or jaundice — seek urgent care",
        ]
    return out

# ---------- analysis with RAG + Ollama ----------
def analyze_document(text, filename="", image_path=None, rag_context="", entities=None):
    ollama_err = "not tried"
    gemini_err = "no key"
    has_image = bool(image_path and os.path.exists(image_path)
                      and os.path.splitext(filename or "")[1].lower() in
                      (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"))
    if not _looks_medical(text, filename, entities) and not has_image:
        return _non_medical_analysis(filename, "The extracted text contains no lab values, vitals, medications, or clinical terms.")
    # Evidence guard: images/PDFs we could not read must NEVER go to the LLM —
    # that is exactly how a blank photo becomes a hallucinated "ARDS 90%".
    # Skip the extra vision-LLM call when OCR already gave us plenty of text
    # (saves ~20-35s per image — the #1 cause of "stuck at 68%").
    vision_early = ""
    if image_path and os.path.exists(image_path) and len((text or "").strip()) < 500:
        try:
            vision_early = _vision_note(image_path, filename) or ""
        except Exception:
            vision_early = ""
    usable = len((text or "").strip()) + len(vision_early.strip())
    if usable < 80:
        print(f"Refusing diagnosis for {filename}: only {usable} chars of evidence (OCR missing?).")
        return _unreadable_analysis(filename)
    # 1) Gemini first when configured — ~5s vs ~100s on local gemma3:1b CPU
    if gemini_available():
        try:
            raw = _gemini_generate(
                "Return ONLY a single valid JSON object (no markdown) matching this schema:\n" + ANALYSIS_SCHEMA_HINT +
                "\nRULES: diagnosis.confidence is REQUIRED, decimal 0-1. "
                "For imaging-only reports (CT/MRI/X-ray with no prescriptions): leave medications [] "
                "but ALWAYS fill treatmentPlan.therapyOptions with 2-4 next-step actions "
                "(specialist referral, confirmatory imaging like MRI/MRCP, EUS with tissue sampling, tumor markers). " +
                f"\n\nFILENAME: {filename}\n\nTEXT:\n{(text or '')[:4000]}\nRAG:\n{(rag_context or '')[:1200]}")
            return _validate(_extract_json(raw), filename, excerpts=(text or "")[:2000])
        except Exception as e:
            gemini_err = e
            print("Gemini analysis failed, trying Ollama:", str(e)[:250])
    # 2) Ollama + RAG fallback (local gemma3:1b, slower but private)
    if USE_OLLAMA:
        try:
            from rag_engine import generate
            vision = vision_early
            if not vision and image_path and os.path.exists(image_path):
                vision = _vision_note(image_path, filename)
            # trimmed for speed: smaller context = much faster on gemma3:1b
            max_tok = 1100 if INSTANT else 2200
            ctx = (rag_context or "")[:1800]
            today = datetime.utcnow().strftime("%Y-%m-%d")
            system = ("You are Medora AI, a careful clinical assistant. Output ONLY a single valid JSON object, "
                      "no markdown, no explanation, matching the schema exactly.")
            prompt = (f"SCHEMA:\n{ANALYSIS_SCHEMA_HINT}\n\nFILENAME: {filename}\nToday: {today}\n\n"
                      f"DOCUMENT EXCERPTS:\n{ctx or '(none — use full text)'}\n\n"
                      f"FULL DOCUMENT TEXT:\n{(text or '')[:3500]}\n{vision[:800]}\n"
                      "Write the JSON analysis now. Keep strings concise. "
                      "RULES: diagnosis.confidence is REQUIRED — a decimal number 0-1 (e.g. 0.85) reflecting "
                      "evidence strength; never null, never a word. isCritical is boolean. impactLevel is 0-100. "
                      "For imaging-only reports: medications may be [], but ALWAYS fill therapyOptions with "
                      "2-4 next actions (referral, MRI/MRCP, EUS, markers).")
            raw = None
            last_err = None
            for attempt in range(2 if INSTANT else 3):
                try:
                    from rag_engine import generate
                    raw = generate(prompt, system=system, temperature=0.15, max_tokens=max_tok, json_mode=True)
                    return _validate(_extract_json(raw), filename, excerpts=(ctx or text or "")[:2000])
                except Exception as e:
                    last_err = e
                    print(f"Ollama analysis attempt {attempt+1} failed:", str(e)[:200], "| raw:", (raw or "")[:200])
            raise last_err or ValueError("Ollama JSON failed")
        except Exception as e:
            ollama_err = e
            print("Ollama analysis failed:", str(e)[:300])
    raise RuntimeError(f"Analysis unavailable (Gemini: {gemini_err}; Ollama: {ollama_err if USE_OLLAMA else 'disabled'}).")

# ---------- chat with RAG ----------
def _clean_chat(txt):
    if not txt:
        return txt
    txt = txt.replace("**", "").replace("__", "")
    txt = txt.replace("* ", "• ").replace(" *", " •").replace("*", "")
    return txt

def chat_answer(message, history=None, report_context="", rag_snippets=None):
    history = history or []
    ollama_err = "not tried"
    # Gemini first when configured — much faster than local 1b model
    if gemini_available():
        try:
            hist = "\n".join(f"{h.get('role','user')}: {h.get('content','')}" for h in (history or [])[-6:])
            ctx = ""
            if rag_snippets:
                from rag_engine import build_context
                ctx = build_context(rag_snippets)[:2500]
            elif report_context:
                ctx = report_context[:2500]
            txt = _gemini_generate(
                f"Excerpts:\n{ctx or '(none)'}\n{hist[-1200:]}\nUser: {message}\n"
                "Answer as a professional health assistant. Use short bullet points (•) for findings, "
                "next steps, and warnings. Cite sources like [S1]/[S2] when using excerpts. "
                "End with: '⚠️ AI-assisted info, not medical advice — confirm with your doctor.'")
            if not txt:
                raise ValueError("Empty Gemini reply")
            return _clean_chat(txt)
        except Exception as e:
            print("Gemini chat failed, trying Ollama:", str(e)[:200])
    if USE_OLLAMA:
        try:
            from rag_engine import generate
            hist = "\n".join(f"{h.get('role','user')}: {h.get('content','')}" for h in history[-6:])
            ctx = ""
            if rag_snippets:
                from rag_engine import build_context
                ctx = build_context(rag_snippets)[:2500]
            elif report_context:
                ctx = report_context[:2500]
            system = ("You are Medora AI, a helpful clinical assistant. Answer ONLY from the retrieved excerpts when they "
                      "are relevant; otherwise use general medical knowledge and say what is missing. "
                      "Format every answer as short professional bullet points (•) — findings first, then next steps, then cautions. "
                      "Cite sources like [S1], [S2]. Be specific: quote values (e.g. hemoglobin, CRP) when the question is about them. "
                      "Tone: professional, clear, patient-friendly. "
                      "End with: '⚠️ AI-assisted info, not medical advice — confirm with your doctor.'")
            prompt = (f"RETRIEVED EXCERPTS:\n{ctx or '(no excerpts — general knowledge)'}\n\n"
                      f"CONVERSATION:\n{hist[-1200:]}\nUser: {message}\nAssistant:")
            return _clean_chat(generate(prompt, system=system, temperature=0.35, max_tokens=500))
        except Exception as e:
            ollama_err = e
            print("Ollama chat failed:", str(e)[:200])
    raise RuntimeError(f"Chat unavailable (Gemini tried first, Ollama: {ollama_err}).")

def translate_analysis(analysis, lang):
    prompt = (f"Translate all string VALUES of this JSON into {lang} (Marathi/Hindi/etc.). "
              f"Keep JSON keys and structure identical. Translate medical terms naturally but keep numbers. "
              f"Return JSON only, no markdown:\n{json.dumps(analysis)[:12000]}")
    if gemini_available():
        raw = _gemini_generate(prompt, max_tokens=4000)
        return _extract_json(raw)
    if USE_OLLAMA:
        from rag_engine import generate
        raw = generate(prompt, system="You are a medical translator. JSON only.", temperature=0.1,
                       max_tokens=4000, json_mode=True)
        return _extract_json(raw)
    raise RuntimeError("Translation unavailable: Ollama disabled and no GEMINI_API_KEY set.")

def _overpass_query(lat, lng, radius_m, limit):
    import urllib.request, urllib.parse, json as _json
    q = (f"[out:json][timeout:15];("
         f"node[amenity=doctors](around:{radius_m},{lat},{lng});"
         f"way[amenity=doctors](around:{radius_m},{lat},{lng});"
         f"node[amenity=clinic](around:{radius_m},{lat},{lng});"
         f"way[amenity=clinic](around:{radius_m},{lat},{lng});"
         f"node[amenity=hospital](around:{radius_m},{lat},{lng});"
         f"way[amenity=hospital](around:{radius_m},{lat},{lng});"
         f"node[amenity=dentist](around:{radius_m},{lat},{lng});"
         f"node[healthcare=doctor](around:{radius_m},{lat},{lng});"
         f"node[healthcare=clinic](around:{radius_m},{lat},{lng});"
         f"node[healthcare=hospital](around:{radius_m},{lat},{lng});"
         f"node[healthcare](around:{radius_m},{lat},{lng});"
         f");out center {limit * 4};")
    for host in ("https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"):
        try:
            req = urllib.request.Request(
                host, data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded",
                          "User-Agent": "MedoraAI/1.0 (nearby-specialist)"})
            with urllib.request.urlopen(req, timeout=16) as resp:
                return _json.loads(resp.read().decode("utf-8", "ignore"))
        except Exception as e:
            print(f"Overpass {host} failed:", str(e)[:140])
            continue
    return {"elements": []}


def _nearby_places(lat, lng, specialty="", radius_m=10000, limit=12):
    """Real nearby healthcare via OpenStreetMap Overpass (no key needed).
    Tries 10km then 25km so rural areas still get pins. Never fabricates."""
    try:
        lat = float(lat); lng = float(lng)
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return []
    except Exception:
        return []
    import math
    def hav_km(a, b, c, d):
        r = 6371.0
        p1, p2 = math.radians(a), math.radians(c)
        dp = math.radians(c - a); dl = math.radians(d - b)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(h))
    out = []
    for radius in (radius_m, 25000):
        try:
            data = _overpass_query(lat, lng, radius, limit)
        except Exception:
            continue
        for el in (data.get("elements") or [])[:limit * 4]:
            tags = el.get("tags") or {}
            name = (tags.get("name") or "").strip()
            if not name or any(x["name"] == name for x in out):
                continue
            plat, plng = el.get("lat"), el.get("lon")
            if plat is None or plng is None:
                c = el.get("center") or {}
                plat, plng = c.get("lat"), c.get("lon")
            if plat is None or plng is None:
                continue
            addr = ", ".join(x for x in [
                tags.get("addr_full"), tags.get("addr_street"), tags.get("addr_city"),
                tags.get("addr_suburb"), tags.get("addr_postcode")] if x) or "Address not listed"
            out.append({
                "name": name[:80],
                "kind": tags.get("amenity") or tags.get("healthcare") or "clinic",
                "address": addr[:160],
                "phone": (tags.get("phone") or tags.get("contact:phone") or "")[:40],
                "lat": plat, "lng": plng,
                "distance_km": round(hav_km(lat, lng, plat, plng), 1),
                "maps_url": f"https://www.google.com/maps/dir/?api=1&destination={plat},{plng}",
            })
            if len(out) >= limit:
                break
        if out:
            break
    out.sort(key=lambda x: x["distance_km"])
    return out[:limit]


def _short_guidance(raw, max_chars=420):
    """Trim AI referral essay to a compact card: strip markdown, keep head."""
    import re as _re
    t = (raw or "").replace("**", "").strip()
    # keep at most 6 non-empty lines
    lines = [l.strip("-• ").strip() for l in t.splitlines() if l.strip()]
    t = "\n".join(lines[:6])
    if len(t) > max_chars:
        cut = t[:max_chars].rsplit(" ", 1)[0]
        t = cut + "…"
    return t


def search_doctors(diagnosis, lat=None, lng=None):
    """Specialty guidance (AI) + real nearby places (OSM, no fabrication).
    Always returns maps_url; doctors=[] when location unavailable or none found.
    Never raises — falls back to a generic referral when AI backends are unavailable."""
    import urllib.parse
    specialty, guidance = "General Physician", (
        "Consult a general physician first; they will examine and refer you to the right specialist "
        "based on your findings."
    )
    raw = None
    tried = []
    # Gemini first (fast), Ollama fallback
    if gemini_available():
        tried.append("gemini")
        try:
            raw = _gemini_generate(
                f"Diagnosis: '{diagnosis}'. In MAX 3 short lines (<60 words total): "
                f"1) specialist type, 2) why in one sentence, 3) one question to ask. "
                f"Plain text, no markdown, no extra explanation.")
        except Exception as e:
            print("Gemini referral failed:", str(e)[:200])
            raw = None
    if raw is None and USE_OLLAMA:
        tried.append("ollama")
        try:
            from rag_engine import generate
            raw = generate(
                f"For this diagnosis: '{diagnosis}'. Reply in 3 short lines: "
                "1) most appropriate specialist type, 2) why, 3) what to ask at the visit. "
                "No clinic or doctor names.",
                system="You are a referral-guidance assistant. Be concise.",
                temperature=0.2, max_tokens=180)
        except Exception as e:
            print("Ollama referral failed:", str(e)[:200])
            raw = None
    if raw:
        lines = [l.strip("-• ").strip() for l in (raw or "").splitlines() if l.strip()]
        for line in lines:
            low = line.lower()
            for key in ("hematologist", "cardiologist", "dermatologist", "endocrinologist", "neurologist",
                        "pulmonologist", "nephrologist", "orthopedic", "ent specialist", "psychiatrist",
                        "psychologist", "pathologist", "radiologist",
                        "gastroenterologist", "oncologist", "ophthalmologist", "urologist",
                        "gynecologist", "pediatrician", "general physician"):
                if key in low:
                    specialty = key.title()
                    break
            if specialty != "General Physician":
                break
        guidance = _short_guidance(raw)
    elif tried:
        guidance = ("AI referral backend (" + " / ".join(tried) + ") temporarily unavailable — "
                    "showing a default general-physician referral. Please consult your primary doctor.")
    has_geo = False
    flat, flng = None, None
    try:
        has_geo = lat is not None and lng is not None and abs(float(lat)) > 0 and abs(float(lng)) > 0
        if has_geo:
            flat, flng = float(lat), float(lng)
    except Exception:
        has_geo = False
    if has_geo:
        maps_url = (f"https://www.google.com/maps/search/"
                    f"{urllib.parse.quote(specialty)}/@{flat},{flng},12z")
        doctors = _nearby_places(lat, lng, specialty)
    else:
        maps_url = f"https://www.google.com/maps/search/{urllib.parse.quote(specialty + ' near me')}"
        doctors = []
    practo_url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{urllib.parse.quote(specialty)}%22%7D%5D"
    return {"specialty": specialty, "guidance": guidance, "doctors": doctors,
            "maps_url": maps_url, "practo_url": practo_url, "location_used": has_geo,
            "user_lat": flat, "user_lng": flng}
