"""Medical NLP: rule-based extraction of patient + clinical entities from OCR text.
No external models required — deterministic regex/vocab layer that feeds
the UI panels, clinician summary, PDF, and RAG metadata."""
import re
from datetime import datetime

LAB_PATTERNS = [
    ("Hemoglobin", r"hemoglobin|\bHb\b|HGB", r"(\d{1,2}(?:\.\d+)?)\s*(?:g/dL|g\/dl|gm%)", "g/dL", (12.0, 17.0)),
    ("WBC", r"WBC|white\s*blood\s*cell|leukocyte|TLC", r"(\d{3,6}(?:\.\d+)?)\s*(?:/uL|/µL|cells?/uL|10\^3/uL)?", "/uL", (4000, 11000)),
    ("Platelets", r"platelet", r"(\d{4,6}(?:\.\d+)?)\s*(?:/uL|/µL)?", "/uL", (150000, 410000)),
    ("Glucose (fasting)", r"glucose.*fast|fasting.*glucose|FBS", r"(\d{2,3}(?:\.\d+)?)\s*mg/dL", "mg/dL", (70, 100)),
    ("HbA1c", r"hba1c|glycated", r"(\d{1,2}(?:\.\d+)?)\s*%", "%", (4.0, 5.6)),
    ("CRP", r"\bCRP\b|c-reactive", r"(\d{1,3}(?:\.\d+)?)\s*mg/L", "mg/L", (0, 5)),
    ("Creatinine", r"creatinine", r"(\d(?:\.\d+)?)\s*mg/dL", "mg/dL", (0.6, 1.2)),
    ("Cholesterol (total)", r"cholesterol.*total|total.*cholesterol", r"(\d{2,3}(?:\.\d+)?)\s*mg/dL", "mg/dL", (0, 200)),
    ("MCV", r"\bMCV\b", r"(\d{2,3}(?:\.\d+)?)\s*fL", "fL", (80, 100)),
    ("MCH", r"\bMCH\b", r"(\d{1,2}(?:\.\d+)?)\s*pg", "pg", (27, 32)),
    ("ESR", r"\bESR\b", r"(\d{1,3})\s*mm/hr", "mm/hr", (0, 20)),
    ("TSH", r"\bTSH\b", r"(\d(?:\.\d+)?)\s*(?:uIU/mL|mIU/L)", "uIU/mL", (0.4, 4.0)),
]

VITAL_PATTERNS = [
    ("Blood pressure", r"(?:BP|blood\s*pressure)\s*[:is]*\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg)?", None),
    ("Heart rate", r"(?:HR|heart\s*rate|pulse)\s*[:is]*\s*(\d{2,3})\s*(?:bpm)?", "bpm"),
    ("SpO2", r"(?:spo2|oxygen\s*saturation|o2\s*sat)\s*[:is]*\s*(\d{2,3})\s*%?", "%"),
    ("Temperature", r"(?:temp|temperature|fever)\s*[:is]*\s*(\d{2,3}(?:\.\d+)?)\s*(?:F|°F|C|°C)?", ""),
    ("Respiratory rate", r"(?:RR|respiratory\s*rate)\s*[:is]*\s*(\d{1,2})\s*(?:/min)?", "/min"),
]

MED_HINTS = r"\b(paracetamol|acetaminophen|ibuprofen|amoxicillin|azithromycin|metformin|atorvastatin|amlodipine|losartan|omeprazole|pantoprazole|levothyroxine|aspirin|clopidogrel|insulin|salbutamol|montelukast|cetirizine|iron|folic|vitamin|calcium|doxycycline|ciprofloxacin)\b"

DOC_TYPE_HINTS = [
    ("lab", r"CBC|complete blood|lipid|thyroid|HbA1c|pathology|hematology|biochemistry"),
    ("imaging", r"X-?ray|MRI|MRCP|CECT|contrast-enhanced|contrast|CT\b|USG|ultrasound|mammograph|ECG|echo|radiology|lesion|hypoenhanc|duct prominence|impression|scan|EUS|PET-CT|pancreas|liver|spleen|gallbladder"),
    ("prescription", r"prescription|Rx\b|dosage|take.*daily|BD\b|TDS\b|OD\b"),
    ("discharge", r"discharge summary|admitted|discharged|hospital course"),
]


def _find(pat, text, flags=re.IGNORECASE):
    return re.search(pat, text, flags)


def extract_entities(text, filename=""):
    t = text or ""
    entities = {"patient": {}, "vitals": [], "labs": [], "medications": [],
                "dates": [], "doc_type": "general", "extraction_notes": []}

    # --- patient ---
    m = _find(r"patient\s*(?:name)?\s*[:\-]\s*([A-Z][A-Za-z. ]+?)(?=\s+(?:age|sex|gender|UHID|MRN|date|reg)|$)", t)
    if m:
        entities["patient"]["name"] = m.group(1).strip(" .,-")
    m = _find(r"age\s*[:\-]?\s*(\d{1,3})\s*(?:y|yr|yrs|years)?", t)
    if m:
        entities["patient"]["age"] = m.group(1)
    m = _find(r"\b(sex|gender)\s*[:\-]?\s*(male|female|M\b|F\b)", t)
    if m:
        entities["patient"]["sex"] = m.group(2)
    m = _find(r"(?:UHID|MRN|patient\s*id|reg\s*no)\s*[:\-]?\s*([A-Za-z0-9/\-]{3,20})", t)
    if m:
        entities["patient"]["id"] = m.group(1)

    # --- labs ---
    for label, anchor, valpat, unit, ref in LAB_PATTERNS:
        for am in re.finditer(anchor, t, re.IGNORECASE):
            window = t[am.start(): am.start() + 80]  # forward-only: value follows the test name
            vm = re.search(valpat, window, re.IGNORECASE)
            if vm:
                try:
                    val = float(vm.group(1))
                except Exception:
                    continue
                flag = "normal"
                if ref and (val < ref[0] or val > ref[1]):
                    flag = "low" if val < ref[0] else "high"
                entities["labs"].append({"test": label, "value": vm.group(1),
                                         "unit": unit, "flag": flag,
                                         "ref": f"{ref[0]}–{ref[1]}" if ref else ""})
                break

    # --- vitals ---
    for label, pat, unit in VITAL_PATTERNS:
        m = _find(pat, t)
        if m:
            if label == "Blood pressure":
                entities["vitals"].append({"measure": label, "value": f"{m.group(1)}/{m.group(2)}",
                                           "unit": "mmHg", "flag": ""})
            else:
                entities["vitals"].append({"measure": label, "value": m.group(1),
                                           "unit": unit or "", "flag": ""})

    # --- medications ---
    meds = sorted(set(m.group(1).capitalize() for m in re.finditer(MED_HINTS, t, re.IGNORECASE)))
    for med in meds[:20]:
        # grab nearby dosage context
        ctx = ""
        am = _find(re.escape(med), t)
        if am:
            ctx = t[max(0, am.start() - 10): am.start() + 60].strip().split("\n")[0][:80]
        entities["medications"].append({"name": med, "context": ctx})

    # --- dates ---
    dates = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b", t, re.IGNORECASE)
    entities["dates"] = sorted(set(dates))[:8]

    # --- doc type ---
    blob = f"{filename} {t[:2000]}"
    for dtype, pat in DOC_TYPE_HINTS:
        if _find(pat, blob):
            entities["doc_type"] = dtype
            break

    if not t.strip():
        entities["extraction_notes"].append("No machine-readable text — entities inferred from filename/metadata only.")
    return entities


def specialty_for_report(diagnosis="", text="", entities=None):
    """Instant rule-based specialty for a report — no LLM call.
    Returns {primary, secondary, why}. Used on the report page header card."""
    blob = f"{diagnosis or ''} {text or ''}".lower()
    primary, secondary, why = "General Physician", "", "For initial evaluation and referral."
    rules = [
        ("Gastroenterologist", ("pancrea", "bile duct", "cholangio", "ampulla", "gallbladder", "cholecyst"),
         "Pancreatic/biliary finding needs GI (HPB) evaluation with confirmatory imaging + EUS."),
        ("Hepatologist", ("liver", "hepatic", "cirrhosis", "hepatitis"),
         "Liver finding needs hepatology/GI evaluation."),
        ("Cardiologist", ("heart", "cardiac", "coronary", "myocard", "ecg", "echo", " Valve".lower()),
         "Cardiac finding needs cardiology evaluation."),
        ("Pulmonologist", ("lung", "pulmon", "pleura", "tuberculosis", "chest nodule"),
         "Chest/lung finding needs pulmonology evaluation."),
        ("Nephrologist", ("kidney", "renal", "nephro", "creatinine", "dialysis"),
         "Kidney finding needs nephrology evaluation."),
        ("Neurologist", ("brain", "neuro", "seizure", "migraine", "stroke", "spine", "disc"),
         "Neuro finding needs neurology evaluation."),
        ("Orthopedic", ("fracture", "bone", "joint", "arthritis", "ortho", "tendon", "ligament"),
         "Bone/joint finding needs orthopedic evaluation."),
        ("Dermatologist", ("skin", "dermat", "rash", "psoriasis", "eczema"),
         "Skin finding needs dermatology evaluation."),
        ("Endocrinologist", ("thyroid", "diabet", "hba1c", "hormone", "endocrine", "pituitary"),
         "Hormonal/metabolic finding needs endocrinology evaluation."),
        ("Hematologist", ("anemia", "hemoglobin", "platelet", "leukemia", "lymphoma", "wbc", "thalassemia"),
         "Blood finding needs hematology evaluation."),
        ("Urologist", ("prostate", "psa", "bladder", "urinary", "uti", "kidney stone", "ureter"),
         "Urinary finding needs urology evaluation."),
        ("Gynecologist", ("uterus", "ovary", "ovarian", "cervix", "pregnan", "fibroid", "pcos"),
         "Pelvic finding needs gynecology evaluation."),
        ("Ophthalmologist", ("eye", "vision", "retina", "cataract", "glaucoma"),
         "Eye finding needs ophthalmology evaluation."),
        ("ENT Specialist", ("ear", "nose", "throat", "sinus", "tonsil", "hearing"),
         "ENT finding needs ENT evaluation."),
        ("Psychiatrist", ("depress", "anxiety", "psych", "bipolar", "schizo"),
         "Mental-health finding needs psychiatry evaluation."),
        ("Pediatrician", ("pediatric", "child", "infant", "neonatal"),
         "Child case needs pediatric evaluation."),
    ]
    for spec, keys, reason in rules:
        if any(k in blob for k in keys):
            primary, why = spec, reason
            break
    if any(k in blob for k in ("malignan", "carcinoma", "adenocarcinoma", "neoplas", "tumor", "tumour",
                               "suspicious", "space-occupying", "metasta")):
        secondary = "Oncologist"
        why += " Suspicious features also warrant oncology opinion after tissue diagnosis."
    return {"primary": primary, "secondary": secondary, "why": why}


def clinician_summary(analysis, entities, filename=""):
    """Compact clinician-oriented structured summary dict."""
    d = (analysis or {}).get("diagnosis", {}) or {}
    probs = [d.get("primary", "")] + list(d.get("differential", []) or [])
    return {
        "document": filename,
        "doc_type": (entities or {}).get("doc_type", "general"),
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "patient": (entities or {}).get("patient", {}),
        "problem_list": [p for p in probs if p][:5],
        "key_labs": (entities or {}).get("labs", [])[:12],
        "vitals": (entities or {}).get("vitals", [])[:8],
        "medications": [m["name"] for m in (entities or {}).get("medications", [])][:12],
        "plan": ((analysis or {}).get("treatmentPlan", {}) or {}),
        "follow_up": (analysis or {}).get("followUpActions", []),
        "red_flags": (analysis or {}).get("emergencyTriggers", []),
        "gaps": (analysis or {}).get("dataGaps", []),
    }
