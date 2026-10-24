import os
import json
import secrets
import threading
import time
import re
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for, flash,
                   jsonify, send_file, abort, session)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from ai_engine import (extract_text_from_file, analyze_document, chat_answer,
                       translate_analysis, search_doctors)

app = Flask(__name__)
_secret = os.environ.get("SECRET_KEY", "")
if not _secret and os.environ.get("FLASK_ENV") == "production":
    raise RuntimeError("SECRET_KEY must be set in production")
app.config["SECRET_KEY"] = _secret or "medinsight-dev-secret-change-me"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///medinsight.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if os.environ.get("COOKIE_SECURE") == "1":
    app.config["SESSION_COOKIE_SECURE"] = True
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─── Flask-Mail (magic link / OTP) ───────────────────────────────────────────
app.config["MAIL_SERVER"]         = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"]           = int(os.environ.get("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"]        = os.environ.get("MAIL_USE_TLS", "1") == "1"
app.config["MAIL_USERNAME"]       = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"]       = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER",
                                                     "Medora AI <noreply@medinsight.in>")
mail = Mail(app)

MAIL_ENABLED = bool(os.environ.get("MAIL_USERNAME"))

# ─── Google OAuth config ──────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return resp

ALLOWED = {"pdf", "png", "jpg", "jpeg", "tiff", "tif", "webp", "dcm", "txt", "doc", "docx"}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------- Models ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)
    name = db.Column(db.String(120), default="Patient")
    avatar = db.Column(db.String(300), default="")
    role = db.Column(db.String(20), default="patient")  # patient | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    file_name = db.Column(db.String(300), default="")
    file_type = db.Column(db.String(20), default="")
    file_path = db.Column(db.String(500), default="")
    status = db.Column(db.String(20), default="processing")  # processing|completed|failed
    progress = db.Column(db.Integer, default=5)
    analysis = db.Column(db.Text, default="{}")
    detected_patient_name = db.Column(db.String(200), default="")
    relationship = db.Column(db.String(50), default="Patient")
    share_token = db.Column(db.String(64), unique=True)
    share_expires_at = db.Column(db.DateTime, nullable=True)  # None = never expires
    share_revoked = db.Column(db.Boolean, default=False)
    entities = db.Column(db.Text, default="{}")  # medical-NLP extracted JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def analysis_json(self):
        try:
            return json.loads(self.analysis or "{}")
        except Exception:
            return {}

    def entities_json(self):
        try:
            return json.loads(self.entities or "{}")
        except Exception:
            return {}

    def share_active(self):
        if self.share_revoked or not self.share_token:
            return False
        if self.share_expires_at and datetime.utcnow() > self.share_expires_at:
            return False
        return True

class DocumentChunk(db.Model):
    """RAG vector store: one row per text chunk with embedding JSON."""
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey("report.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    chunk_index = db.Column(db.Integer, default=0)
    text = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.Text, default="[]")  # JSON list[float]
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ShareAccess(db.Model):
    """Audit log for public shared-link views."""
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey("report.id"), nullable=False, index=True)
    viewed_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip = db.Column(db.String(80), default="")
    user_agent = db.Column(db.String(300), default="")

class AppConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    landing_headline = db.Column(db.String(300), default="Clear answers. Confident care.")
    maintenance_mode = db.Column(db.Boolean, default=False)

class ErrorLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text)
    path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    # lightweight SQLite migration for columns added after first deploy
    try:
        from sqlalchemy import inspect, text as _text
        insp = inspect(db.engine)
        if "report" in insp.get_table_names():
            cols = {c["name"] for c in insp.get_columns("report")}
            with db.engine.begin() as conn:
                if "share_expires_at" not in cols:
                    conn.execute(_text("ALTER TABLE report ADD COLUMN share_expires_at DATETIME"))
                if "share_revoked" not in cols:
                    conn.execute(_text("ALTER TABLE report ADD COLUMN share_revoked BOOLEAN DEFAULT 0"))
                if "entities" not in cols:
                    conn.execute(_text("ALTER TABLE report ADD COLUMN entities TEXT DEFAULT '{}'"))
    except Exception as mex:
        print("migration check:", str(mex)[:200])
    if not AppConfig.query.first():
        db.session.add(AppConfig())
        db.session.commit()
    # seed admin if env given
    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_pass = os.environ.get("ADMIN_PASSWORD")
    if admin_email and admin_pass and not User.query.filter_by(email=admin_email).first():
        db.session.add(User(email=admin_email, password_hash=generate_password_hash(admin_pass),
                            name="Admin", role="admin"))
        db.session.commit()
    # stale cleanup: threads die on restart, so any 'processing' report would
    # hang at 14%/68% forever. Mark them failed so the UI stops polling.
    try:
        stale = Report.query.filter_by(status="processing").all()
        for s in stale:
            s.status = "failed"
            s.progress = 0
            s.analysis = json.dumps({"executiveSummary": "Analysis was interrupted (server restarted or timed out). Please upload again.",
                                     "diagnosis": {"primary": "Analysis interrupted — please retry", "confidence": 0.0, "differential": []}})
        if stale:
            db.session.commit()
            print(f"marked {len(stale)} stale processing report(s) as failed")
    except Exception as sex:
        print("stale cleanup:", str(sex)[:150])

@login_manager.user_loader
def load_user(uid):
    try:
        return db.session.get(User, int(uid))
    except Exception:
        return None

def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not current_user.is_authenticated or current_user.role != "admin":
            abort(403)
        return fn(*a, **kw)
    return wrapper

@app.context_processor
def inject_pending():
    from datetime import datetime as _dt
    try:
        if current_user.is_authenticated:
            n = Report.query.filter_by(user_id=current_user.id, status="processing").count()
            return {"pending_count": n,
                    "now_day": _dt.now().strftime("%a, %d %b"),
                    "google_oauth_enabled": GOOGLE_OAUTH_ENABLED}
    except Exception:
        pass
    return {"pending_count": 0,
            "now_day": _dt.now().strftime("%a, %d %b"),
            "google_oauth_enabled": GOOGLE_OAUTH_ENABLED}

@app.before_request
def check_maintenance():
    if request.path.startswith("/static"):
        return None
    try:
        cfg = AppConfig.query.first()
    except Exception:
        return None
    if cfg and cfg.maintenance_mode:
        is_admin = current_user.is_authenticated and current_user.role == "admin"
        if not is_admin and request.path not in ("/login", "/adminpanel/login") and not request.path.startswith("/api/"):
            return render_template("maintenance.html"), 503
    return None

@app.errorhandler(Exception)
def handle_err(e):
    try:
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            # API routes must get JSON, not HTML error pages
            if request.path.startswith("/api/"):
                return jsonify({"error": e.description if hasattr(e, "description") else str(e)}), e.code
            return e
        db.session.add(ErrorLog(message=str(e)[:2000], path=request.path[:300]))
        db.session.commit()
    except Exception:
        pass
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal error. The team has been notified."}), 500
    raise e

# ---------------- Background analysis (RAG + Ollama) ----------------
ANALYSIS_TIMEOUT_S = int(os.environ.get("ANALYSIS_TIMEOUT_S", "150"))

def run_analysis_async(report_id):
    def worker():
        import time as _t
        t0 = _t.time()
        with app.app_context():
            r = db.session.get(Report, report_id)
            if not r:
                return
            try:
                r.progress = 14
                db.session.commit()
                text = extract_text_from_file(r.file_path, r.file_name)
                if _t.time() - t0 > ANALYSIS_TIMEOUT_S:
                    raise TimeoutError(f"Analysis timed out during text extraction ({ANALYSIS_TIMEOUT_S}s). Try a smaller/clearer file.")
                # Feature 1 — OCR/NLP: structured entity extraction
                try:
                    from medical_nlp import extract_entities
                    ents = extract_entities(text, r.file_name)
                    r.entities = json.dumps(ents)
                    pname = (ents.get("patient") or {}).get("name")
                    if pname and not r.detected_patient_name:
                        r.detected_patient_name = pname
                    db.session.commit()
                except Exception as nex:
                    print("NLP extraction failed:", str(nex)[:200])
                # RAG ingest: chunk + embed (TF-IDF instant when no embed model)
                rag_ctx = ""
                try:
                    from rag_engine import ingest_chunks, retrieve, build_context
                    n = ingest_chunks(db, DocumentChunk, r.id, r.user_id, text, r.file_name)
                    print(f"RAG ingested {n} chunks for report {r.id}")
                    r.progress = 55
                    db.session.commit()
                    # retrieve smaller context for speed (top-4 is enough for gemma3:1b)
                    probe = f"key clinical findings diagnosis lab values impression {r.file_name}"
                    snippets = retrieve(db, DocumentChunk, probe, report_id=r.id, top_k=4)
                    rag_ctx = build_context(snippets)
                except Exception as rex:
                    print("RAG ingest failed:", str(rex)[:250])
                if _t.time() - t0 > ANALYSIS_TIMEOUT_S:
                    raise TimeoutError(f"Analysis timed out during indexing ({ANALYSIS_TIMEOUT_S}s). Try a smaller/clearer file.")
                r.progress = 68
                db.session.commit()
                # bump to 80 while the LLM runs so the UI never looks frozen at 68%
                def _touch(p):
                    try:
                        rr = db.session.get(Report, report_id)
                        if rr and rr.status == "processing":
                            rr.progress = p
                            db.session.commit()
                    except Exception:
                        pass
                import threading as _th
                stop = {"v": False}
                def _ticker():
                    for p in (75, 82, 88, 93):
                        for _ in range(15):
                            if stop["v"]:
                                return
                            _t.sleep(1)
                        _touch(p)
                _th.Thread(target=_ticker, daemon=True).start()
                try:
                    result = analyze_document(text, r.file_name, image_path=r.file_path, rag_context=rag_ctx,
                                              entities=r.entities_json())
                finally:
                    stop["v"] = True
                if _t.time() - t0 > ANALYSIS_TIMEOUT_S + 60:
                    raise TimeoutError("Analysis timed out waiting for the AI model. Please retry.")
                r.analysis = json.dumps(result)
                r.status = "completed"
                r.progress = 100
                db.session.commit()
                print(f"analysis {report_id} done in {_t.time()-t0:.0f}s")
            except Exception as ex:
                import traceback; traceback.print_exc()
                r.status = "failed"
                r.progress = 0
                try:
                    r.analysis = json.dumps({"executiveSummary": f"Analysis failed: {ex}. Please try again with a clearer file.",
                                             "diagnosis": {"primary": "Analysis failed — please retry", "confidence": 0.0, "differential": []}})
                    db.session.add(ErrorLog(message=f"analysis {report_id}: {ex}", path="/upload"))
                    db.session.commit()
                except Exception:
                    pass
    threading.Thread(target=worker, daemon=True).start()

# ---------------- Public / landing ----------------
@app.route("/")
def index():
    cfg = AppConfig.query.first()
    headline = cfg.landing_headline if cfg else "Clear answers. Confident care."
    return render_template("index.html", headline=headline)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/security")
def security():
    return render_template("security.html")

# ---------------- Auth ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        name = request.form.get("name", "").strip()
        mode = request.form.get("mode", "signin")
        user = User.query.filter_by(email=email).first()
        if mode == "signup":
            if user:
                flash("Account exists. Please sign in.", "error")
            elif not email or not password:
                flash("Email and password required.", "error")
            else:
                user = User(email=email, password_hash=generate_password_hash(password),
                            name=name or email.split("@")[0], role="patient")
                db.session.add(user)
                db.session.commit()
                login_user(user)
                user.last_login = datetime.utcnow()
                db.session.commit()
                return redirect(url_for("dashboard"))
        else:
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                user.last_login = datetime.utcnow()
                db.session.commit()
                nxt = request.args.get("next") or url_for("dashboard")
                return redirect(nxt)
            flash("Invalid email or password.", "error")
    return render_template("login.html", initial_mode=request.args.get("mode", "signin"))

@app.route("/register")
def register_page():
    return redirect(url_for("login", mode="signup"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ─────────────────────────────────────────────────────────────────────────────
# Magic-Link authentication
# ─────────────────────────────────────────────────────────────────────────────
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

def _magic_serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="magic-link")

def _send_magic_link(email: str):
    """Generate a signed magic link and send it. Returns (ok, error_msg)."""
    if not MAIL_ENABLED:
        return False, "Email is not configured on this server."
    token = _magic_serializer().dumps(email)
    link  = url_for("magic_verify", token=token, _external=True)
    body  = f"""Hi,

Click the link below to sign in to Medora AI (expires in 15 minutes):

  {link}

If you did not request this, please ignore this email.

— Medora AI Care Portal"""
    try:
        msg = Message("Your Medora AI sign-in link", recipients=[email], body=body)
        mail.send(msg)
        return True, ""
    except Exception as e:
        return False, str(e)[:300]

@app.route("/auth/magic", methods=["POST"])
def magic_request():
    """Send a magic-link email."""
    email = (request.form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("login"))
    ok, err = _send_magic_link(email)
    if ok:
        flash(f"Magic link sent to {email} — check your inbox (and spam folder).", "ok")
    else:
        if "not configured" in err:
            # Dev fallback: show the token directly
            token = _magic_serializer().dumps(email)
            link  = url_for("magic_verify", token=token, _external=True)
            flash(f"[Dev mode — email not configured] Magic link: {link}", "ok")
        else:
            flash(f"Could not send email: {err}", "error")
    return redirect(url_for("login"))

@app.route("/auth/magic/<token>")
def magic_verify(token):
    """Verify a magic-link token and log the user in."""
    try:
        email = _magic_serializer().loads(token, max_age=900)  # 15 min
    except SignatureExpired:
        flash("This magic link has expired. Please request a new one.", "error")
        return redirect(url_for("login"))
    except BadSignature:
        flash("Invalid magic link.", "error")
        return redirect(url_for("login"))
    user = User.query.filter_by(email=email).first()
    if not user:
        # Auto-create account
        user = User(email=email,
                    password_hash=generate_password_hash(secrets.token_hex(24)),
                    name=email.split("@")[0], role="patient")
        db.session.add(user)
        db.session.commit()
    login_user(user)
    user.last_login = datetime.utcnow()
    db.session.commit()
    flash("Welcome! You are now signed in.", "ok")
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────────────────────────────────────────
# Email OTP authentication
# ─────────────────────────────────────────────────────────────────────────────
def _send_otp(email: str):
    """Generate a 6-digit OTP, store in session, send by email. Returns (ok, otp, error)."""
    otp  = str(secrets.randbelow(900000) + 100000)  # 100000-999999
    session["otp_code"]    = otp
    session["otp_email"]   = email
    session["otp_expires"] = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    if not MAIL_ENABLED:
        return False, otp, "email not configured"
    body = f"""Hi,

Your Medora AI one-time login code is:

  {otp}

This code expires in 10 minutes. Do not share it with anyone.

— Medora AI Care Portal"""
    try:
        msg = Message("Your Medora AI login code", recipients=[email], body=body)
        mail.send(msg)
        return True, otp, ""
    except Exception as e:
        return False, otp, str(e)[:300]

@app.route("/auth/otp", methods=["POST"])
def otp_request():
    """Send a 6-digit OTP to the user's email."""
    email = (request.form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("login"))
    ok, otp, err = _send_otp(email)
    if ok:
        flash(f"A 6-digit code was sent to {email}.", "ok")
    else:
        if "email not configured" in err:
            flash(f"[Dev mode] Your OTP is: {otp}  (email not configured)", "ok")
        else:
            flash(f"Could not send OTP: {err}", "error")
    return redirect(url_for("otp_verify_page", email=email))

@app.route("/auth/otp/verify", methods=["GET", "POST"])
def otp_verify_page():
    email = request.args.get("email") or session.get("otp_email") or ""
    if request.method == "POST":
        code  = (request.form.get("otp") or "").strip()
        s_otp = session.get("otp_code", "")
        s_em  = session.get("otp_email", "")
        s_exp = session.get("otp_expires", "")
        if not s_otp or not s_em:
            flash("Session expired — please request a new code.", "error")
            return redirect(url_for("login"))
        try:
            if datetime.utcnow() > datetime.fromisoformat(s_exp):
                flash("OTP expired — please request a new code.", "error")
                session.pop("otp_code", None); session.pop("otp_email", None)
                return redirect(url_for("login"))
        except Exception:
            pass
        if not secrets.compare_digest(code, s_otp):
            flash("Incorrect code. Try again.", "error")
            return render_template("otp_verify.html", email=s_em)
        # Valid OTP
        session.pop("otp_code", None); session.pop("otp_email", None); session.pop("otp_expires", None)
        user = User.query.filter_by(email=s_em).first()
        if not user:
            user = User(email=s_em,
                        password_hash=generate_password_hash(secrets.token_hex(24)),
                        name=s_em.split("@")[0], role="patient")
            db.session.add(user)
            db.session.commit()
        login_user(user)
        user.last_login = datetime.utcnow()
        db.session.commit()
        flash("You are now signed in.", "ok")
        return redirect(url_for("dashboard"))
    return render_template("otp_verify.html", email=email)


# ─────────────────────────────────────────────────────────────────────────────
# Google OAuth (manual PKCE-less flow — configure CLIENT_ID + CLIENT_SECRET)
# ─────────────────────────────────────────────────────────────────────────────
import urllib.parse, urllib.request

@app.route("/auth/google")
def google_auth():
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google sign-in is unavailable. Please choose another sign-in method.", "error")
        return redirect(url_for("login"))
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    params = urllib.parse.urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  url_for("google_callback", _external=True),
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "prompt":        "select_account",
    })
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")

@app.route("/auth/google/callback")
def google_callback():
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google Sign-In is not configured.", "error")
        return redirect(url_for("login"))
    state = request.args.get("state", "")
    if state != session.pop("oauth_state", ""):
        flash("OAuth state mismatch — please try again.", "error")
        return redirect(url_for("login"))
    code = request.args.get("code")
    if not code:
        flash("Google sign-in was cancelled.", "error")
        return redirect(url_for("login"))
    # Exchange code for token
    token_data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri":  url_for("google_callback", _external=True),
        "grant_type":    "authorization_code",
    }).encode()
    try:
        req  = urllib.request.Request("https://oauth2.googleapis.com/token",
                                      data=token_data, method="POST")
        resp = urllib.request.urlopen(req, timeout=10)
        tok  = json.loads(resp.read())
    except Exception as ex:
        flash(f"Google sign-in failed: {ex}", "error")
        return redirect(url_for("login"))
    id_token_raw = tok.get("id_token", "")
    if not id_token_raw:
        flash("Google did not return an ID token.", "error")
        return redirect(url_for("login"))
    # Decode payload (we trust Google's TLS; skip sig verification for simplicity)
    import base64
    payload_b64 = id_token_raw.split(".")[1]
    payload_b64 += "==" * (4 - len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        flash("Could not decode Google token.", "error")
        return redirect(url_for("login"))
    g_email = payload.get("email", "").lower()
    g_name  = payload.get("name") or g_email.split("@")[0]
    if not g_email:
        flash("Google did not return an email.", "error")
        return redirect(url_for("login"))
    user = User.query.filter_by(email=g_email).first()
    if not user:
        user = User(email=g_email,
                    password_hash=generate_password_hash(secrets.token_hex(24)),
                    name=g_name, role="patient")
        db.session.add(user)
        db.session.commit()
    login_user(user)
    user.last_login = datetime.utcnow()
    db.session.commit()
    flash("Signed in with Google.", "ok")
    return redirect(url_for("dashboard"))



# ---------------- App pages ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    q = (request.args.get("q") or "").strip().lower()
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.desc()).all()
    if q:
        reports = [r for r in reports if q in (r.file_name or "").lower() or q in (r.detected_patient_name or "").lower()]
    all_reports = Report.query.filter_by(user_id=current_user.id).all()
    total = len(all_reports)
    done = len([r for r in all_reports if r.status == "completed"])
    awaiting = len([r for r in all_reports if r.status == "processing"])
    people = len({(r.detected_patient_name or "").strip().lower() for r in all_reports if r.detected_patient_name}) or (1 if total else 0)
    pending_count = awaiting
    return render_template("dashboard.html", reports=reports, total=total, done=done,
                           awaiting=awaiting, people=people, user=current_user, pending_count=pending_count)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "No file selected"}), 400
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED:
            return jsonify({"error": f"Unsupported format .{ext}. Use PDF, JPG, PNG, TIFF, DICOM."}), 400
        fname = secure_filename(f.filename)
        save_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}_{fname}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        f.save(save_path)
        from datetime import timedelta as _td
        r = Report(user_id=current_user.id, file_name=fname, file_type=ext,
                   file_path=save_path, status="processing", progress=5,
                   detected_patient_name=request.form.get("patient_name", "") or current_user.name,
                   relationship=request.form.get("relationship", "Patient"),
                   share_token=secrets.token_urlsafe(16),
                   share_expires_at=datetime.utcnow() + _td(days=7))
        db.session.add(r)
        db.session.commit()
        run_analysis_async(r.id)
        return jsonify({"id": r.id, "redirect": url_for("analysis", report_id=r.id)})
    return render_template("upload.html")

@app.route("/analysis/<int:report_id>")
@login_required
def analysis(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    return render_template("analysis.html", report=r)

@app.route("/report/<int:report_id>")
@login_required
def report_detail(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    from medical_nlp import clinician_summary, specialty_for_report
    data = r.analysis_json()
    diag = ((data.get("diagnosis") or {}).get("primary") or "")
    return render_template("report.html", report=r, data=data,
                           entities=r.entities_json(),
                           specialty=specialty_for_report(diag, data.get("executiveSummary", ""), r.entities_json()),
                           clinician=clinician_summary(data, r.entities_json(), r.file_name))

@app.route("/history")
@login_required
def history():
    page = int(request.args.get("page", 1))
    per = 10
    f = (request.args.get("q") or "").strip().lower()
    base = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.desc()).all()
    if f:
        base = [r for r in base if f in (r.file_name or "").lower() or f in (r.detected_patient_name or "").lower()]
    total = len(base)
    reports = base[(page - 1) * per: page * per]
    return render_template("history.html", reports=reports, page=page, total=total, per=per)

@app.route("/trends")
@login_required
def trends():
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.asc()).all()
    points = []
    for r in reports:
        try:
            conf = (r.analysis_json().get("diagnosis") or {}).get("confidence", 0)
        except Exception:
            conf = 0
        points.append({"date": r.created_at.strftime("%Y-%m-%d"), "file": r.file_name, "confidence": round(float(conf or 0) * 100)})
    return render_template("trends.html", points=points, reports=reports)

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        current_user.name = request.form.get("name", current_user.name)
        current_user.avatar = request.form.get("avatar", current_user.avatar or "")
        if request.form.get("new_password"):
            current_user.password_hash = generate_password_hash(request.form["new_password"])
        db.session.commit()
        flash("Profile updated.", "ok")
        return redirect(url_for("settings"))
    return render_template("settings.html", user=current_user)

@app.route("/shared/<token>")
def shared(token):
    r = Report.query.filter_by(share_token=token).first_or_404()
    if not r.share_active():
        return render_template("shared_expired.html", report=r), 410
    try:
        db.session.add(ShareAccess(report_id=r.id, ip=(request.remote_addr or "")[:80],
                                   user_agent=(request.headers.get("User-Agent", "") or "")[:300]))
        db.session.commit()
    except Exception:
        pass
    from medical_nlp import clinician_summary
    return render_template("shared.html", report=r, data=r.analysis_json(),
                           entities=r.entities_json(),
                           clinician=clinician_summary(r.analysis_json(), r.entities_json(), r.file_name))

@app.route("/report/<int:report_id>/share", methods=["POST"])
@login_required
def share_report(report_id):
    from datetime import timedelta
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    body = {}
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        body = {}
    days = body.get("expiry_days", request.form.get("expiry_days", "7"))
    try:
        days = int(days)
    except Exception:
        days = 7
    if not r.share_token or r.share_revoked:
        r.share_token = secrets.token_urlsafe(16)
    r.share_revoked = False
    r.share_expires_at = (datetime.utcnow() + timedelta(days=days)) if days > 0 else None
    db.session.commit()
    views = ShareAccess.query.filter_by(report_id=r.id).count()
    return jsonify({"url": url_for("shared", token=r.share_token, _external=True),
                    "expires_at": r.share_expires_at.isoformat() if r.share_expires_at else None,
                    "views": views})

@app.route("/report/<int:report_id>/share", methods=["DELETE"])
@login_required
def revoke_share(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    r.share_revoked = True
    db.session.commit()
    return jsonify({"revoked": True})

@app.route("/report/<int:report_id>/share/views")
@login_required
def share_views(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    rows = ShareAccess.query.filter_by(report_id=r.id).order_by(ShareAccess.viewed_at.desc()).limit(50).all()
    def _mask(ip):
        ip = ip or ""
        if ":" in ip:  # IPv6
            return ip.split(":")[0] + ":****"
        parts = ip.split(".")
        return ".".join(parts[:3] + ["*"]) if len(parts) == 4 else "***"
    return jsonify({"active": r.share_active(),
                    "expires_at": r.share_expires_at.isoformat() if r.share_expires_at else None,
                    "views": [{"at": x.viewed_at.isoformat(), "ip": _mask(x.ip)} for x in rows]})

@app.route("/report/<int:report_id>/delete", methods=["POST"])
@login_required
def delete_report(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    try:
        if r.file_path and os.path.exists(r.file_path):
            os.remove(r.file_path)
    except Exception:
        pass
    DocumentChunk.query.filter_by(report_id=r.id).delete()
    ShareAccess.query.filter_by(report_id=r.id).delete()
    db.session.delete(r)
    db.session.commit()
    return redirect(url_for("dashboard"))

@app.route("/report/<int:report_id>/file")
@login_required
def report_file(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    if not r.file_path or not os.path.exists(r.file_path):
        abort(404)
    return send_file(r.file_path)

@app.route("/report/<int:report_id>/download.pdf")
@login_required
def download_pdf(report_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, KeepTogether, Image)
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as _canvas
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    data = r.analysis_json()
    import tempfile, os
    _tmp = tempfile.NamedTemporaryFile(prefix=f"MedoraAI_{r.id}_", suffix=".pdf",
                                       dir=app.config["UPLOAD_FOLDER"], delete=False)
    out = _tmp.name
    _tmp.close()

    is_img = r.file_type in ("png", "jpg", "jpeg", "webp")

    # ---------- styles ----------
    ss = getSampleStyleSheet()
    brand = colors.HexColor("#1e5eff")
    dark = colors.HexColor("#0f235a")
    light = colors.HexColor("#eef4ff")
    muted = colors.HexColor("#64748b")

    title_s = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                             fontSize=20, textColor=dark, spaceAfter=2)
    sub_s = ParagraphStyle("st", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=10.5, textColor=muted, spaceAfter=14)
    h_s = ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=13, textColor=brand, spaceBefore=14, spaceAfter=4,
                         borderPadding=(0,0,2,0))
    b_s = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                         fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=4)
    small_s = ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica-Oblique",
                             fontSize=8, textColor=muted, alignment=TA_LEFT)
    conf_s = ParagraphStyle("c", parent=b_s, fontName="Helvetica-Bold", textColor=brand)

    story = []

    def header_footer(cv, doc):
        cv.saveState()
        # header band
        cv.setFillColor(brand)
        cv.rect(0, A4[1] - 18 * mm, A4[0], 18 * mm, stroke=0, fill=1)
        cv.setFillColor(colors.white)
        cv.setFont("Helvetica-Bold", 13)
        cv.drawString(20 * mm, A4[1] - 11.5 * mm, "MEDORA AI")
        cv.setFont("Helvetica", 9)
        cv.drawRightString(A4[0] - 20 * mm, A4[1] - 11.5 * mm,
                           "Clinical Analysis Report")
        # footer
        cv.setStrokeColor(muted)
        cv.setLineWidth(0.5)
        cv.line(20 * mm, 16 * mm, A4[0] - 20 * mm, 16 * mm)
        cv.setFillColor(muted)
        cv.setFont("Helvetica", 7.5)
        cv.drawString(20 * mm, 11 * mm,
                      "AI-assisted analysis — not a replacement for professional medical advice.")
        cv.drawRightString(A4[0] - 20 * mm, 11 * mm, f"Page {doc.page}")
        cv.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=26 * mm, bottomMargin=22 * mm,
                            title=f"Medora AI Report {r.id}",
                            author="Medora AI")

    story.append(Paragraph("Clinical Analysis Report", title_s))
    story.append(Paragraph(
        f"<b>File:</b> {r.file_name}&nbsp;&nbsp;|&nbsp;&nbsp;<b>Patient:</b> "
        f"{r.detected_patient_name or 'Unknown'}&nbsp;&nbsp;|&nbsp;&nbsp;<b>Relationship:</b> "
        f"{r.relationship or 'Self'}&nbsp;&nbsp;|&nbsp;&nbsp;<b>Date:</b> "
        f"{r.created_at.strftime('%d %b %Y')}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=brand, spaceAfter=8))

    # confidence strip
    d = data.get("diagnosis", {}) or {}
    conf = round(float(d.get("confidence", 0) or 0) * 100)
    primary = d.get("primary") or "—"
    conf_tbl = Table([[Paragraph(f"<b>Primary Diagnosis</b>", b_s),
                       Paragraph(f"<b>{primary}</b>", conf_s)],
                      [Paragraph(f"<b>Confidence</b>", b_s),
                       Paragraph(f"<b>{conf}%</b>", conf_s)]],
                     colWidths=[34 * mm, 140 * mm])
    conf_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), light),
        ("BOX", (0,0), (-1,-1), 0.6, brand),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#c7d2fe")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(conf_tbl)
    story.append(Spacer(1, 8))

    # --- Uploaded document preview (image reports) ---
    if is_img and r.file_path and os.path.exists(r.file_path):
        try:
            img_reader = ImageReader(r.file_path)
            iw, ih = img_reader.getSize()
            max_w = A4[0] - 40 * mm
            scale = min(max_w / iw, 1.0)
            disp_w = iw * scale
            disp_h = ih * scale
            cap = min(disp_h, A4[1] - 130 * mm)
            scale2 = cap / disp_h
            disp_w *= scale2
            disp_h *= scale2
            story.append(Paragraph("Uploaded Document Preview", h_s))
            story.append(Image(r.file_path, width=disp_w, height=disp_h))
            story.append(Spacer(1, 6))
        except Exception as iex:
            print("PDF image embed failed:", str(iex)[:150])

    def section(title, body_paras, bullet=False):
        el = [Paragraph(title, h_s)]
        for p in body_paras:
            if bullet:
                el.append(Paragraph(f"•&nbsp;{p}", b_s))
            else:
                el.append(Paragraph(str(p), b_s))
        return KeepTogether(el)

    story.append(section("Executive Summary", [data.get("executiveSummary") or "—"]))
    story.append(section("Diagnosis", [
        f"Primary: {primary}",
        f"Confidence: {conf}%",
        f"Differential: {', '.join(d.get('differential', []) or []) or '—'}",
    ]))
    story.append(section("Pathophysiology", [data.get("pathophysiology") or "—"]))
    story.append(section("Treatment Plan", [
        f"Medications: {', '.join(m.get('name','') for m in data.get('treatmentPlan',{}).get('medications',[])) or '—'}",
        f"Therapy: {', '.join(data.get('treatmentPlan',{}).get('therapyOptions',[])) or '—'}",
    ]))
    story.append(section("Lifestyle Recommendations", data.get("lifestyleRecommendations", []) or ["—"], bullet=True))
    story.append(section("Follow-up Actions", data.get("followUpActions", []) or ["—"], bullet=True))
    story.append(section("Emergency Triggers", data.get("emergencyTriggers", []) or ["—"], bullet=True))
    story.append(section("Risk Factors", [
        f"{rf.get('factor','')}: {rf.get('explanation','')}" for rf in data.get("riskFactors", [])
    ] or ["—"]))
    story.append(section("Current Status", [
        f"As of: {(data.get('currentStatus',{}) or {}).get('asOfDate','—')}",
        (data.get('currentStatus',{}) or {}).get('summary','—'),
        f"Trajectory: {(data.get('currentStatus',{}) or {}).get('trajectory','—')}",
    ]))

    # extracted values + problem list (NLP)
    try:
        from medical_nlp import clinician_summary
        cs = clinician_summary(data, r.entities_json(), r.file_name)
        labs_rows = [[Paragraph("<b>Test</b>", b_s), Paragraph("<b>Value</b>", b_s),
                      Paragraph("<b>Unit</b>", b_s), Paragraph("<b>Flag</b>", b_s)]]
        for l in cs.get("key_labs", []):
            labs_rows.append([Paragraph(str(l.get("test","")), b_s),
                              Paragraph(str(l.get("value","")), b_s),
                              Paragraph(str(l.get("unit","")), b_s),
                              Paragraph(str(l.get("flag","")), b_s)])
        labs_tbl = Table(labs_rows, colWidths=[50 * mm, 30 * mm, 20 * mm, 20 * mm],
                         repeatRows=1)
        labs_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), brand),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, light]),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c7d2fe")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(KeepTogether([Paragraph("Key Extracted Values (NLP)", h_s), labs_tbl]))
        pl = cs.get("problem_list", [])
        if pl:
            story.append(KeepTogether([Paragraph("Clinician Problem List", h_s)] +
                                      [Paragraph(f"• {p}", b_s) for p in pl]))
    except Exception:
        pass

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.6, color=muted, spaceAfter=6))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is AI-generated for informational purposes only and is "
        "not a medical diagnosis or treatment recommendation. Always consult a qualified healthcare "
        "professional before making medical decisions.", small_s))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return send_file(out, as_attachment=True, download_name=f"Medora_AI_Report_{r.id}.pdf")

# ---------------- JSON APIs (parity with original /api/*) ----------------
@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

@app.route("/api/newsletter", methods=["POST"])
def api_newsletter():
    body = request.get_json(force=True, silent=True) or {}
    email = (body.get("email") or "").strip().lower()[:254]
    if not email or "@" not in email or "." not in email:
        return jsonify({"error": "Valid email required"}), 400
    try:
        with open(os.path.join(app.config["UPLOAD_FOLDER"], "newsletter.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()},{email}\n")
    except Exception:
        pass
    return jsonify({"ok": True})

@app.route("/api/report_status/<int:report_id>")
@login_required
def api_status(report_id):
    r = Report.query.get_or_404(report_id)
    if r.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    err = ""
    if r.status == "failed":
        try:
            err = (r.analysis_json().get("executiveSummary") or "")[:300]
        except Exception:
            pass
    return jsonify({"id": r.id, "status": r.status, "progress": r.progress, "error": err,
                    "redirect": url_for("report_detail", report_id=r.id) if r.status == "completed" else None})

@app.route("/api/translate", methods=["POST"])
@login_required
def api_translate():
    body = request.get_json(force=True) or {}
    analysis = body.get("analysis") or {}
    lang = body.get("targetLanguage") or body.get("language") or body.get("lang") or "Hindi"
    # allow passing report id instead
    if not analysis and body.get("report_id"):
        r = db.session.get(Report, body["report_id"])
        if r:
            analysis = r.analysis_json()
    try:
        out = translate_analysis(analysis, lang)
    except Exception as ex:
        return jsonify({"error": f"Translation failed: {ex}"}), 503
    return jsonify({"text": json.dumps(out), "data": out})

@app.route("/api/doctors", methods=["POST"])
@login_required
def api_doctors():
    body = request.get_json(force=True)
    try:
        return jsonify(search_doctors(body.get("diagnosis", ""), body.get("latitude"), body.get("longitude")))
    except Exception as ex:
        return jsonify({"error": f"Referral guidance failed: {ex}"}), 503

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    body = request.get_json(force=True)
    msg = body.get("message", "")
    history = body.get("history", [])
    ctx = body.get("reportContext", "")
    snippets = []
    rid = body.get("report_id")
    try:
        from rag_engine import retrieve
        if rid:
            r = db.session.get(Report, rid)
            if r and (r.user_id == current_user.id or current_user.role == "admin"):
                snippets = retrieve(db, DocumentChunk, msg, report_id=r.id, top_k=5)
                if not snippets:  # fall back to whole analysis JSON
                    ctx = json.dumps(r.analysis_json())[:4000]
        else:
            snippets = retrieve(db, DocumentChunk, msg, user_id=current_user.id, top_k=5)
    except Exception as ex:
        print("RAG retrieve failed:", str(ex)[:200])
    try:
        text = chat_answer(msg, history, ctx, rag_snippets=snippets)
    except Exception as ex:
        return jsonify({"error": f"Chat failed: {ex}"}), 503
    src = [{"report_id": s["report_id"], "chunk": s["chunk"], "score": s["score"],
            "preview": s["text"][:220]} for s in snippets]
    return jsonify({"text": text, "sources": src})

@app.route("/api/rag_status")
@login_required
def api_rag_status():
    from rag_engine import ollama_ok, GEN_MODEL, EMBED_MODEL, OLLAMA_HOST, _available_models
    from ai_engine import GEMINI_KEY, GEMINI_MODELS, USE_OLLAMA
    n_chunks = DocumentChunk.query.filter_by(user_id=current_user.id).count()
    n_reports = Report.query.filter_by(user_id=current_user.id).count()
    return jsonify({"ollama_reachable": ollama_ok(), "host": OLLAMA_HOST,
                    "ollama_models": _available_models(), "use_ollama": USE_OLLAMA,
                    "gen_model": GEN_MODEL, "embed_model": EMBED_MODEL,
                    "gemini_configured": bool(GEMINI_KEY), "gemini_models": GEMINI_MODELS,
                    "my_chunks": n_chunks, "my_reports": n_reports})

@app.route("/api/explain", methods=["POST"])
@login_required
def api_explain():
    """Feature 2 — simplify any medical term/finding into plain language."""
    body = request.get_json(force=True)
    term = (body.get("term") or "")[:300]
    if not term:
        return jsonify({"error": "No term given"}), 400
    lang = body.get("language", "simple English")
    try:
        from ai_engine import GEMINI_KEY, _gemini_generate, USE_OLLAMA
        from rag_engine import generate
        prompt = (f"Explain this medical term/finding in {lang} for a patient with no medical background: '{term}'. "
            "Use 2-4 short sentences, one everyday analogy, and one line on why a doctor might care. No diagnosis.")
        txt = ""
        # Gemini first (fast, configured) so explain doesn't hang on a slow/unreachable Ollama model.
        if GEMINI_KEY:
            try:
                txt = _gemini_generate(prompt, max_tokens=400)
            except Exception as e:
                print("Gemini explain failed:", str(e)[:160])
        if not txt.strip() and USE_OLLAMA:
            try:
                txt = generate(prompt, system="You are a patient-education assistant. Simple words only.",
                               temperature=0.3, max_tokens=400)
            except Exception as e:
                print("Ollama explain failed:", str(e)[:160])
        if not txt.strip():
            raise ValueError("Empty model reply")
        return jsonify({"term": term, "explanation": txt})
    except Exception as ex:
        return jsonify({"term": term, "error": f"Explanation failed: {ex}"}), 503

@app.route("/api/reindex", methods=["POST"])
@login_required
@admin_required
def api_reindex():
    from rag_engine import ingest_chunks
    from medical_nlp import extract_entities
    from ai_engine import _parse_confidence, _repair_confidence
    from rag_engine import build_context
    from rag_engine import retrieve as _retr
    done, failed = 0, []
    for r in Report.query.all():
        try:
            text = extract_text_from_file(r.file_path, r.file_name)
            r.entities = json.dumps(extract_entities(text, r.file_name))
            n = ingest_chunks(db, DocumentChunk, r.id, r.user_id, text, r.file_name)
            # repair legacy 0.0 confidences with a real model rating
            try:
                a = r.analysis_json()
                d = (a.get("diagnosis") or {})
                if d.get("primary") and not _parse_confidence(d.get("confidence")):
                    snips = _retr(db, DocumentChunk, d["primary"], report_id=r.id, top_k=4)
                    d["confidence"] = _repair_confidence(d["primary"], build_context(snips) or text[:2500])
                    a["diagnosis"] = d
                    r.analysis = json.dumps(a)
            except Exception as rex:
                print("conf repair skipped:", str(rex)[:150])
            db.session.commit()
            done += 1
        except Exception as ex:
            failed.append({"id": r.id, "err": str(ex)[:200]})
    return jsonify({"reindexed": done, "failed": failed})

# ---------------- Admin ----------------
@app.route("/adminpanel/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email", "").lower()
        pw = request.form.get("password", "")
        u = User.query.filter_by(email=email, role="admin").first()
        if u and check_password_hash(u.password_hash, pw):
            login_user(u)
            return redirect(url_for("admin_dash"))
        flash("Invalid admin credentials.", "error")
    return render_template("admin_login.html")

@app.route("/adminpanel/dashboard")
@login_required
@admin_required
def admin_dash():
    return render_template("admin_dashboard.html",
                           users=User.query.count(), reports=Report.query.count(),
                           recent=Report.query.order_by(Report.created_at.desc()).limit(8).all(),
                           errors=ErrorLog.query.order_by(ErrorLog.created_at.desc()).limit(8).all())

@app.route("/adminpanel/reports")
@login_required
@admin_required
def admin_reports():
    reports = Report.query.order_by(Report.created_at.desc()).limit(100).all()
    return render_template("admin_reports.html", reports=reports)

@app.route("/adminpanel/users")
@login_required
@admin_required
def admin_users():
    return render_template("admin_users.html", users=User.query.order_by(User.created_at.desc()).all())

@app.route("/adminpanel/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(user_id):
    u = User.query.get_or_404(user_id)
    if u.id == current_user.id:
        return jsonify({"error": "You cannot delete your own admin account."}), 400
    # cascade: reports -> files, chunks, share logs, then the user
    for r in Report.query.filter_by(user_id=u.id).all():
        try:
            if r.file_path and os.path.exists(r.file_path):
                os.remove(r.file_path)
        except Exception:
            pass
        DocumentChunk.query.filter_by(report_id=r.id).delete()
        ShareAccess.query.filter_by(report_id=r.id).delete()
        db.session.delete(r)
    DocumentChunk.query.filter_by(user_id=u.id).delete()
    db.session.delete(u)
    db.session.commit()
    return jsonify({"deleted": user_id})

@app.route("/adminpanel/settings", methods=["GET", "POST"])
@login_required
@admin_required
def admin_settings():
    cfg = AppConfig.query.first()
    if request.method == "POST":
        cfg.landing_headline = request.form.get("landing_headline", cfg.landing_headline)
        cfg.maintenance_mode = bool(request.form.get("maintenance_mode"))
        db.session.commit()
        flash("Settings saved.", "ok")
    return render_template("admin_settings.html", cfg=cfg)

@app.route("/adminpanel/errors")
@login_required
@admin_required
def admin_errors():
    return render_template("admin_errors.html", errors=ErrorLog.query.order_by(ErrorLog.created_at.desc()).limit(200).all())

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_ENV") != "production", port=5000)
