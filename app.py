import os
import io
import base64
import re
import json
import datetime
import hashlib
import secrets
import time
from functools import wraps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Voice Libraries ──
try:
    import speech_recognition as sr
    STT_SUPPORT = True
except ImportError:
    STT_SUPPORT = False

try:
    from gtts import gTTS
    TTS_SUPPORT = True
except ImportError:
    TTS_SUPPORT = False

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ══════════════════════════════════════
#  ENVIRONMENT VARIABLES
# ══════════════════════════════════════
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set!")

genai.configure(api_key=GOOGLE_API_KEY)

# ── Gemini Models ──
gemini_flash = genai.GenerativeModel('gemini-2.0-flash-lite')
gemini_check = genai.GenerativeModel('gemini-2.0-flash-lite')

# ── File Paths ──
HISTORY_FILE   = "history.json"
BLACKLIST_FILE = "blacklist.json"
PDF_FOLDER     = "textbooks"
STATS_FILE     = "stats.json"
RATE_FILE      = "rate_limits.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# ══════════════════════════════════════
#  LANGUAGE INSTRUCTIONS
# ══════════════════════════════════════
LANG_INSTRUCTIONS = {
    "si": {
        "instruction": "සිංහල භාෂාවෙන් පමණක් පිළිතුරු දෙන්න. පියවරෙන් පියවර සිංහලෙන් පැහැදිලි කරන්න.",
        "cross_check": "නිවැරදි නම් ONLY '✅ නිවැරදියි' කියා reply කරන්න. වැරදි නම් නිවැරදි පිළිතුර සිංහලෙන් දෙන්න.",
        "correct_word": "✅ නිවැරදියි",
        "gtts_lang": "si"
    },
    "en": {
        "instruction": "Answer ONLY in English. Explain step by step clearly in English.",
        "cross_check": "If correct reply ONLY '✅ Correct'. If wrong, give the correct answer in English.",
        "correct_word": "✅ Correct",
        "gtts_lang": "en"
    },
    "ta": {
        "instruction": "தமிழ் மொழியில் மட்டும் பதில் சொல்லுங்கள். படிப்படியாக தமிழில் விளக்குங்கள்.",
        "cross_check": "சரியாக இருந்தால் ONLY '✅ சரியானது' என்று reply கொடுங்கள். தவறாக இருந்தால் சரியான பதிலை தமிழில் தாருங்கள்.",
        "correct_word": "✅ சரியானது",
        "gtts_lang": "ta"
    }
}

# ══════════════════════════════════════
#  HELPERS — JSON + LANGUAGE
# ══════════════════════════════════════
def detect_language(text):
    sinhala_count = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    tamil_count   = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if sinhala_count > 2: return "si"
    elif tamil_count > 2: return "ta"
    else: return "en"

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] save_json failed for {path}: {e}")

# ══════════════════════════════════════
#  SECURITY — BAD WORDS + BLACKLIST
# ══════════════════════════════════════
BAD_WORDS = [
    "fuck", "shit", "bitch", "ass", "bastard", "cunt",
    "හුත්ත", "පුකේ", "මෝඩ", "පට්ට", "හුකන"
]

def contains_bad_words(text):
    text_lower = text.lower()
    return any(w in text_lower for w in BAD_WORDS)

def is_blacklisted(user_name):
    blacklist = load_json(BLACKLIST_FILE, {})
    key = user_name.lower().strip()
    if key in blacklist:
        ban_time = datetime.datetime.fromisoformat(blacklist[key])
        if datetime.datetime.now() < ban_time:
            remaining = max(0, (ban_time - datetime.datetime.now()).seconds // 60)
            return True, remaining
        else:
            del blacklist[key]
            save_json(BLACKLIST_FILE, blacklist)
    return False, 0

def add_to_blacklist(user_name, hours=2):
    blacklist = load_json(BLACKLIST_FILE, {})
    ban_until = datetime.datetime.now() + datetime.timedelta(hours=hours)
    blacklist[user_name.lower().strip()] = ban_until.isoformat()
    save_json(BLACKLIST_FILE, blacklist)

def remove_from_blacklist(user_name):
    blacklist = load_json(BLACKLIST_FILE, {})
    key = user_name.lower().strip()
    if key in blacklist:
        del blacklist[key]
        save_json(BLACKLIST_FILE, blacklist)
        return True
    return False

# ══════════════════════════════════════
#  RATE LIMITING  (per user: 30 req/min)
# ══════════════════════════════════════
RATE_LIMIT_MAX    = 30   # requests
RATE_LIMIT_WINDOW = 60   # seconds

def check_rate_limit(user_name):
    """Returns True if request is allowed, False if rate limited."""
    rates = load_json(RATE_FILE, {})
    key   = user_name.lower().strip()
    now   = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    if key not in rates:
        rates[key] = []

    # Remove old entries outside the window
    rates[key] = [t for t in rates[key] if t > window_start]

    if len(rates[key]) >= RATE_LIMIT_MAX:
        save_json(RATE_FILE, rates)
        return False

    rates[key].append(now)
    save_json(RATE_FILE, rates)
    return True

# ══════════════════════════════════════
#  ADMIN AUTH DECORATOR
# ══════════════════════════════════════
def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        pw = request.headers.get("X-Admin-Password", "")
        if pw != ADMIN_PASSWORD:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════
#  STATS + HISTORY
# ══════════════════════════════════════
def update_stats(subject, language="si"):
    stats = load_json(STATS_FILE, {"total": 0, "subjects": {}, "languages": {}})
    stats["total"] = stats.get("total", 0) + 1
    stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
    stats["languages"][language] = stats["languages"].get(language, 0) + 1
    save_json(STATS_FILE, stats)

def save_history(user_name, grade, subject, question, answer, language="si"):
    history = load_json(HISTORY_FILE, [])
    history.append({
        "user":     user_name,
        "grade":    grade,
        "subject":  subject,
        "question": question,
        "answer":   answer[:500],
        "language": language,
        "time":     datetime.datetime.now().isoformat()
    })
    save_json(HISTORY_FILE, history[-1000:])   # keep last 1000

# ══════════════════════════════════════
#  RAG — PDF CONTEXT
# ══════════════════════════════════════
def get_rag_context(question, grade):
    if not PDF_SUPPORT:
        return ""
    context_parts = []
    keywords = [kw for kw in question.lower().split() if len(kw) > 3]
    try:
        pdf_files = os.listdir(PDF_FOLDER)
    except:
        return ""
    for filename in pdf_files:
        if not filename.endswith(".pdf"):
            continue
        try:
            doc = fitz.open(os.path.join(PDF_FOLDER, filename))
            for page in doc:
                text = page.get_text()
                if any(kw in text.lower() for kw in keywords):
                    context_parts.append(f"[{filename}]\n{text[:1000]}")
                    break
            doc.close()
        except:
            pass
        if len(context_parts) >= 3:
            break
    return "\n\n".join(context_parts)

# ══════════════════════════════════════
#  ROUTES — HEALTH
# ══════════════════════════════════════
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "stt":    STT_SUPPORT,
        "tts":    TTS_SUPPORT,
        "pdf":    PDF_SUPPORT,
        "time":   datetime.datetime.now().isoformat()
    })

# ══════════════════════════════════════
#  ROUTES — TTS
# ══════════════════════════════════════
@app.route('/tts', methods=['POST'])
def text_to_speech():
    if not TTS_SUPPORT:
        return jsonify({"status": "error", "message": "TTS not available"}), 503
    try:
        data      = request.json or {}
        text      = str(data.get('text', ''))[:1000]
        language  = data.get('language', 'si')
        if language not in LANG_INSTRUCTIONS:
            language = 'si'
        gtts_lang = LANG_INSTRUCTIONS[language]['gtts_lang']
        clean_text = re.sub(r'[*_#`\[\]()~>]', '', text)
        tts = gTTS(text=clean_text, lang=gtts_lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return send_file(audio_buffer, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ══════════════════════════════════════
#  ROUTES — SOLVE  (main AI endpoint)
# ══════════════════════════════════════
@app.route('/solve', methods=['POST'])
def solve_math():
    try:
        # ── Parse request ──
        if request.is_json:
            data = request.json or {}
        else:
            data = request.form

        user_name  = str(data.get('name', 'User')).strip()[:50]
        user_grade = str(data.get('grade', 'N/A')).strip()[:3]
        subject    = str(data.get('subject', 'Mathematics')).strip()[:50]
        question   = str(data.get('question', '')).strip()[:2000]
        language   = str(data.get('language', 'si')).strip()
        is_summary = data.get('is_summary_request', False)
        image_file = request.files.get('image') if not request.is_json else None

        if not question:
            return jsonify({"status": "error", "message": "Question is empty"}), 400

        if language not in LANG_INSTRUCTIONS:
            language = detect_language(question)
        lang_cfg = LANG_INSTRUCTIONS[language]

        # ── Summary (quick voice response) ──
        if is_summary:
            summary_prompt = f"Summarize this answer into 2 simple sentences in {lang_cfg['gtts_lang']}. No markdown."
            response = gemini_flash.generate_content([summary_prompt, question])
            return jsonify({
                "status":     "success",
                "answer":     response.text.strip(),
                "is_summary": True
            })

        # ── Rate limit check ──
        if not check_rate_limit(user_name):
            return jsonify({
                "status":  "error",
                "message": "Too many requests. Please wait a moment."
            }), 429

        # ── Blacklist check ──
        banned, remaining = is_blacklisted(user_name)
        if banned:
            return jsonify({
                "status":  "banned",
                "message": f"You are blocked for {remaining} more minutes."
            })

        # ── Bad words check ──
        if contains_bad_words(question):
            add_to_blacklist(user_name, hours=2)
            return jsonify({
                "status":  "banned",
                "message": "Inappropriate language detected. You have been blocked for 2 hours."
            })

        # ── Validate image ──
        image_content = None
        if image_file:
            allowed_types = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}
            if image_file.mimetype not in allowed_types:
                return jsonify({"status": "error", "message": "Invalid image type"}), 400
            img_data = image_file.read(5 * 1024 * 1024)   # max 5MB
            image_content = {"mime_type": image_file.mimetype, "data": img_data}

        # ── Build AI prompt ──
        rag_context = get_rag_context(question, user_grade)
        instruction = (
            f"You are an expert {subject} teacher for Grade {user_grade} students in Sri Lanka. "
            f"{lang_cfg['instruction']} "
            "Solve step by step clearly. Use examples where helpful. "
            "If a graph or chart would help understanding, produce matplotlib code between [GRAPH_START] and [GRAPH_END]. "
            "Do NOT include any harmful, political, or off-topic content."
        )

        content_parts = [instruction, f"Question: {question}"]
        if rag_context:
            content_parts.append(f"\nTextbook Context:\n{rag_context}")
        if image_content:
            content_parts.append(image_content)

        response     = gemini_flash.generate_content(content_parts)
        fin
