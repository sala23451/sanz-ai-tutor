import os
import io
import base64
import re
import json
import datetime
import hashlib
import secrets
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi import FastAPI, Form, File, UploadFile, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Scheduler ──
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_SUPPORT = True
except ImportError:
    SCHEDULER_SUPPORT = False

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

# ── Cloudinary ──
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    CLOUDINARY_SUPPORT = True
except ImportError:
    CLOUDINARY_SUPPORT = False

# ── Pinecone ──
try:
    from pinecone import Pinecone
    PINECONE_SUPPORT = True
except ImportError:
    PINECONE_SUPPORT = False

# ══════════════════════════════════════════
# ── App Setup ──
# ══════════════════════════════════════════
app = FastAPI(title="Sanz AI Tutor", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set!")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel('gemini-2.5-flash')
gemini_check = genai.GenerativeModel('gemini-2.5-flash-lite')

# ── File Paths ──
HISTORY_FILE      = "history.json"
BLACKLIST_FILE    = "blacklist.json"
PDF_FOLDER        = "textbooks"
STATS_FILE        = "stats.json"
CACHE_FILE        = "semantic_cache.json"
PROGRESS_FILE     = "progress.json"
QUIZ_FILE         = "quiz_sessions.json"
LEADERBOARD_FILE  = "leaderboard.json"
API_USAGE_FILE    = "api_usage.json"
PARENTS_FILE      = "parents.json"
CHILDREN_FILE     = "children.json"
USER_TOKENS_FILE  = "user_tokens.json"
EMAIL_LOG_FILE    = "email_log.json"
USERS_FILE        = "users.json"

# ── Email Config ──
EMAIL_HOST     = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT     = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USER     = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASS", "")

# ── Cache Config ──
CACHE_SIMILARITY_THRESHOLD = 0.82
CACHE_MAX_SIZE             = 200
CACHE_TTL_HOURS            = 48

# ── API Limits ──
API_LIMITS = {
    "gemini-2.5-flash":      500,
    "gemini-2.5-flash-lite": 1000,
}

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# ── Cloudinary Setup ──
if CLOUDINARY_SUPPORT:
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET")
    )

# ── Pinecone Setup ──
pinecone_index = None
if PINECONE_SUPPORT:
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pinecone_index = pc.Index("sanz-tutor")
    except Exception as e:
        print(f"Pinecone setup error: {e}")

# ══════════════════════════════════════════
# ── Language Instructions ──
# ══════════════════════════════════════════
SOCRATIC_INSTRUCTIONS = {
    "si": """ඔබ ඉවසිලිවන්ත Socratic ගණිත ගුරුවරයෙකි. සිංහල භාෂාවෙන් පමණක් පිළිතුරු දෙන්න.
නීති:
1. කිසිවිටෙකත් සෘජු පිළිතුර හෝ සම්පූර්ණ විසඳුම දෙන්න එපා.
2. එක් වරකට එක් ප්‍රශ්නයක් හෝ hint එකක් පමණයි.
3. වැරදි නම් ශිෂ්‍යයාට වැරැද්ද තෝරාගන්නට ප්‍රශ්නයක් අහන්න.
4. සෑම පියවරකටම ශිෂ්‍යයාගේ reasoning check කරන්න.
Tone: "හොඳ ආරම්භයක්!", "ඔබ නිවැරදි මාර්ගයේ!", "නැවත බලමු" වැනි වාක්‍ය use කරන්න.""",
    "en": """You are a supportive, patient Socratic Math Tutor. Answer ONLY in English.
Rules:
1. Never give the final answer or full step-by-step solution.
2. Only ask ONE question or give ONE hint per turn.
3. If student makes a mistake, ask a question to help them spot their error.
4. Check student's reasoning before moving to the next step.
Tone: Use phrases like "Great start!", "You're on the right track!", "Let's look at that again." """,
    "ta": """நீங்கள் ஒரு பொறுமையான Socratic கணித ஆசிரியர். தமிழில் மட்டும் பதில் சொல்லுங்கள்.
விதிகள்:
1. இறுதி விடையை அல்லது முழு தீர்வையும் கொடுக்காதீர்கள்.
2. ஒரு முறைக்கு ஒரே ஒரு கேள்வி அல்லது hint மட்டும்.
3. தவறு இருந்தால் மாணவர் தாமே கண்டுபிடிக்க கேள்வி கேளுங்கள்.
Tone: "நல்ல தொடக்கம்!", "சரியான பாதையில் இருக்கிறீர்கள்!" போன்ற வார்த்தைகள் பயன்படுத்துங்கள்."""
}

LANG_INSTRUCTIONS = {
    "si": {
        "instruction":  "සිංහල භාෂාවෙන් පමණක් පිළිතුරු දෙන්න. පියවරෙන් පියවර සිංහලෙන් පැහැදිලි කරන්න.",
        "socratic":     SOCRATIC_INSTRUCTIONS["si"],
        "cross_check":  "නිවැරදි නම් ONLY '✅ නිවැරදියි' කියා reply කරන්න. වැරදි නම් නිවැරදි පිළිතුර සිංහලෙන් දෙන්න.",
        "correct_word": "✅ නිවැරදියි",
        "gtts_lang":    "si"
    },
    "en": {
        "instruction":  "Answer ONLY in English. Explain step by step clearly in English.",
        "socratic":     SOCRATIC_INSTRUCTIONS["en"],
        "cross_check":  "If correct reply ONLY '✅ Correct'. If wrong, give the correct answer in English.",
        "correct_word": "✅ Correct",
        "gtts_lang":    "en"
    },
    "ta": {
        "instruction":  "தமிழ் மொழியில் மட்டும் பதில் சொல்லுங்கள். படிப்படியாக தமிழில் விளக்குங்கள்.",
        "socratic":     SOCRATIC_INSTRUCTIONS["ta"],
        "cross_check":  "சரியாக இருந்தால் ONLY '✅ சரியானது' என்று reply கொடுங்கள். தவறாக இருந்தால் சரியான பதிலை தமிழில் தாருங்கள்.",
        "correct_word": "✅ சரியானது",
        "gtts_lang":    "ta"
    }
}

# ══════════════════════════════════════════
# ── Pydantic Models ──
# ══════════════════════════════════════════
class LoginRequest(BaseModel):
    name: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    password: str
    role: str = "user"

class ParentRegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    phone: str = ""

class ParentLoginRequest(BaseModel):
    email: str
    password: str

class AddChildRequest(BaseModel):
    parent_email: str
    name: str
    birthday: str
    grade: str
    school: str
    location: str
    child_email: str = ""

class UpdateChildRequest(BaseModel):
    child_id: str
    parent_email: str
    name: str = ""
    grade: str = ""
    school: str = ""
    location: str = ""

class DeleteChildRequest(BaseModel):
    child_id: str
    parent_email: str

class QuizStartRequest(BaseModel):
    user_name: str
    grade: str
    subject: str
    language: str = "si"

class QuizAnswerRequest(BaseModel):
    session_id: str
    user_name: str
    user_answer: str

class BlacklistRemoveRequest(BaseModel):
    user_name: str

class TTSRequest(BaseModel):
    text: str
    language: str = "si"

class ImageGenRequest(BaseModel):
    prompt: str
    subject: str = "General"
    language: str = "en"

# ══════════════════════════════════════════
# ── Helper Functions ──
# ══════════════════════════════════════════
def detect_language(text: str) -> str:
    sinhala_count = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    tamil_count   = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if sinhala_count > 2:   return "si"
    elif tamil_count > 2:   return "ta"
    return "en"

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_admin(x_admin_password: str = "") -> bool:
    return x_admin_password == ADMIN_PASSWORD

def calculate_age(birthday: str) -> int:
    try:
        dob   = datetime.datetime.strptime(birthday, "%Y-%m-%d")
        today = datetime.datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return 0

def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)

# ── Bad Words ──
BAD_WORDS = [
    "fuck","shit","bitch","ass","bastard","dick","cock","pussy","cunt","whore","slut",
    "nigga","faggot","moron","dumbass","retard","loser",
    "හුත්ත","පුකේ","මෝඩ","හරක","කුකු","කපන්","බේපන්","ගනින්","හූකර","හාමු",
    "පට්ට හරක","කෙල්ල","කොල්ල","ගොන්","වාහල","හුකන්","හූකන","පිස්සු","උන්","බල්ලො",
    "හූලං","හූකල","ගෑනු","ගෑනි",
    "hutta","puke","moda","harak","kuku","kapan","bepan","ganin","hookar","pissu",
    "marana","ballo","hulan","wahala","goni","kotiya"
]

RAGE_WORDS = [
    "stupid","idiot","useless","hate","terrible","worst",
    "මෝඩ","නිකම්","වැඩක් නෑ","අකාරයි","හොඳ නෑ"
]

def contains_bad_words(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in BAD_WORDS)

def is_rage(text: str) -> bool:
    text_lower = text.lower()
    return sum(1 for w in RAGE_WORDS if w in text_lower) >= 2

def is_blacklisted(user_name: str):
    blacklist = load_json(BLACKLIST_FILE, {})
    if user_name in blacklist:
        ban_time  = datetime.datetime.fromisoformat(blacklist[user_name])
        if datetime.datetime.now() < ban_time:
            remaining = (ban_time - datetime.datetime.now()).seconds // 60
            return True, remaining
        else:
            del blacklist[user_name]
            save_json(BLACKLIST_FILE, blacklist)
    return False, 0

def add_to_blacklist(user_name: str):
    blacklist             = load_json(BLACKLIST_FILE, {})
    ban_until             = datetime.datetime.now() + datetime.timedelta(hours=2)
    blacklist[user_name]  = ban_until.isoformat()
    save_json(BLACKLIST_FILE, blacklist)

# ══════════════════════════════════════════
# ── Token Tracking ──
# ══════════════════════════════════════════
def track_user_tokens(user_name: str, input_tokens: int, output_tokens: int):
    tokens = load_json(USER_TOKENS_FILE, {})
    today  = datetime.date.today().isoformat()
    week   = datetime.date.today().strftime("%Y-W%W")
    month  = datetime.date.today().strftime("%Y-%m")

    if user_name not in tokens:
        tokens[user_name] = {
            "total_input": 0, "total_output": 0, "total_calls": 0,
            "today": {}, "weekly": {}, "monthly": {}
        }

    u = tokens[user_name]
    u["total_input"]  += input_tokens
    u["total_output"] += output_tokens
    u["total_calls"]  += 1

    u.setdefault("today",   {}).setdefault(today,  {"input": 0, "output": 0, "calls": 0})
    u.setdefault("weekly",  {}).setdefault(week,   {"input": 0, "output": 0, "calls": 0})
    u.setdefault("monthly", {}).setdefault(month,  {"input": 0, "output": 0, "calls": 0})

    for period, key in [(u["today"], today), (u["weekly"], week), (u["monthly"], month)]:
        period[key]["input"]  += input_tokens
        period[key]["output"] += output_tokens
        period[key]["calls"]  += 1

    if len(u["today"]) > 30:
        oldest = sorted(u["today"].keys())[0]
        del u["today"][oldest]

    tokens[user_name] = u
    save_json(USER_TOKENS_FILE, tokens)

def track_api_call(model: str, call_type: str = "solve"):
    usage = load_json(API_USAGE_FILE, {})
    today = datetime.date.today().isoformat()
    if today not in usage:
        usage[today] = {"total": 0, "models": {}, "by_type": {}}
    usage[today]["total"] += 1
    usage[today]["models"][model]    = usage[today]["models"].get(model, 0) + 1
    usage[today]["by_type"][call_type] = usage[today]["by_type"].get(call_type, 0) + 1
    keys = sorted(usage.keys())
    if len(keys) > 7:
        for old_key in keys[:-7]:
            del usage[old_key]
    save_json(API_USAGE_FILE, usage)

def get_api_usage() -> dict:
    usage      = load_json(API_USAGE_FILE, {})
    today      = datetime.date.today().isoformat()
    today_data = usage.get(today, {"total": 0, "models": {}, "by_type": {}})
    remaining  = {}
    for model, limit in API_LIMITS.items():
        used           = today_data.get("models", {}).get(model, 0)
        remaining[model] = {"used": used, "limit": limit, "remaining": max(0, limit - used)}
    return {
        "today":             today,
        "total_calls_today": today_data["total"],
        "models":            remaining,
        "by_type":           today_data.get("by_type", {}),
        "total_remaining":   sum(r["remaining"] for r in remaining.values()),
        "history":           {k: v["total"] for k, v in usage.items()}
    }

# ══════════════════════════════════════════
# ── Semantic Cache ──
# ══════════════════════════════════════════
def simple_vectorize(text: str) -> dict:
    words = re.findall(r"\w+", text.lower())
    vec   = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    total = sum(vec.values()) or 1
    return {k: v / total for k, v in vec.items()}

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    if not common: return 0.0
    dot   = sum(vec1[w] * vec2[w] for w in common)
    norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
    norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

def cache_lookup(question: str, subject: str, language: str):
    cache = load_json(CACHE_FILE, [])
    now   = datetime.datetime.now()
    q_vec = simple_vectorize(question)
    valid_cache = []
    for entry in cache:
        try:
            if (now - datetime.datetime.fromisoformat(entry["time"])).total_seconds() < CACHE_TTL_HOURS * 3600:
                valid_cache.append(entry)
        except:
            pass
    if len(valid_cache) != len(cache):
        save_json(CACHE_FILE, valid_cache)
    best_score, best_entry = 0.0, None
    for entry in valid_cache:
        if entry.get("subject") != subject or entry.get("language") != language:
            continue
        score = cosine_similarity(q_vec, simple_vectorize(entry["question"]))
        if score > best_score:
            best_score, best_entry = score, entry
    if best_score >= CACHE_SIMILARITY_THRESHOLD and best_entry:
        return best_entry, best_score
    return None, 0.0

def cache_store(question: str, subject: str, language: str, answer: str, graph_url, verified: bool):
    cache = load_json(CACHE_FILE, [])
    q_vec = simple_vectorize(question)
    for entry in cache:
        if entry.get("subject") == subject and entry.get("language") == language:
            if cosine_similarity(q_vec, simple_vectorize(entry["question"])) >= CACHE_SIMILARITY_THRESHOLD:
                return
    cache.append({
        "question": question, "subject": subject, "language": language,
        "answer": answer, "graph_url": graph_url, "verified": verified,
        "time": datetime.datetime.now().isoformat()
    })
    if len(cache) > CACHE_MAX_SIZE:
        cache = cache[-CACHE_MAX_SIZE:]
    save_json(CACHE_FILE, cache)

# ══════════════════════════════════════════
# ── RAG System ──
# ══════════════════════════════════════════
def pinecone_embed(text: str) -> list:
    try:
        pc     = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        result = pc.inference.embed(model="multilingual-e5-large", inputs=[text[:500]],
                                    parameters={"input_type": "passage"})
        return result[0].values
    except Exception as e:
        print(f"Embed error: {e}")
        return [0.0] * 1024

def pinecone_search(question: str, grade: str = "", top_k: int = 5) -> list:
    if not PINECONE_SUPPORT or not pinecone_index: return []
    try:
        pc       = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        q_vector = pc.inference.embed(model="multilingual-e5-large", inputs=[question[:500]],
                                      parameters={"input_type": "query"})
        grade_num   = re.search(r'\d+', str(grade))
        filter_dict = {"grade": {"$eq": grade_num.group()}} if grade_num else None
        results     = pinecone_index.query(vector=q_vector[0].values, top_k=top_k,
                                           include_metadata=True, filter=filter_dict)
        return [m["metadata"] for m in results["matches"] if m["score"] > 0.5]
    except Exception as e:
        print(f"Pinecone search error: {e}")
        return []

def get_all_pdf_chunks(question: str, grade: str = "") -> list:
    if PINECONE_SUPPORT and pinecone_index:
        results = pinecone_search(question, grade)
        if results:
            return [{"filename": r["filename"], "page": r["page"],
                     "text": r["text"], "grade": r.get("grade", ""), "score": 1.0} for r in results]
    if not PDF_SUPPORT: return []
    chunks   = []
    keywords = [kw for kw in question.lower().split() if len(kw) > 3]
    for filename in os.listdir(PDF_FOLDER):
        if not filename.endswith(".pdf"): continue
        try:
            doc = fitz.open(os.path.join(PDF_FOLDER, filename))
            for page_num, page in enumerate(doc):
                text  = page.get_text()
                score = sum(1 for kw in keywords if kw in text.lower())
                if score > 0:
                    chunks.append({"filename": filename, "page": page_num+1,
                                   "text": text[:1500], "score": score})
        except:
            pass
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:5]

def agent_decide_rag(question: str, subject: str, chunks: list) -> dict:
    if not chunks:
        return {"use_rag": False, "reason": "No chunks", "selected_chunks": []}
    chunks_summary = "".join([f"\nChunk {i+1} [{c['filename']} - Page {c['page']}]:\n{c['text'][:300]}...\n"
                               for i, c in enumerate(chunks)])
    agent_prompt = f"""You are a RAG decision agent.
Question: "{question}"
Subject: {subject}
Available PDF chunks: {chunks_summary}
Decide:
1. Are these chunks RELEVANT? (yes/no)
2. Which chunk numbers? (e.g. "1,3" or "none")
Reply format:
USE_RAG: yes/no
CHUNKS: 1,2 or none
REASON: one short sentence"""
    try:
        response = gemini_check.generate_content(agent_prompt)
        track_api_call("gemini-2.5-flash-lite", "rag_agent")
        text    = response.text.strip()
        use_rag = "USE_RAG: yes" in text
        selected = []
        chunks_line = re.search(r"CHUNKS:\s*(.+)", text)
        if chunks_line and use_rag:
            raw = chunks_line.group(1).strip()
            if raw.lower() != "none":
                for num in raw.split(","):
                    try:
                        idx = int(num.strip()) - 1
                        if 0 <= idx < len(chunks):
                            selected.append(chunks[idx])
                    except:
                        pass
        reason_line = re.search(r"REASON:\s*(.+)", text)
        reason = reason_line.group(1).strip() if reason_line else ""
        return {"use_rag": use_rag, "reason": reason, "selected_chunks": selected}
    except:
        return {"use_rag": False, "reason": "Agent error", "selected_chunks": []}

def get_rag_context(question: str, grade: str) -> tuple:
    chunks   = get_all_pdf_chunks(question, grade)
    decision = agent_decide_rag(question, grade, chunks)
    if not decision["use_rag"] or not decision["selected_chunks"]:
        return "", False
    context_parts = [f"[{c['filename']} - Page {c['page']}]\n{c['text']}"
                     for c in decision["selected_chunks"]]
    return "\n\n".join(context_parts), True

# ══════════════════════════════════════════
# ── Progress & XP ──
# ══════════════════════════════════════════
def get_progress(user_name: str) -> dict:
    progress = load_json(PROGRESS_FILE, {})
    if user_name not in progress:
        progress[user_name] = {
            "total_questions": 0, "correct_answers": 0,
            "subjects": {}, "daily_counts": {},
            "streak": 0, "last_active_date": None,
            "xp": 0, "badges": [], "quiz_scores": []
        }
        save_json(PROGRESS_FILE, progress)
    return progress[user_name]

def update_progress(user_name: str, subject: str, correct: bool, xp_earned: int = 5):
    progress = load_json(PROGRESS_FILE, {})
    if user_name not in progress:
        get_progress(user_name)
        progress = load_json(PROGRESS_FILE, {})
    p     = progress[user_name]
    today = datetime.date.today().isoformat()
    p["total_questions"] = p.get("total_questions", 0) + 1
    if correct:
        p["correct_answers"] = p.get("correct_answers", 0) + 1
        p["xp"]              = p.get("xp", 0) + xp_earned
    else:
        p["xp"] = p.get("xp", 0) + 2
    if subject not in p.get("subjects", {}):
        p["subjects"][subject] = {"asked": 0, "correct": 0}
    p["subjects"][subject]["asked"] += 1
    if correct:
        p["subjects"][subject]["correct"] += 1
    p.setdefault("daily_counts", {})[today] = p["daily_counts"].get(today, 0) + 1
    last = p.get("last_active_date")
    if last:
        diff = (datetime.date.today() - datetime.date.fromisoformat(last)).days
        if diff == 1:   p["streak"] = p.get("streak", 0) + 1
        elif diff > 1:  p["streak"] = 1
    else:
        p["streak"] = 1
    p["last_active_date"] = today
    badges    = p.get("badges", [])
    new_badge = None
    xp        = p.get("xp", 0)
    total_q   = p["total_questions"]
    streak    = p.get("streak", 0)
    badge_rules = [
        (total_q >= 1,    "Beginner"),
        (total_q >= 10,   "Scholar"),
        (total_q >= 50,   "Dedicated"),
        (total_q >= 100,  "Champion"),
        (total_q >= 250,  "Legend"),
        (streak >= 3,     "3-Day Streak"),
        (streak >= 7,     "Week Streak"),
        (streak >= 30,    "Month Streak"),
        (xp >= 100,       "XP 100"),
        (xp >= 500,       "XP 500"),
        (xp >= 1000,      "XP 1000"),
    ]
    for condition, badge in badge_rules:
        if condition and badge not in badges:
            badges.append(badge)
            new_badge = badge
            break
    p["badges"]          = badges
    progress[user_name]  = p
    save_json(PROGRESS_FILE, progress)
    return new_badge

def update_leaderboard(user_name: str, grade: str):
    lb       = load_json(LEADERBOARD_FILE, {})
    progress = load_json(PROGRESS_FILE, {})
    if user_name in progress:
        p             = progress[user_name]
        lb[user_name] = {
            "xp": p.get("xp", 0), "streak": p.get("streak", 0),
            "total_q": p.get("total_questions", 0),
            "badges": len(p.get("badges", [])), "grade": grade,
            "updated": datetime.datetime.now().isoformat()
        }
        save_json(LEADERBOARD_FILE, lb)

def update_stats(subject: str):
    stats = load_json(STATS_FILE, {"total": 0, "subjects": {}})
    stats["total"] += 1
    stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
    save_json(STATS_FILE, stats)

def save_history(user_name, grade, subject, question, answer):
    history = load_json(HISTORY_FILE, [])
    history.append({
        "user": user_name, "grade": grade, "subject": subject,
        "question": question, "answer": answer[:500],
        "time": datetime.datetime.now().isoformat()
    })
    history = history[-500:]
    save_json(HISTORY_FILE, history)

# ══════════════════════════════════════════
# ── Weekly Email System ──
# ══════════════════════════════════════════
def get_child_week_stats(child_name: str) -> dict:
    progress   = load_json(PROGRESS_FILE, {})
    tokens_db  = load_json(USER_TOKENS_FILE, {})
    name_lower = child_name.lower()
    week       = datetime.date.today().strftime("%Y-W%W")
    p          = progress.get(name_lower, {})
    t          = tokens_db.get(name_lower, {})
    subjects   = p.get("subjects", {})
    subject_stats = []
    for subj, data in subjects.items():
        asked   = data.get("asked", 0)
        correct = data.get("correct", 0)
        pct     = round(correct / asked * 100) if asked > 0 else 0
        subject_stats.append({"subject": subj, "asked": asked, "correct": correct, "percent": pct})
    subject_stats.sort(key=lambda x: x["percent"])
    week_tokens = t.get("weekly", {}).get(week, {})
    return {
        "name":             child_name,
        "total_questions":  p.get("total_questions", 0),
        "correct":          p.get("correct_answers", 0),
        "streak":           p.get("streak", 0),
        "xp":               p.get("xp", 0),
        "badges":           p.get("badges", []),
        "subjects":         subject_stats,
        "tokens_this_week": week_tokens.get("input", 0) + week_tokens.get("output", 0),
    }

def generate_ai_recommendation(stats: dict) -> str:
    weak       = [s for s in stats["subjects"] if s["percent"] < 60]
    strong     = [s for s in stats["subjects"] if s["percent"] >= 80]
    weak_str   = ", ".join([s["subject"] for s in weak])   or "None"
    strong_str = ", ".join([s["subject"] for s in strong]) or "None"
    prompt = f"""You are a warm, caring education advisor writing to a Sri Lankan parent.
Child: {stats['name']}
This week:
- Questions answered: {stats['total_questions']}
- Correct answers: {stats['correct']}
- Active streak: {stats['streak']} days
- XP earned: {stats['xp']}
- Strong subjects: {strong_str}
- Weak subjects: {weak_str}
- New badges: {', '.join(stats['badges'][-3:]) if stats['badges'] else 'None'}

Write a warm 3-paragraph progress report in Sinhala mixed with English for the parent.
Paragraph 1: Praise what went well this week.
Paragraph 2: Gently mention areas needing improvement.
Paragraph 3: 2-3 specific action steps the parent can take.
Keep it encouraging and under 200 words."""
    try:
        response = gemini_check.generate_content(prompt)
        track_api_call("gemini-2.5-flash-lite", "weekly_report")
        return response.text
    except:
        return f"{stats['name']} this week හොඳ progress show කළා! Regular practice continue කරන්න."

def build_email_html(child_name: str, stats: dict, recommendation: str) -> str:
    subjects_html = ""
    for s in stats["subjects"]:
        pct   = s["percent"]
        color = "#2E7D32" if pct >= 80 else "#E65100" if pct >= 50 else "#B71C1C"
        bar   = "█" * (pct // 10) + "░" * (10 - pct // 10)
        subjects_html += f"""
        <tr>
          <td style="padding:6px 0;font-size:13px;">{s['subject']}</td>
          <td style="font-family:monospace;color:{color};font-size:13px;">{bar}</td>
          <td style="color:{color};font-weight:bold;font-size:13px;">{pct}%</td>
        </tr>"""
    badges_html = " ".join(stats["badges"][-5:]) if stats["badges"] else "No badges yet"
    correct_pct = round(stats["correct"] / stats["total_questions"] * 100) if stats["total_questions"] > 0 else 0
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:Arial,sans-serif;">
<div style="max-width:600px;margin:20px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,0.1);">
  <div style="background:linear-gradient(135deg,#1A1A2E,#5B3CDD);padding:30px;text-align:center;">
    <h1 style="color:white;margin:0;font-size:24px;">Sanz AI Tutor</h1>
    <p style="color:#C9B8E8;margin:8px 0 0;">Weekly Progress Report</p>
  </div>
  <div style="padding:24px;border-bottom:1px solid #eee;">
    <h2 style="color:#1A1A2E;margin:0;">{child_name}'s Week</h2>
    <p style="color:#888;margin:4px 0 0;">{datetime.date.today().strftime('%B %d, %Y')}</p>
  </div>
  <div style="padding:20px;background:#F9F8FF;display:flex;gap:12px;">
    <table width="100%"><tr>
      <td style="text-align:center;background:white;padding:16px;border-radius:8px;">
        <div style="font-size:28px;font-weight:bold;color:#5B3CDD;">{stats['total_questions']}</div>
        <div style="font-size:11px;color:#888;">Questions</div>
      </td>
      <td style="text-align:center;background:white;padding:16px;border-radius:8px;">
        <div style="font-size:28px;font-weight:bold;color:#2E7D32;">{correct_pct}%</div>
        <div style="font-size:11px;color:#888;">Accuracy</div>
      </td>
      <td style="text-align:center;background:white;padding:16px;border-radius:8px;">
        <div style="font-size:28px;font-weight:bold;color:#E65100;">{stats['streak']}</div>
        <div style="font-size:11px;color:#888;">Day Streak</div>
      </td>
      <td style="text-align:center;background:white;padding:16px;border-radius:8px;">
        <div style="font-size:28px;font-weight:bold;color:#0D47A1;">{stats['xp']}</div>
        <div style="font-size:11px;color:#888;">XP</div>
      </td>
    </tr></table>
  </div>
  <div style="padding:24px;">
    <h3 style="color:#1A1A2E;margin:0 0 12px;">Subject Performance</h3>
    <table style="width:100%;border-collapse:collapse;">{subjects_html}</table>
  </div>
  <div style="padding:0 24px 20px;">
    <h3 style="color:#1A1A2E;margin:0 0 8px;">Badges Earned</h3>
    <p style="font-size:20px;margin:0;">{badges_html}</p>
  </div>
  <div style="margin:0 24px 24px;background:#F0EBF8;border-left:4px solid #5B3CDD;border-radius:8px;padding:20px;">
    <h3 style="color:#5B3CDD;margin:0 0 12px;">AI Advisor Recommendation</h3>
    <p style="color:#333;line-height:1.7;margin:0;font-size:14px;">{recommendation.replace(chr(10), '<br>')}</p>
  </div>
  <div style="background:#1A1A2E;padding:20px;text-align:center;">
    <p style="color:#888;font-size:12px;margin:0;">Sanz AI Tutor | sala23451.github.io/sanz-ai-tutor</p>
    <p style="color:#666;font-size:11px;margin:4px 0 0;">Developed by Sanduni Hansika</p>
  </div>
</div></body></html>"""

def send_weekly_email(parent_email: str, child_name: str) -> bool:
    try:
        stats          = get_child_week_stats(child_name)
        recommendation = generate_ai_recommendation(stats)
        html_content   = build_email_html(child_name, stats, recommendation)
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"{child_name}'s Weekly Progress Report — Sanz AI"
        msg["From"]    = EMAIL_USER
        msg["To"]      = parent_email
        msg.attach(MIMEText(html_content, "html"))
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, parent_email, msg.as_string())
        log = load_json(EMAIL_LOG_FILE, [])
        log.append({"parent": parent_email, "child": child_name,
                    "sent_at": datetime.datetime.now().isoformat(), "status": "success"})
        save_json(EMAIL_LOG_FILE, log)
        return True
    except Exception as e:
        log = load_json(EMAIL_LOG_FILE, [])
        log.append({"parent": parent_email, "child": child_name,
                    "sent_at": datetime.datetime.now().isoformat(), "status": "failed", "error": str(e)})
        save_json(EMAIL_LOG_FILE, log)
        return False

def send_all_weekly_reports():
    parents     = load_json(PARENTS_FILE, {})
    children_db = load_json(CHILDREN_FILE, {})
    for email, parent in parents.items():
        if not parent.get("email_reports", True): continue
        for cid in parent.get("children", []):
            if cid in children_db:
                send_weekly_email(email, children_db[cid]["name"])
                print(f"Weekly email sent: {email} -> {children_db[cid]['name']}")

# ── Scheduler ──
if SCHEDULER_SUPPORT:
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_all_weekly_reports, "cron", day_of_week="sun", hour=20, minute=0)
    scheduler.start()

# ══════════════════════════════════════════
# ── AUTH: Legacy User System ──
# ══════════════════════════════════════════
def get_users():
    default = {
        "sanduni": {
            "password_hash": hashlib.sha256("momsanzdad2001#".encode()).hexdigest(),
            "role": "admin"
        }
    }
    users = load_json(USERS_FILE, None)
    if users is None:
        save_json(USERS_FILE, default)
        return default
    return users

@app.post("/auth/login")
async def login(body: LoginRequest):
    users = get_users()
    name  = body.name.strip().lower()
    if name not in users:
        raise HTTPException(status_code=401, detail="User not found")
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if users[name]["password_hash"] != pw_hash:
        raise HTTPException(status_code=401, detail="Wrong password")
    token = secrets.token_hex(32)
    return {"status": "success", "token": token, "role": users[name]["role"], "name": name}

@app.post("/auth/register")
async def register(body: RegisterRequest, x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Only admin can register users")
    users = get_users()
    name  = body.name.strip().lower()
    if name in users:
        raise HTTPException(status_code=400, detail="User already exists")
    users[name] = {
        "password_hash": hashlib.sha256(body.password.encode()).hexdigest(),
        "role": body.role
    }
    save_json(USERS_FILE, users)
    return {"status": "success", "message": f"{name} registered!"}

# ══════════════════════════════════════════
# ── PARENT SYSTEM ──
# ══════════════════════════════════════════
@app.post("/parent/register")
async def parent_register(body: ParentRegisterRequest):
    parents = load_json(PARENTS_FILE, {})
    email   = body.email.strip().lower()
    if email in parents:
        raise HTTPException(status_code=400, detail="Email already registered!")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be 6+ characters!")
    parents[email] = {
        "name":          body.name.strip(),
        "email":         email,
        "password_hash": hashlib.sha256(body.password.encode()).hexdigest(),
        "phone":         body.phone,
        "children":      [],
        "created":       datetime.datetime.now().isoformat(),
        "plan":          "free",
        "email_reports": True
    }
    save_json(PARENTS_FILE, parents)
    return {"status": "success", "message": f"Welcome {body.name}! Account created.", "email": email}

@app.post("/parent/login")
async def parent_login(body: ParentLoginRequest):
    parents = load_json(PARENTS_FILE, {})
    email   = body.email.strip().lower()
    if email not in parents:
        raise HTTPException(status_code=401, detail="Email not found!")
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if parents[email]["password_hash"] != pw_hash:
        raise HTTPException(status_code=401, detail="Wrong password!")
    children_db = load_json(CHILDREN_FILE, {})
    children    = []
    for cid in parents[email]["children"]:
        if cid in children_db:
            child        = children_db[cid].copy()
            tokens       = load_json(USER_TOKENS_FILE, {})
            today        = datetime.date.today().isoformat()
            child_tokens = tokens.get(child["name"].lower(), {})
            today_data   = child_tokens.get("today", {}).get(today, {})
            child["tokens_today"] = today_data.get("input", 0) + today_data.get("output", 0)
            child["tokens_total"] = child_tokens.get("total_input", 0) + child_tokens.get("total_output", 0)
            children.append({"id": cid, **child})
    token = secrets.token_hex(32)
    return {
        "status":   "success",
        "token":    token,
        "name":     parents[email]["name"],
        "email":    email,
        "children": children,
        "plan":     parents[email]["plan"]
    }

@app.post("/parent/add-child")
async def add_child(body: AddChildRequest):
    parents     = load_json(PARENTS_FILE, {})
    children_db = load_json(CHILDREN_FILE, {})
    email       = body.parent_email.strip().lower()
    if email not in parents:
        raise HTTPException(status_code=404, detail="Parent not found!")
    if len(parents[email]["children"]) >= 5:
        raise HTTPException(status_code=400, detail="Maximum 5 children per account!")
    child_id = str(uuid.uuid4())[:8]
    age      = calculate_age(body.birthday)
    children_db[child_id] = {
        "parent_email": email,
        "name":         body.name.strip(),
        "child_email":  body.child_email.strip().lower(),
        "birthday":     body.birthday,
        "age":          age,
        "grade":        body.grade,
        "school":       body.school.strip(),
        "location":     body.location.strip(),
        "created":      datetime.datetime.now().isoformat(),
        "active":       True,
        "avatar":       "student"
    }
    parents[email]["children"].append(child_id)
    save_json(CHILDREN_FILE, children_db)
    save_json(PARENTS_FILE, parents)
    return {"status": "success", "message": f"{body.name}'s account created!", "child_id": child_id, "age": age}

@app.get("/parent/children")
async def get_parent_children(parent_email: str):
    parents     = load_json(PARENTS_FILE, {})
    children_db = load_json(CHILDREN_FILE, {})
    email       = parent_email.strip().lower()
    if email not in parents:
        raise HTTPException(status_code=404, detail="Parent not found!")
    children = []
    for cid in parents[email]["children"]:
        if cid in children_db:
            child        = children_db[cid].copy()
            tokens       = load_json(USER_TOKENS_FILE, {})
            today        = datetime.date.today().isoformat()
            child_tokens = tokens.get(child["name"].lower(), {})
            today_data   = child_tokens.get("today", {}).get(today, {})
            child["tokens_today"] = today_data.get("input", 0) + today_data.get("output", 0)
            child["tokens_total"] = child_tokens.get("total_input", 0) + child_tokens.get("total_output", 0)
            child["progress"]     = get_progress(child["name"].lower())
            children.append({"id": cid, **child})
    return {"status": "success", "children": children}

@app.put("/parent/child/update")
async def update_child(body: UpdateChildRequest):
    children_db = load_json(CHILDREN_FILE, {})
    email       = body.parent_email.strip().lower()
    if body.child_id not in children_db:
        raise HTTPException(status_code=404, detail="Child not found!")
    if children_db[body.child_id]["parent_email"] != email:
        raise HTTPException(status_code=403, detail="Not your child's account!")
    child = children_db[body.child_id]
    if body.name:     child["name"]     = body.name
    if body.grade:    child["grade"]    = body.grade
    if body.school:   child["school"]   = body.school
    if body.location: child["location"] = body.location
    children_db[body.child_id] = child
    save_json(CHILDREN_FILE, children_db)
    return {"status": "success", "message": "Profile updated!"}

@app.delete("/parent/child")
async def delete_child(body: DeleteChildRequest):
    children_db = load_json(CHILDREN_FILE, {})
    parents     = load_json(PARENTS_FILE, {})
    email       = body.parent_email.strip().lower()
    if body.child_id not in children_db:
        raise HTTPException(status_code=404, detail="Child not found!")
    if children_db[body.child_id]["parent_email"] != email:
        raise HTTPException(status_code=403, detail="Not your child's account!")
    del children_db[body.child_id]
    if body.child_id in parents[email]["children"]:
        parents[email]["children"].remove(body.child_id)
    save_json(CHILDREN_FILE, children_db)
    save_json(PARENTS_FILE, parents)
    return {"status": "success", "message": "Child account removed!"}

# ══════════════════════════════════════════
# ── PROGRESS & LEADERBOARD ──
# ══════════════════════════════════════════
@app.get("/progress/{user_name}")
async def get_student_progress(user_name: str):
    return {"status": "success", "progress": get_progress(user_name.lower())}

@app.get("/leaderboard")
async def get_leaderboard(grade: str = ""):
    lb      = load_json(LEADERBOARD_FILE, {})
    entries = list(lb.items())
    if grade:
        entries = [(k, v) for k, v in entries if str(v.get("grade", "")) == grade]
    entries.sort(key=lambda x: x[1].get("xp", 0), reverse=True)
    return {"status": "success", "leaderboard": [
        {"rank": i+1, "name": name, **data} for i, (name, data) in enumerate(entries[:20])
    ]}

# ══════════════════════════════════════════
# ── QUIZ SYSTEM ──
# ══════════════════════════════════════════
@app.post("/quiz/start")
async def start_quiz(body: QuizStartRequest):
    lang       = body.language if body.language in LANG_INSTRUCTIONS else "en"
    difficulty = "easy" if int(body.grade) <= 5 else "medium" if int(body.grade) <= 9 else "hard"
    prompt = f"""Generate exactly 5 multiple-choice quiz questions for:
Grade: {body.grade}, Subject: {body.subject}, Difficulty: {difficulty}, Language: {body.language}
Return ONLY valid JSON:
{{"questions":[{{"q":"Question","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"answer":"A","explanation":"Why A is correct"}}]}}"""
    try:
        response  = gemini_flash.generate_content(prompt)
        track_api_call("gemini-2.5-flash", "quiz_generate")
        raw       = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
        data      = json.loads(raw)
        questions = data.get("questions", [])
        if not questions: raise ValueError("No questions")
    except Exception as e:
        return {"status": "error", "message": f"Quiz generation failed: {e}"}
    session_id = secrets.token_hex(8)
    sessions   = load_json(QUIZ_FILE, {})
    sessions[session_id] = {
        "user_name": body.user_name, "grade": body.grade,
        "subject": body.subject, "language": lang,
        "questions": questions, "current": 0,
        "score": 0, "answers": [],
        "started": datetime.datetime.now().isoformat(), "completed": False
    }
    save_json(QUIZ_FILE, sessions)
    q = questions[0]
    return {
        "status": "success", "session_id": session_id,
        "total": len(questions), "current": 1,
        "question": q["q"], "options": q["options"], "progress_pct": 0
    }

@app.post("/quiz/answer")
async def answer_quiz(body: QuizAnswerRequest):
    sessions = load_json(QUIZ_FILE, {})
    if body.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Quiz not found")
    sess = sessions[body.session_id]
    if sess["completed"]:
        raise HTTPException(status_code=400, detail="Quiz already done")
    idx     = sess["current"]
    q       = sess["questions"][idx]
    correct = body.user_answer.strip().upper().startswith(q["answer"].upper())
    if correct: sess["score"] += 1
    sess["answers"].append({
        "question": q["q"], "user_answer": body.user_answer,
        "correct": correct, "right_answer": q["answer"],
        "explanation": q.get("explanation", "")
    })
    sess["current"] += 1
    if sess["current"] >= len(sess["questions"]):
        sess["completed"] = True
        sessions[body.session_id] = sess
        save_json(QUIZ_FILE, sessions)
        xp_earned = sess["score"] * 15
        user      = body.user_name.lower()
        progress  = load_json(PROGRESS_FILE, {})
        if user not in progress:
            get_progress(user)
            progress = load_json(PROGRESS_FILE, {})
        p         = progress.get(user, {})
        p["xp"]   = p.get("xp", 0) + xp_earned
        p.setdefault("quiz_scores", []).append({
            "subject": sess["subject"], "score": sess["score"],
            "total": len(sess["questions"]), "xp": xp_earned,
            "date": datetime.date.today().isoformat()
        })
        progress[user] = p
        save_json(PROGRESS_FILE, progress)
        update_leaderboard(user, sess["grade"])
        return {
            "status": "completed", "correct": correct,
            "explanation": q.get("explanation", ""),
            "score": sess["score"], "total": len(sess["questions"]),
            "percent": round(sess["score"] / len(sess["questions"]) * 100),
            "xp_earned": xp_earned, "answers": sess["answers"]
        }
    sessions[body.session_id] = sess
    save_json(QUIZ_FILE, sessions)
    next_q = sess["questions"][sess["current"]]
    return {
        "status": "ongoing", "correct": correct,
        "explanation": q.get("explanation", ""),
        "current": sess["current"] + 1, "total": len(sess["questions"]),
        "question": next_q["q"], "options": next_q["options"],
        "score_so_far": sess["score"],
        "progress_pct": round(sess["current"] / len(sess["questions"]) * 100)
    }

# ══════════════════════════════════════════
# ── VOICE & TTS ──
# ══════════════════════════════════════════
@app.post("/voice-input")
async def voice_input(audio: UploadFile = File(...)):
    if not STT_SUPPORT:
        raise HTTPException(status_code=500, detail="STT not installed. pip install SpeechRecognition")
    temp_path = os.path.join("temp_audio", "input.wav")
    contents  = await audio.read()
    with open(temp_path, "wb") as f:
        f.write(contents)
    try:
        recognizer    = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
        detected_text, detected_lang = None, "en"
        for lang_code, lang_key in [("si-LK", "si"), ("ta-LK", "ta"), ("en-US", "en")]:
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
                if text:
                    detected_text = text
                    detected_lang = lang_key
                    break
            except:
                pass
        try: os.remove(temp_path)
        except: pass
        if not detected_text:
            raise HTTPException(status_code=400, detail="Could not understand audio.")
        text_lang = detect_language(detected_text)
        if text_lang != "en": detected_lang = text_lang
        return {"status": "success", "text": detected_text, "detected_language": detected_lang}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(body: TTSRequest):
    if not TTS_SUPPORT:
        raise HTTPException(status_code=500, detail="TTS not installed. pip install gTTS")
    language   = body.language if body.language in LANG_INSTRUCTIONS else detect_language(body.text)
    gtts_lang  = LANG_INSTRUCTIONS[language]['gtts_lang']
    clean_text = re.sub(r'[*_#`\[\]()~>]', '', body.text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:1000]
    try:
        tts          = gTTS(text=clean_text, lang=gtts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return StreamingResponse(audio_buffer, media_type="audio/mpeg",
                                 headers={"Content-Disposition": "inline; filename=response.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════
# ── IMAGE GENERATION ──
# ══════════════════════════════════════════
@app.post("/generate-image")
async def generate_image(body: ImageGenRequest):
    try:
        img_prompt = f"""Create a clear educational diagram for: "{body.prompt}"
Subject: {body.subject}
Provide matplotlib code between [GRAPH_START] and [GRAPH_END] tags.
Do NOT use plt.savefig()."""
        response = gemini_flash.generate_content(img_prompt)
        track_api_call("gemini-2.5-flash", "image_generate")
        answer      = response.text
        graph_url   = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', answer, re.DOTALL)
        description = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', answer, flags=re.DOTALL).strip()
        if graph_match:
            graph_code = graph_match.group(1).strip()
            try:
                plt.figure(figsize=(8, 6))
                safe_globals = {
                    "__builtins__": {}, "plt": plt, "np": np,
                    "range": range, "len": len, "zip": zip,
                    "list": list, "tuple": tuple, "int": int, "float": float,
                    "str": str, "abs": abs, "min": min, "max": max,
                    "sum": sum, "round": round, "enumerate": enumerate,
                    "math": __import__('math'),
                }
                exec(graph_code, safe_globals)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                plt.close()
            except Exception as e:
                print(f"Image gen error: {e}")
                plt.close()
        return {"status": "success", "description": description[:500], "image_url": graph_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ══════════════════════════════════════════
# ── MAIN SOLVE ──
# ══════════════════════════════════════════
@app.post("/solve")
async def solve_math(
    name:                 str        = Form(default="User"),
    grade:                str        = Form(default="N/A"),
    subject:              str        = Form(default="Mathematics"),
    question:             str        = Form(default=""),
    language:             str        = Form(default="si"),
    image:                UploadFile = File(default=None),
    conversation_history: str        = Form(default="[]"),
):
    try:
        if language not in LANG_INSTRUCTIONS:
            language = detect_language(question)
        lang_cfg = LANG_INSTRUCTIONS[language]

        # 1. Blacklist check
        banned, remaining = is_blacklisted(name)
        if banned:
            msgs = {
                "si": f"⛔ ඔයා {remaining} minutes ගෙවෙන තෙක් use කරන්න බෑ!",
                "en": f"⛔ You are blocked for {remaining} more minutes!",
                "ta": f"⛔ இன்னும் {remaining} நிமிடங்கள் பயன்படுத்த முடியாது!"
            }
            return {"status": "banned", "message": msgs.get(language, msgs["si"])}

        # 2. Creator check
        creator_names = ["sanduni","hansika","sanduni hansika","sanz","sanz queen"]
        if any(n in question.lower() for n in creator_names):
            return {
                "status": "creator",
                "message": "අනේ... ඔයා Sanz Queen ගැන දන්නවාද? 👑\n\nඒ කෙනා තමයි මේ platform එකම හැදුවේ. Please — misuse කරන්න එපා. 🙏"
            }

        # 3. Bad words
        if contains_bad_words(question):
            add_to_blacklist(name)
            msgs = {
                "si": "⛔ නුසුදුසු වචන use කළා! ඔයා පැය 2ක් blacklist කරලා!",
                "en": "⛔ Inappropriate words! You are blacklisted for 2 hours!",
                "ta": "⛔ தகாத வார்த்தைகள்! 2 மணி நேரம் தடைசெய்யப்பட்டீர்கள்!"
            }
            return {"status": "banned", "message": msgs.get(language, msgs["si"])}

        # 4. Rage warning
        rage_warning = ""
        if is_rage(question):
            rage_msgs = {
                "si": "⚠️ කරුණාකර සංසුන්ව ප්‍රශ්නය අහන්න! ",
                "en": "⚠️ Please ask calmly! ",
                "ta": "⚠️ தயவுசெய்து அமைதியாக கேளுங்கள்! "
            }
            rage_warning = rage_msgs.get(language, "")

        # 5. Conversation history
        conv_history = []
        try:
            conv_history = json.loads(conversation_history)[-10:]
        except:
            conv_history = []

        # 6. Cache check
        if not conv_history:
            cached_entry, cache_score = cache_lookup(question, subject, language)
            if cached_entry:
                save_history(name, grade, subject, question, cached_entry["answer"])
                update_stats(subject)
                track_user_tokens(name.lower(), estimate_tokens(question), estimate_tokens(cached_entry["answer"]))
                return {
                    "status": "success", "answer": rage_warning + cached_entry["answer"],
                    "graph_url": cached_entry.get("graph_url"), "verified": cached_entry.get("verified", True),
                    "rag_used": False, "cache_hit": True,
                    "cache_score": round(cache_score, 3), "detected_language": language
                }

        # 7. RAG
        rag_context, rag_used = get_rag_context(question, grade)

        # 8. Conversation context
        conversation_context = ""
        if conv_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conv_history:
                role = msg.get("role", "")
                text = msg.get("text", "")[:300]
                if role == "user":  conversation_context += f"Student: {text}\n"
                elif role == "ai":  conversation_context += f"Tutor: {text}\n"
            conversation_context += "\nContinue naturally based on above context.\n"

        # 9. Build instruction
        is_math = any(w in subject.lower() for w in ["math", "maths", "ගණිත", "கணித"])
        if is_math:
            instruction = f"""
{lang_cfg['socratic']}
Student: {name}, Grade {grade}, Subject: {subject}.
{conversation_context}
If a graph is needed, provide matplotlib code between [GRAPH_START] and [GRAPH_END]. Do NOT use plt.savefig().
{f'Textbook reference: {rag_context}' if rag_context else ''}"""
        else:
            instruction = f"""
You are an expert {subject} teacher. Student: {name}, Grade {grade}.
{lang_cfg['instruction']}
{conversation_context}
Solve step by step. If graph needed, provide matplotlib code between [GRAPH_START] and [GRAPH_END]. Do NOT use plt.savefig().
{f'Textbook reference: {rag_context}' if rag_context else ''}"""

        content_list = [f"User: {name}, Grade: {grade}, Subject: {subject}. Question: {question}"]
        if image and image.filename:
            image_data = await image.read()
            content_list.append({"mime_type": "image/jpeg", "data": image_data})

        # 10. Main answer
        response1 = gemini_flash.generate_content([instruction] + content_list)
        answer1   = response1.text
        track_api_call("gemini-2.5-flash", "main_answer")

        # 11. Cross-check (math only)
        if is_math:
            cross_check_prompt = f"""Is this answer correct?
Question: {question}
Answer: {answer1[:500]}
{lang_cfg['cross_check']}"""
            cross_response = gemini_check.generate_content(cross_check_prompt)
            cross_check    = cross_response.text
            track_api_call("gemini-2.5-flash-lite", "cross_check")
            if lang_cfg['correct_word'] in cross_check:
                final_answer = answer1
                verified     = True
            else:
                redirect_prompt = f"""
{lang_cfg['socratic']}
{conversation_context}
The student asked: "{question}"
Their answer has issues. WITHOUT saying wrong/incorrect,
ask ONE gentle guiding question. Be warm. 2-3 sentences max."""
                redirect_resp = gemini_check.generate_content(redirect_prompt)
                final_answer  = redirect_resp.text
                track_api_call("gemini-2.5-flash-lite", "socratic_redirect")
                verified = False
        else:
            final_answer = answer1
            verified     = True

        # 12. Graph generation
        graph_url   = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', final_answer, re.DOTALL)
        if graph_match:
            graph_code   = graph_match.group(1).strip()
            final_answer = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', final_answer, flags=re.DOTALL)
            try:
                plt.figure(figsize=(6, 4))
                safe_globals = {
                    "__builtins__": {}, "plt": plt, "np": np,
                    "range": range, "len": len, "zip": zip,
                    "list": list, "tuple": tuple, "int": int, "float": float,
                    "str": str, "abs": abs, "min": min, "max": max,
                    "sum": sum, "round": round, "enumerate": enumerate,
                    "math": __import__('math'),
                }
                exec(graph_code, safe_globals)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                plt.close()
            except Exception as e:
                print(f"Graph error: {e}")
                plt.close()

        # 13. Save & track
        save_history(name, grade, subject, question, final_answer)
        update_stats(subject)
        new_badge = update_progress(name.lower(), subject, verified, xp_earned=5 if verified else 2)
        update_leaderboard(name.lower(), grade)

        # Token tracking
        input_est  = estimate_tokens(question + (rag_context or "") + str(conv_history))
        output_est = estimate_tokens(final_answer)
        track_user_tokens(name.lower(), input_est, output_est)

        # Cache store
        if not conv_history and not (image and image.filename) and not is_math:
            cache_store(question, subject, language, final_answer, graph_url, verified)

        return {
            "status":            "success",
            "answer":            rage_warning + final_answer,
            "graph_url":         graph_url,
            "verified":          verified,
            "rag_used":          rag_used,
            "cache_hit":         False,
            "detected_language": language,
            "new_badge":         new_badge,
            "progress":          get_progress(name.lower())
        }

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            retry_msgs = {
                "si": "⏳ AI එක ටිකක් විවේක ගන්නවා! තත්පර 30කින් නැවත try කරන්න. 😊",
                "en": "⏳ AI is taking a short break! Please try again in 30 seconds. 😊",
                "ta": "⏳ AI சிறிது ஓய்வு எடுக்கிறது! 30 வினாடிகளில் மீண்டும் முயற்சிக்கவும். 😊"
            }
            return {"status": "success", "answer": retry_msgs.get(language, retry_msgs["en"]),
                    "graph_url": None, "verified": True, "rag_used": False,
                    "cache_hit": False, "detected_language": language}
        return {"status": "error", "message": error_msg}

# ══════════════════════════════════════════
# ── ADMIN ROUTES ──
# ══════════════════════════════════════════
@app.get("/health")
async def health_check():
    return {
        "status": "ok", "version": "3.0.0",
        "stt_support": STT_SUPPORT, "tts_support": TTS_SUPPORT,
        "pdf_support": PDF_SUPPORT, "cloudinary_support": CLOUDINARY_SUPPORT,
        "pinecone_support": PINECONE_SUPPORT, "scheduler_support": SCHEDULER_SUPPORT
    }

@app.get("/admin/stats")
async def admin_stats(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    stats     = load_json(STATS_FILE, {"total": 0, "subjects": {}})
    history   = load_json(HISTORY_FILE, [])
    blacklist = load_json(BLACKLIST_FILE, {})
    parents   = load_json(PARENTS_FILE, {})
    children  = load_json(CHILDREN_FILE, {})
    return {
        "status": "success",
        "total_questions":  stats["total"],
        "subjects":         stats["subjects"],
        "recent_users":     list(set([h["user"] for h in history[-50:]])),
        "blacklisted_users":list(blacklist.keys()),
        "api_usage":        get_api_usage(),
        "total_parents":    len(parents),
        "total_children":   len(children)
    }

@app.get("/admin/history")
async def admin_history(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "success", "history": load_json(HISTORY_FILE, [])[-100:]}

@app.get("/admin/api-usage")
async def admin_api_usage(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "success", "usage": get_api_usage()}

@app.get("/admin/token-usage")
async def admin_token_usage(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tokens  = load_json(USER_TOKENS_FILE, {})
    today   = datetime.date.today().isoformat()
    month   = datetime.date.today().strftime("%Y-%m")
    summary = []
    for user, data in tokens.items():
        today_data = data.get("today", {}).get(today, {})
        month_data = data.get("monthly", {}).get(month, {})
        t_in       = today_data.get("input", 0)
        t_out      = today_data.get("output", 0)
        summary.append({
            "user":           user,
            "today_tokens":   t_in + t_out,
            "today_calls":    today_data.get("calls", 0),
            "month_tokens":   month_data.get("input", 0) + month_data.get("output", 0),
            "total_tokens":   data.get("total_input", 0) + data.get("total_output", 0),
            "total_calls":    data.get("total_calls", 0),
            "est_cost_today": round((t_in * 0.10 + t_out * 0.40) / 1_000_000, 6),
            "est_cost_lkr":   round((t_in * 0.10 + t_out * 0.40) / 1_000_000 * 300, 4),
        })
    summary.sort(key=lambda x: x["today_tokens"], reverse=True)
    total_today    = sum(u["today_tokens"] for u in summary)
    total_cost_usd = sum(u["est_cost_today"] for u in summary)
    return {
        "status":       "success",
        "date":         today,
        "users":        summary,
        "total_today":  total_today,
        "est_cost_usd": round(total_cost_usd, 6),
        "est_cost_lkr": round(total_cost_usd * 300, 4)
    }

@app.get("/admin/token-usage/{user_name}")
async def admin_user_tokens(user_name: str, x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tokens = load_json(USER_TOKENS_FILE, {})
    if user_name not in tokens:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "user": user_name, "data": tokens[user_name]}

@app.get("/admin/parents")
async def admin_parents(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    parents     = load_json(PARENTS_FILE, {})
    children_db = load_json(CHILDREN_FILE, {})
    result      = []
    for email, p in parents.items():
        children = [
            {"id": cid, "name": children_db[cid]["name"], "grade": children_db[cid]["grade"]}
            for cid in p.get("children", []) if cid in children_db
        ]
        result.append({
            "email": email, "name": p["name"],
            "children": children, "plan": p.get("plan", "free"),
            "created": p.get("created", "")
        })
    return {"status": "success", "parents": result, "total": len(result)}

@app.post("/admin/email/send-now")
async def admin_send_email_now(
    parent_email: str, child_name: str,
    x_admin_password: str = Header(default="")
):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    success = send_weekly_email(parent_email, child_name)
    return {
        "status":  "success" if success else "failed",
        "message": f"Email {'sent' if success else 'failed'} to {parent_email}"
    }

@app.get("/admin/email/log")
async def admin_email_log(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "success", "log": load_json(EMAIL_LOG_FILE, [])[-50:]}

@app.post("/admin/blacklist/remove")
async def admin_remove_blacklist(body: BlacklistRemoveRequest, x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    blacklist = load_json(BLACKLIST_FILE, {})
    if body.user_name in blacklist:
        del blacklist[body.user_name]
        save_json(BLACKLIST_FILE, blacklist)
    return {"status": "success", "message": f"{body.user_name} removed!"}

@app.post("/admin/blacklist/add")
async def admin_add_blacklist(body: BlacklistRemoveRequest, x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    add_to_blacklist(body.user_name)
    return {"status": "success", "message": f"{body.user_name} banned for 2 hours!"}

@app.post("/admin/upload_pdf")
async def admin_upload_pdf(pdf: UploadFile = File(...), x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    contents = await pdf.read()
    filename = pdf.filename
    with open(os.path.join(PDF_FOLDER, filename), "wb") as f:
        f.write(contents)
    cloud_url = ""
    if CLOUDINARY_SUPPORT:
        try:
            result    = cloudinary.uploader.upload(contents, resource_type="raw",
                                                   folder="sanz_textbooks",
                                                   public_id=filename.replace(".pdf", ""),
                                                   overwrite=True)
            cloud_url = result.get("secure_url", "")
        except Exception as e:
            print(f"Cloudinary error: {e}")
    indexed = False
    if PINECONE_SUPPORT and pinecone_index and PDF_SUPPORT:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            doc     = fitz.open(tmp_path)
            vectors = []
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if not text or len(text) < 50: continue
                grade_match = re.search(r'g[-_]?(\d+)', filename.lower())
                file_grade  = grade_match.group(1) if grade_match else "unknown"
                vectors.append({
                    "id": f"{filename}_page_{page_num+1}",
                    "values": pinecone_embed(text),
                    "metadata": {"filename": filename, "page": page_num+1,
                                 "text": text[:1000], "grade": file_grade}
                })
            os.unlink(tmp_path)
            for i in range(0, len(vectors), 50):
                pinecone_index.upsert(vectors=vectors[i:i+50])
            indexed = True
        except Exception as e:
            print(f"Pinecone index error: {e}")
    return {
        "status": "success", "message": f"{filename} uploaded!",
        "cloud_url": cloud_url, "pinecone_indexed": indexed
    }

@app.delete("/admin/pdfs/{filename}")
async def admin_delete_pdf(filename: str, x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    local_path = os.path.join(PDF_FOLDER, filename)
    if os.path.exists(local_path): os.remove(local_path)
    if CLOUDINARY_SUPPORT:
        try: cloudinary.uploader.destroy(f"sanz_textbooks/{filename.replace('.pdf','')}", resource_type="raw")
        except: pass
    return {"status": "success", "message": f"{filename} deleted!"}

@app.get("/admin/pdfs")
async def admin_list_pdfs(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    return {"status": "success", "pdfs": pdfs}

# ══════════════════════════════════════════
# ── RUN ──
# ══════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
