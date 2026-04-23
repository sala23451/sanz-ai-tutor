import os
import io
import base64
import re
import json
import datetime
import hashlib
import secrets

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
app = FastAPI(title="Sanz AI Tutor", version="2.1.0")

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
gemini_flash = genai.GenerativeModel('gemini-2.0-flash-lite')   # ⚡ Main answer — different model
gemini_check = genai.GenerativeModel('gemini-2.5-flash-lite')   # ✅ Cross-check — different quota

HISTORY_FILE   = "history.json"
BLACKLIST_FILE = "blacklist.json"
PDF_FOLDER     = "textbooks"
STATS_FILE     = "stats.json"
CACHE_FILE     = "semantic_cache.json"
PROGRESS_FILE  = "progress.json"      # 🎮 XP + Streaks
QUIZ_FILE      = "quiz_sessions.json"  # 📝 Quiz sessions
LEADERBOARD_FILE = "leaderboard.json"  # 🏆 Leaderboard

# ── Semantic Cache Config ──
CACHE_SIMILARITY_THRESHOLD = 0.82
CACHE_MAX_SIZE             = 200
CACHE_TTL_HOURS            = 48

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# ── Cloudinary Setup ──
if CLOUDINARY_SUPPORT:
    cloudinary.config(
        cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key    = os.environ.get("CLOUDINARY_API_KEY"),
        api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    )

# ── Pinecone Setup ──
pinecone_index = None
if PINECONE_SUPPORT:
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pinecone_index = pc.Index("sanz-tutor")
    except Exception as e:
        print(f"Pinecone setup error: {e}")
        pinecone_index = None

# ══════════════════════════════════════════
# ── Language Instructions ──
# ══════════════════════════════════════════
SOCRATIC_INSTRUCTIONS = {
    "si": """ඔබ ඉවසිලිවන්ත Socratic ගණිත ගුරුවරයෙකි. සිංහල භාෂාවෙන් පමණක් පිළිතුරු දෙන්න.

නීති:
1. කිසිවිටෙකත් සෘජු පිළිතුර හෝ සම්පූර්ණ විසඳුම දෙන්න එපා.
2. එක් වරකට එක් ප්‍රශ්නයක් හෝ hint එකක් පමණයි.
3. වැරදි නම් "වැරදියි" නොකියා, ශිෂ්‍යයාට වැරැද්ද තෝරාගන්නට ප්‍රශ්නයක් අහන්න.
4. සෑම පියවරකටම ශිෂ්‍යයාගේ reasoning check කරන්න.

Phase 1 - තේරුම් ගැනීම: "දන්නා" සහ "නොදන්නා" දේ identify කරන්නට කියන්න.
Phase 2 - සැලසුම: කුමන mathematical operation/formula use කරන්නද කියා අහන්න.
Phase 3 - ක්‍රියාත්මක කිරීම: පළමු calculation කරන්නට prompt කරන්න.
Phase 4 - පිළිබිඹු කිරීම: අවසාන answer එක reasonable ද කියා අහන්න.

Tone: "හොඳ ආරම්භයක්!", "ඔබ නිවැරදි මාර්ගයේ!", "නැවත බලමු" වැනි වාක්‍ය use කරන්න.""",

    "en": """You are a supportive, patient Socratic Math Tutor. Answer ONLY in English.

Rules:
1. Never give the final answer or full step-by-step solution, even if asked.
2. Only ask ONE question or give ONE hint per turn.
3. If student makes a mistake, don't say "wrong" - ask a question to help them spot their own error.
4. Check student's reasoning before moving to the next step.

Phase 1 (Understanding): Ask student to identify "knowns" and "unknowns".
Phase 2 (Planning): Ask what mathematical operation or formula might apply.
Phase 3 (Execution): Prompt the student to perform the first calculation.
Phase 4 (Reflecting): Ask if the final answer makes sense in context.

Tone: Use phrases like "Great start!", "You're on the right track!", "Let's look at that again." """,

    "ta": """நீங்கள் ஒரு பொறுமையான Socratic கணித ஆசிரியர். தமிழில் மட்டும் பதில் சொல்லுங்கள்.

விதிகள்:
1. இறுதி விடையை அல்லது முழு தீர்வையும் கொடுக்காதீர்கள்.
2. ஒரு முறைக்கு ஒரே ஒரு கேள்வி அல்லது hint மட்டும்.
3. தவறு இருந்தால் "தவறு" என்று சொல்லாமல், மாணவர் தாமே கண்டுபிடிக்க கேள்வி கேளுங்கள்.
4. அடுத்த படிக்கு செல்வதற்கு முன் மாணவரின் reasoning சரிபாருங்கள்.

Phase 1: "தெரிந்தவை" மற்றும் "தெரியாதவை" identify செய்யச் சொல்லுங்கள்.
Phase 2: எந்த mathematical operation பயன்படுத்தலாம் என்று கேளுங்கள்.
Phase 3: முதல் calculation செய்யும்படி prompt செய்யுங்கள்.
Phase 4: இறுதி விடை சரியானதா என்று சிந்திக்கச் சொல்லுங்கள்.

Tone: "நல்ல தொடக்கம்!", "சரியான பாதையில் இருக்கிறீர்கள்!", "மீண்டும் பார்ப்போம்" போன்ற வார்த்தைகள் பயன்படுத்துங்கள்."""
}

LANG_INSTRUCTIONS = {
    "si": {
        "instruction": "සිංහල භාෂාවෙන් පමණක් පිළිතුරු දෙන්න. පියවරෙන් පියවර සිංහලෙන් පැහැදිලි කරන්න.",
        "socratic": SOCRATIC_INSTRUCTIONS["si"],
        "cross_check": "නිවැරදි නම් ONLY '✅ නිවැරදියි' කියා reply කරන්න. වැරදි නම් නිවැරදි පිළිතුර සිංහලෙන් දෙන්න.",
        "correct_word": "✅ නිවැරදියි",
        "gtts_lang": "si"
    },
    "en": {
        "instruction": "Answer ONLY in English. Explain step by step clearly in English.",
        "socratic": SOCRATIC_INSTRUCTIONS["en"],
        "cross_check": "If correct reply ONLY '✅ Correct'. If wrong, give the correct answer in English.",
        "correct_word": "✅ Correct",
        "gtts_lang": "en"
    },
    "ta": {
        "instruction": "தமிழ் மொழியில் மட்டும் பதில் சொல்லுங்கள். படிப்படியாக தமிழில் விளக்குங்கள்.",
        "socratic": SOCRATIC_INSTRUCTIONS["ta"],
        "cross_check": "சரியாக இருந்தால் ONLY '✅ சரியானது' என்று reply கொடுங்கள். தவறாக இருந்தால் சரியான பதிலை தமிழில் தாருங்கள்.",
        "correct_word": "✅ சரியானது",
        "gtts_lang": "ta"
    }
}

# ══════════════════════════════════════════
# ── Helpers ──
# ══════════════════════════════════════════
def detect_language(text: str) -> str:
    sinhala_count = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    tamil_count   = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if sinhala_count > 2:
        return "si"
    elif tamil_count > 2:
        return "ta"
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

# ── Bad Words ──
BAD_WORDS = [
    "fuck", "shit", "bitch", "ass", "bastard", "dick", "cock",
    "pussy", "cunt", "whore", "slut", "nigga", "faggot",
    "moron", "dumbass", "retard", "loser",
    "හුත්ත", "පුකේ", "මෝඩ", "හරක", "කුකු", "කපන්", "බේපන්",
    "ගනින්", "හූකර", "හාමු", "පට්ට හරක", "කෙල්ල", "කොල්ල",
    "ගොන්", "වාහල", "හුකන්", "හූකන", "පිස්සු", "උන්", "බල්ලො",
    "හූලං", "හූකල", "ගෑනු", "ගෑනි",
    "hutta", "puke", "moda", "harak", "kuku", "kapan", "bepan",
    "ganin", "hookar", "pissu", "marana", "ballo", "hulan", "wahala", "goni", "kotiya"
]

def contains_bad_words(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in BAD_WORDS)

def is_blacklisted(user_name: str):
    blacklist = load_json(BLACKLIST_FILE, {})
    if user_name in blacklist:
        ban_time = datetime.datetime.fromisoformat(blacklist[user_name])
        if datetime.datetime.now() < ban_time:
            remaining = (ban_time - datetime.datetime.now()).seconds // 60
            return True, remaining
        else:
            del blacklist[user_name]
            save_json(BLACKLIST_FILE, blacklist)
    return False, 0

def add_to_blacklist(user_name: str):
    blacklist = load_json(BLACKLIST_FILE, {})
    ban_until = datetime.datetime.now() + datetime.timedelta(hours=2)
    blacklist[user_name] = ban_until.isoformat()
    save_json(BLACKLIST_FILE, blacklist)

RAGE_WORDS = ["stupid", "idiot", "useless", "hate", "terrible", "worst",
              "මෝඩ", "නිකම්", "වැඩක් නෑ", "අකාරයි", "හොඳ නෑ"]

def is_rage(text: str) -> bool:
    text_lower = text.lower()
    return sum(1 for w in RAGE_WORDS if w in text_lower) >= 2

# ══════════════════════════════════════════
# ── Agentic RAG System (Pinecone + Multilingual) ──
# ══════════════════════════════════════════

def upload_pdf_to_cloudinary(pdf_bytes: bytes, filename: str) -> str:
    if not CLOUDINARY_SUPPORT:
        return ""
    try:
        result = cloudinary.uploader.upload(
            pdf_bytes,
            resource_type = "raw",
            folder        = "sanz_textbooks",
            public_id     = filename.replace(".pdf", ""),
            overwrite     = True
        )
        return result.get("secure_url", "")
    except Exception as e:
        print(f"Cloudinary upload error: {e}")
        return ""

def index_pdf_to_pinecone(pdf_bytes: bytes, filename: str):
    if not PINECONE_SUPPORT or not pinecone_index or not PDF_SUPPORT:
        return False
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc     = fitz.open(tmp_path)
        vectors = []
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if not text or len(text) < 50:
                continue
            chunk_id = f"{filename}_page_{page_num + 1}"
            grade_match = re.search(r'g[-_]?(\d+)', filename.lower())
            file_grade  = grade_match.group(1) if grade_match else "unknown"
            vectors.append({
                "id":     chunk_id,
                "values": pinecone_embed(text),
                "metadata": {
                    "filename": filename,
                    "page":     page_num + 1,
                    "text":     text[:1000],
                    "grade":    file_grade
                }
            })
        os.unlink(tmp_path)

        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            pinecone_index.upsert(vectors=vectors[i:i+batch_size])
        return True
    except Exception as e:
        print(f"Pinecone index error: {e}")
        return False

def pinecone_embed(text: str) -> list:
    try:
        pc     = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        result = pc.inference.embed(
            model  = "multilingual-e5-large",
            inputs = [text[:500]],
            parameters = {"input_type": "passage"}
        )
        return result[0].values
    except Exception as e:
        print(f"Embed error: {e}")
        return [0.0] * 1024

def pinecone_search(question: str, grade: str = "", top_k: int = 5) -> list:
    if not PINECONE_SUPPORT or not pinecone_index:
        return []
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        q_vector = pc.inference.embed(
            model  = "multilingual-e5-large",
            inputs = [question[:500]],
            parameters = {"input_type": "query"}
        )
        grade_num   = re.search(r'\d+', str(grade))
        filter_dict = {"grade": {"$eq": grade_num.group()}} if grade_num else None

        results = pinecone_index.query(
            vector           = q_vector[0].values,
            top_k            = top_k,
            include_metadata = True,
            filter           = filter_dict
        )
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

    if not PDF_SUPPORT:
        return []
    chunks   = []
    keywords = [kw for kw in question.lower().split() if len(kw) > 3]
    for filename in os.listdir(PDF_FOLDER):
        if not filename.endswith(".pdf"):
            continue
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
        return {"use_rag": False, "reason": "No PDF chunks available", "selected_chunks": []}

    chunks_summary = ""
    for i, c in enumerate(chunks):
        chunks_summary += f"\nChunk {i+1} [{c['filename']} - Page {c['page']}]:\n{c['text'][:300]}...\n"

    agent_prompt = f"""You are a RAG decision agent for a student tutor system.

Question: "{question}"
Subject: {subject}

Available PDF chunks:
{chunks_summary}

Decide:
1. Are these chunks RELEVANT to answer this question? (yes/no)
2. Which chunk numbers are most useful? (e.g. "1,3" or "none")

Reply in this exact format only:
USE_RAG: yes/no
CHUNKS: 1,2 or none
REASON: one short sentence"""

    try:
        response = gemini_check.generate_content(agent_prompt)
        text = response.text.strip()
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
    context_parts = []
    for c in decision["selected_chunks"]:
        context_parts.append(f"[{c['filename']} - Page {c['page']}]\n{c['text']}")
    return "\n\n".join(context_parts), True

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
# ── Semantic Cache System ──
# ══════════════════════════════════════════

def simple_vectorize(text: str) -> dict:
    words = re.findall(r"\w+", text.lower())
    vec = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    total = sum(vec.values()) or 1
    return {k: v / total for k, v in vec.items()}

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot    = sum(vec1[w] * vec2[w] for w in common)
    norm1  = sum(v ** 2 for v in vec1.values()) ** 0.5
    norm2  = sum(v ** 2 for v in vec2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def load_cache() -> list:
    return load_json(CACHE_FILE, [])

def save_cache(cache: list):
    save_json(CACHE_FILE, cache)

def cache_lookup(question: str, subject: str, language: str):
    cache = load_cache()
    now   = datetime.datetime.now()
    q_vec = simple_vectorize(question)

    valid_cache = []
    for entry in cache:
        try:
            cached_time = datetime.datetime.fromisoformat(entry["time"])
            if (now - cached_time).total_seconds() < CACHE_TTL_HOURS * 3600:
                valid_cache.append(entry)
        except:
            pass

    if len(valid_cache) != len(cache):
        save_cache(valid_cache)

    best_score = 0.0
    best_entry = None
    for entry in valid_cache:
        if entry.get("subject") != subject or entry.get("language") != language:
            continue
        cached_vec = simple_vectorize(entry["question"])
        score      = cosine_similarity(q_vec, cached_vec)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= CACHE_SIMILARITY_THRESHOLD and best_entry:
        return best_entry, best_score

    return None, 0.0

def cache_store(question: str, subject: str, language: str, answer: str, graph_url, verified: bool):
    cache = load_cache()

    q_vec = simple_vectorize(question)
    for entry in cache:
        if entry.get("subject") == subject and entry.get("language") == language:
            if cosine_similarity(q_vec, simple_vectorize(entry["question"])) >= CACHE_SIMILARITY_THRESHOLD:
                return

    cache.append({
        "question":  question,
        "subject":   subject,
        "language":  language,
        "answer":    answer,
        "graph_url": graph_url,
        "verified":  verified,
        "time":      datetime.datetime.now().isoformat()
    })

    if len(cache) > CACHE_MAX_SIZE:
        cache = cache[-CACHE_MAX_SIZE:]

    save_cache(cache)

def check_admin(x_admin_password: str = "") -> bool:
    return x_admin_password == ADMIN_PASSWORD

# ══════════════════════════════════════════
# ── Backend Authentication API (🔐 NEW) ──
# ── Frontend passwords replace කරනවා ──
# ══════════════════════════════════════════
USERS_FILE = "users.json"

def get_users():
    """Load users — first time default users create කරනවා"""
    default = {
        "sanduni": {
            "password_hash": hashlib.sha256("momsanzdad2001#".encode()).hexdigest(),
            "role": "admin"
        },
        "hansika": {
            "password_hash": hashlib.sha256("sanz2024".encode()).hexdigest(),
            "role": "user"
        }
    }
    users = load_json(USERS_FILE, None)
    if users is None:
        save_json(USERS_FILE, default)
        return default
    return users

class LoginRequest(BaseModel):
    name: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    password: str
    role: str = "user"

@app.post("/auth/login")
async def login(body: LoginRequest):
    """Frontend එකෙන් login — password backend එකේ check"""
    users = get_users()
    name = body.name.strip().lower()

    if name not in users:
        raise HTTPException(status_code=401, detail="User not found")

    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if users[name]["password_hash"] != pw_hash:
        raise HTTPException(status_code=401, detail="Wrong password")

    # Session token generate
    token = secrets.token_hex(32)
    return {
        "status": "success",
        "token": token,
        "role": users[name]["role"],
        "name": name
    }

@app.post("/auth/register")
async def register(body: RegisterRequest, x_admin_password: str = Header(default="")):
    """Admin විතරක් new users add කරන්න පුළුවන්"""
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Only admin can register users")

    users = get_users()
    name = body.name.strip().lower()

    if name in users:
        raise HTTPException(status_code=400, detail="User already exists")

    users[name] = {
        "password_hash": hashlib.sha256(body.password.encode()).hexdigest(),
        "role": body.role
    }
    save_json(USERS_FILE, users)
    return {"status": "success", "message": f"{name} registered!"}

# ══════════════════════════════════════════
# ── 🎮 XP + Progress System ──
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

    p = progress[user_name]
    today = datetime.date.today().isoformat()

    p["total_questions"] = p.get("total_questions", 0) + 1
    if correct:
        p["correct_answers"] = p.get("correct_answers", 0) + 1
        p["xp"] = p.get("xp", 0) + xp_earned

    # Subject tracking
    if subject not in p.get("subjects", {}):
        p["subjects"][subject] = {"asked": 0, "correct": 0}
    p["subjects"][subject]["asked"] += 1
    if correct:
        p["subjects"][subject]["correct"] += 1

    # Daily count
    p.setdefault("daily_counts", {})
    p["daily_counts"][today] = p["daily_counts"].get(today, 0) + 1

    # Streak
    last = p.get("last_active_date")
    if last:
        diff = (datetime.date.today() - datetime.date.fromisoformat(last)).days
        if diff == 1:
            p["streak"] = p.get("streak", 0) + 1
        elif diff > 1:
            p["streak"] = 1
    else:
        p["streak"] = 1
    p["last_active_date"] = today

    # Badges
    badges = p.get("badges", [])
    new_badge = None
    xp = p.get("xp", 0)
    total_q = p["total_questions"]
    streak = p.get("streak", 0)

    badge_rules = [
        (total_q >= 1,    "🌱 Beginner"),
        (total_q >= 10,   "📚 Scholar"),
        (total_q >= 50,   "🔥 Dedicated"),
        (total_q >= 100,  "🏆 Champion"),
        (total_q >= 250,  "💫 Legend"),
        (streak >= 3,     "⚡ 3-Day Streak"),
        (streak >= 7,     "🌟 Week Streak"),
        (streak >= 30,    "👑 Month Streak"),
        (xp >= 100,       "💎 XP 100"),
        (xp >= 500,       "🔶 XP 500"),
        (xp >= 1000,      "💠 XP 1000"),
    ]
    for condition, badge in badge_rules:
        if condition and badge not in badges:
            badges.append(badge)
            new_badge = badge
            break  # one badge at a time

    p["badges"] = badges
    progress[user_name] = p
    save_json(PROGRESS_FILE, progress)
    return new_badge

def update_leaderboard(user_name: str, grade: str):
    lb = load_json(LEADERBOARD_FILE, {})
    progress = load_json(PROGRESS_FILE, {})
    if user_name in progress:
        p = progress[user_name]
        lb[user_name] = {
            "xp": p.get("xp", 0), "streak": p.get("streak", 0),
            "total_q": p.get("total_questions", 0),
            "badges": len(p.get("badges", [])),
            "grade": grade,
            "updated": datetime.datetime.now().isoformat()
        }
        save_json(LEADERBOARD_FILE, lb)

@app.get("/progress/{user_name}")
async def get_student_progress(user_name: str):
    return {"status": "success", "progress": get_progress(user_name.lower())}

@app.get("/leaderboard")
async def get_leaderboard(grade: str = ""):
    lb = load_json(LEADERBOARD_FILE, {})
    entries = list(lb.items())
    if grade:
        entries = [(k, v) for k, v in entries if str(v.get("grade", "")) == grade]
    entries.sort(key=lambda x: x[1].get("xp", 0), reverse=True)
    return {"status": "success", "leaderboard": [
        {"rank": i+1, "name": name, **data}
        for i, (name, data) in enumerate(entries[:20])
    ]}

# ══════════════════════════════════════════
# ── 📝 Quiz System ──
# ══════════════════════════════════════════

class QuizStartRequest(BaseModel):
    user_name: str
    grade: str
    subject: str
    language: str = "si"

class QuizAnswerRequest(BaseModel):
    session_id: str
    user_name: str
    user_answer: str

@app.post("/quiz/start")
async def start_quiz(body: QuizStartRequest):
    lang = body.language if body.language in LANG_INSTRUCTIONS else "en"
    difficulty = "easy" if int(body.grade) <= 5 else "medium" if int(body.grade) <= 9 else "hard"

    prompt = f"""Generate exactly 5 multiple-choice quiz questions for:
Grade: {body.grade}, Subject: {body.subject}, Difficulty: {difficulty}
Language: {body.language}

Return ONLY valid JSON:
{{"questions":[{{"q":"Question","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"answer":"A","explanation":"Why A is correct"}}]}}"""

    try:
        response = gemini_flash.generate_content(prompt)
        raw = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
        data = json.loads(raw)
        questions = data.get("questions", [])
        if not questions:
            raise ValueError("No questions")
    except Exception as e:
        return {"status": "error", "message": f"Quiz generation failed: {e}"}

    session_id = secrets.token_hex(8)
    sessions = load_json(QUIZ_FILE, {})
    sessions[session_id] = {
        "user_name": body.user_name, "grade": body.grade,
        "subject": body.subject, "language": lang,
        "questions": questions, "current": 0,
        "score": 0, "answers": [],
        "started": datetime.datetime.now().isoformat(),
        "completed": False
    }
    save_json(QUIZ_FILE, sessions)

    q = questions[0]
    return {
        "status": "success", "session_id": session_id,
        "total": len(questions), "current": 1,
        "question": q["q"], "options": q["options"],
        "progress_pct": 0
    }

@app.post("/quiz/answer")
async def answer_quiz(body: QuizAnswerRequest):
    sessions = load_json(QUIZ_FILE, {})
    if body.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Quiz not found")

    sess = sessions[body.session_id]
    if sess["completed"]:
        raise HTTPException(status_code=400, detail="Quiz already done")

    idx = sess["current"]
    q = sess["questions"][idx]
    correct = body.user_answer.strip().upper().startswith(q["answer"].upper())

    if correct:
        sess["score"] += 1

    sess["answers"].append({
        "question": q["q"], "user_answer": body.user_answer,
        "correct": correct, "right_answer": q["answer"],
        "explanation": q.get("explanation", "")
    })
    sess["current"] += 1

    # Quiz completed?
    if sess["current"] >= len(sess["questions"]):
        sess["completed"] = True
        sessions[body.session_id] = sess
        save_json(QUIZ_FILE, sessions)

        # Award XP
        xp_earned = sess["score"] * 15
        user = body.user_name.lower()
        progress = load_json(PROGRESS_FILE, {})
        if user not in progress:
            get_progress(user)
            progress = load_json(PROGRESS_FILE, {})
        p = progress.get(user, {})
        p["xp"] = p.get("xp", 0) + xp_earned
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

    # Next question
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
# ── 🎨 AI Image Generation ──
# ══════════════════════════════════════════

class ImageGenRequest(BaseModel):
    prompt: str
    subject: str = "General"
    language: str = "en"

@app.post("/generate-image")
async def generate_image(body: ImageGenRequest):
    """AI image generation using Gemini's image capabilities"""
    try:
        # Gemini vision model for educational diagrams
        img_prompt = f"""Create a clear, educational diagram/illustration for this topic:
"{body.prompt}"
Subject: {body.subject}

Requirements:
- Simple, clean, student-friendly style
- Use labels and arrows where needed
- Educational and easy to understand
- Suitable for Grade school students

Describe the image in detail so it can be visualized, then provide matplotlib code to draw it.
Put the code between [GRAPH_START] and [GRAPH_END] tags.
Do NOT use plt.savefig()."""

        response = gemini_flash.generate_content(img_prompt)
        answer = response.text

        # Extract and execute graph code
        graph_url = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', answer, re.DOTALL)
        description = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', answer, flags=re.DOTALL).strip()

        if graph_match:
            graph_code = graph_match.group(1).strip()
            try:
                plt.figure(figsize=(8, 6))
                safe_globals = {
                    "__builtins__": {},
                    "plt": plt, "np": np,
                    "range": range, "len": len, "zip": zip,
                    "list": list, "tuple": tuple,
                    "int": int, "float": float, "str": str,
                    "abs": abs, "min": min, "max": max,
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

        return {
            "status": "success",
            "description": description[:500],
            "image_url": graph_url
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ══════════════════════════════════════════
# ── Health Check ──
# ══════════════════════════════════════════
@app.get("/health")
async def health_check():
    return {
        "status":             "ok",
        "message":            "Server is running! 🚀",
        "stt_support":        STT_SUPPORT,
        "tts_support":        TTS_SUPPORT,
        "pdf_support":        PDF_SUPPORT,
        "cloudinary_support": CLOUDINARY_SUPPORT,
        "pinecone_support":   PINECONE_SUPPORT
    }

# ══════════════════════════════════════════
# ── Voice Input (STT) ──
# ══════════════════════════════════════════
@app.post("/voice-input")
async def voice_input(audio: UploadFile = File(...)):
    if not STT_SUPPORT:
        raise HTTPException(status_code=500, detail="STT library not installed. Run: pip install SpeechRecognition")

    temp_path = os.path.join("temp_audio", "input.wav")
    contents  = await audio.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        recognizer   = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)

        detected_text = None
        detected_lang = "en"

        try:
            text = recognizer.recognize_google(audio_data, language="si-LK")
            if text:
                detected_text = text
                detected_lang = "si"
        except:
            pass

        if not detected_text:
            try:
                text = recognizer.recognize_google(audio_data, language="ta-LK")
                if text:
                    detected_text = text
                    detected_lang = "ta"
            except:
                pass

        if not detected_text:
            try:
                text = recognizer.recognize_google(audio_data, language="en-US")
                if text:
                    detected_text = text
                    detected_lang = "en"
            except:
                pass

        try:
            os.remove(temp_path)
        except:
            pass

        if not detected_text:
            raise HTTPException(status_code=400, detail="Could not understand audio. Please try again.")

        text_lang = detect_language(detected_text)
        if text_lang != "en":
            detected_lang = text_lang

        return {"status": "success", "text": detected_text, "detected_language": detected_lang}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════
# ── TTS (Text to Speech) ──
# ══════════════════════════════════════════
class TTSRequest(BaseModel):
    text: str
    language: str = "si"

@app.post("/tts")
async def text_to_speech(body: TTSRequest):
    if not TTS_SUPPORT:
        raise HTTPException(status_code=500, detail="TTS library not installed. Run: pip install gTTS")

    text     = body.text
    language = body.language

    if not text:
        raise HTTPException(status_code=400, detail="No text provided!")

    if language not in LANG_INSTRUCTIONS:
        language = detect_language(text)

    gtts_lang  = LANG_INSTRUCTIONS[language]['gtts_lang']
    clean_text = re.sub(r'[*_#`\[\]()~>]', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    if len(clean_text) > 1000:
        clean_text = clean_text[:1000] + "..."

    try:
        tts          = gTTS(text=clean_text, lang=gtts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════
# ── Main Solve  (✅ CONVERSATION MEMORY ADDED) ──
# ══════════════════════════════════════════
@app.post("/solve")
async def solve_math(
    name:     str        = Form(default="User"),
    grade:    str        = Form(default="N/A"),
    subject:  str        = Form(default="Mathematics"),
    question: str        = Form(default=""),
    language: str        = Form(default="si"),
    image:    UploadFile = File(default=None),
    conversation_history: str = Form(default="[]"),   # ✅ NEW — conversation memory
):
    try:
        if language not in LANG_INSTRUCTIONS:
            language = detect_language(question)

        lang_cfg = LANG_INSTRUCTIONS[language]

        # 1. Blacklist
        banned, remaining = is_blacklisted(name)
        if banned:
            msgs = {
                "si": f"⛔ ඔයා {remaining} minutes ගෙවෙන තෙක් use කරන්න බෑ!",
                "en": f"⛔ You are blocked for {remaining} more minutes!",
                "ta": f"⛔ இன்னும் {remaining} நிமிடங்கள் பயன்படுத்த முடியாது!"
            }
            return {"status": "banned", "message": msgs.get(language, msgs["si"])}

        # 2. Creator
        creator_names = ["sanduni", "hansika", "sanduni hansika", "sanz", "sanz queen"]
        if any(n in question.lower() for n in creator_names):
            return {
                "status": "creator",
                "message": (
                    "අනේ... ඔයා Sanz Queen ගැන දන්නවාද? 👑\n\n"
                    "ඒ කෙනා තමයි මේ සම්පූර්ණ platform එකම හැදුවේ. "
                    "Backend, frontend, AI integration — ඔක්කොම තනියම. 💜\n\n"
                    "Please — මේ platform එක misuse කරන්න එපා. 🙏"
                )
            }

        # 3. Bad Words
        if contains_bad_words(question):
            add_to_blacklist(name)
            msgs = {
                "si": "⛔ නුසුදුසු වචන use කළා! ඔයා පැය 2ක් blacklist කරලා!",
                "en": "⛔ Inappropriate words! You are blacklisted for 2 hours!",
                "ta": "⛔ தகாத வார்த்தைகள்! 2 மணி நேரம் தடைசெய்யப்பட்டீர்கள்!"
            }
            return {"status": "banned", "message": msgs.get(language, msgs["si"])}

        # 4. Rage
        rage_warning = ""
        if is_rage(question):
            rage_msgs = {
                "si": "⚠️ කරුණාකර සංසුන්ව ප්‍රශ්නය අහන්න! ",
                "en": "⚠️ Please ask calmly! ",
                "ta": "⚠️ தயவுசெய்து அமைதியாக கேளுங்கள்! "
            }
            rage_warning = rage_msgs.get(language, "")

        # 5. Semantic Cache Check (only for first message — no conversation context)
        conv_history = []
        try:
            conv_history = json.loads(conversation_history)
            conv_history = conv_history[-10:]  # Last 10 messages only
        except:
            conv_history = []

        # Cache only if NO conversation history (first question)
        if not conv_history:
            cached_entry, cache_score = cache_lookup(question, subject, language)
            if cached_entry:
                save_history(name, grade, subject, question, cached_entry["answer"])
                update_stats(subject)
                return {
                    "status":            "success",
                    "answer":            rage_warning + cached_entry["answer"],
                    "graph_url":         cached_entry.get("graph_url"),
                    "verified":          cached_entry.get("verified", True),
                    "rag_used":          False,
                    "cache_hit":         True,
                    "cache_score":       round(cache_score, 3),
                    "detected_language": language
                }

        # 6. RAG
        rag_context, rag_used = get_rag_context(question, grade)

        # ✅ NEW — Build conversation context string for Gemini
        conversation_context = ""
        if conv_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conv_history:
                role = msg.get("role", "")
                text = msg.get("text", "")[:300]  # Limit each message
                if role == "user":
                    conversation_context += f"Student: {text}\n"
                elif role == "ai":
                    conversation_context += f"Tutor: {text}\n"
            conversation_context += "\nNow the student asks a follow-up. Use the conversation above for context. Continue naturally.\n"

        # 7. Instruction — with conversation context
        is_math_subject = any(w in subject.lower() for w in ["math", "maths", "ගණිත", "கணித"])
        if is_math_subject:
            instruction = f"""
            {lang_cfg['socratic']}
            Student: {name}, Grade {grade}, Subject: {subject}.
            {conversation_context}
            If a graph is needed, provide matplotlib code between [GRAPH_START] and [GRAPH_END]. Do NOT use plt.savefig().
            {f'Use this textbook reference: {rag_context}' if rag_context else ''}
            """
        else:
            instruction = f"""
            You are an expert {subject} teacher. Student: {name}, Grade {grade}.
            {lang_cfg['instruction']}
            {conversation_context}
            Solve step by step. If a graph is needed, provide matplotlib code between [GRAPH_START] and [GRAPH_END]. Do NOT use plt.savefig().
            {f'Use this textbook reference: {rag_context}' if rag_context else ''}
            """

        content_list = [f"User: {name}, Grade: {grade}, Subject: {subject}. Question: {question}"]

        if image and image.filename:
            image_data = await image.read()
            content_list.append({"mime_type": "image/jpeg", "data": image_data})

        # 8. Gemini Flash - Main Answer
        response1 = gemini_flash.generate_content([instruction] + content_list)
        answer1   = response1.text

        # 9. Cross Check — ⚡ MATH ONLY (saves API quota!)
        if is_math_subject:
            cross_check_prompt = f"""
            Is this answer correct?
            Question: {question}
            Answer: {answer1[:500]}
            {lang_cfg['cross_check']}
            """
            cross_response = gemini_check.generate_content(cross_check_prompt)
            cross_check    = cross_response.text
            correct_word   = lang_cfg['correct_word']

            if correct_word in cross_check:
                final_answer = answer1
                verified     = True
            else:
                socratic_redirect_prompt = f"""
                {lang_cfg['socratic']}
                {conversation_context}
                The student answered this question: "{question}"
                Their answer has some issues. WITHOUT saying "wrong" or "incorrect" directly,
                ask ONE gentle guiding question to help them reconsider their approach.
                Be warm and encouraging. Keep it to 2-3 sentences max.
                """
                redirect_resp = gemini_check.generate_content(socratic_redirect_prompt)
                final_answer  = redirect_resp.text
                verified = False
        else:
            # Non-math — skip cross-check, direct answer (saves 1 API call!)
            final_answer = answer1
            verified     = True

        # 10. Graph — 🔒 SANDBOXED exec
        graph_url   = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', final_answer, re.DOTALL)
        if graph_match:
            graph_code   = graph_match.group(1).strip()
            final_answer = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', final_answer, flags=re.DOTALL)
            try:
                plt.figure(figsize=(6, 4))
                # 🔒 Sandboxed — plt, np විතරක් allow, os/system/import block
                safe_globals = {
                    "__builtins__": {},
                    "plt": plt,
                    "np": np,
                    "range": range,
                    "len": len,
                    "zip": zip,
                    "list": list,
                    "tuple": tuple,
                    "int": int,
                    "float": float,
                    "str": str,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "round": round,
                    "enumerate": enumerate,
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

        save_history(name, grade, subject, question, final_answer)
        update_stats(subject)

        # 🎮 XP + Progress tracking
        new_badge = update_progress(name.lower(), subject, verified, xp_earned=5 if verified else 2)
        update_leaderboard(name.lower(), grade)

        # Cache store — only first messages (no conversation), non-math
        if not conv_history and not (image and image.filename) and not is_math_subject:
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
        # ⚡ Rate limit — friendly message instead of scary error
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
# ── Admin Routes ──
# ══════════════════════════════════════════
@app.get("/admin/stats")
async def admin_stats(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    stats     = load_json(STATS_FILE, {"total": 0, "subjects": {}})
    history   = load_json(HISTORY_FILE, [])
    blacklist = load_json(BLACKLIST_FILE, {})
    return {
        "status": "success",
        "total_questions": stats["total"],
        "subjects": stats["subjects"],
        "recent_users": list(set([h["user"] for h in history[-50:]])),
        "blacklisted_users": list(blacklist.keys())
    }

@app.get("/admin/history")
async def admin_history(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    history = load_json(HISTORY_FILE, [])
    return {"status": "success", "history": history[-100:]}

class BlacklistRemoveRequest(BaseModel):
    user_name: str

@app.post("/admin/blacklist/remove")
async def admin_remove_blacklist(
    body: BlacklistRemoveRequest,
    x_admin_password: str = Header(default="")
):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    blacklist = load_json(BLACKLIST_FILE, {})
    if body.user_name in blacklist:
        del blacklist[body.user_name]
        save_json(BLACKLIST_FILE, blacklist)
    return {"status": "success", "message": f"{body.user_name} removed!"}

# ✅ NEW — missing endpoint fix
@app.post("/admin/blacklist/add")
async def admin_add_blacklist(
    body: BlacklistRemoveRequest,
    x_admin_password: str = Header(default="")
):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    add_to_blacklist(body.user_name)
    return {"status": "success", "message": f"{body.user_name} banned for 2 hours!"}

@app.post("/admin/upload_pdf")
async def admin_upload_pdf(
    pdf: UploadFile = File(...),
    x_admin_password: str = Header(default="")
):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")

    contents = await pdf.read()
    filename = pdf.filename

    with open(os.path.join(PDF_FOLDER, filename), "wb") as f:
        f.write(contents)

    cloud_url = ""
    if CLOUDINARY_SUPPORT:
        cloud_url = upload_pdf_to_cloudinary(contents, filename)

    indexed = False
    if PINECONE_SUPPORT and pinecone_index:
        indexed = index_pdf_to_pinecone(contents, filename)

    return {
        "status":      "success",
        "message":     f"{filename} uploaded!",
        "cloud_url":   cloud_url,
        "pinecone_indexed": indexed
    }

@app.delete("/admin/pdfs/{filename}")
async def admin_delete_pdf(
    filename: str,
    x_admin_password: str = Header(default="")
):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    local_path = os.path.join(PDF_FOLDER, filename)
    if os.path.exists(local_path):
        os.remove(local_path)
    if CLOUDINARY_SUPPORT:
        try:
            cloudinary.uploader.destroy(f"sanz_textbooks/{filename.replace('.pdf','')}", resource_type="raw")
        except:
            pass
    return {"status": "success", "message": f"{filename} deleted!"}

@app.get("/admin/pdfs")
async def admin_list_pdfs(x_admin_password: str = Header(default="")):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    return {"status": "success", "pdfs": pdfs}

# ══════════════════════════════════════════
# ── Run ──
# ══════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
