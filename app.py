import os
import io
import base64
import re
import json
import datetime

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

# ══════════════════════════════════════════
# ── App Setup ──
# ══════════════════════════════════════════
app = FastAPI(title="Sanz AI Tutor", version="2.0.0")

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
gemini_flash = genai.GenerativeModel('gemini-2.5-flash-lite')
gemini_check = genai.GenerativeModel('gemini-2.5-flash-lite')

HISTORY_FILE   = "history.json"
BLACKLIST_FILE = "blacklist.json"
PDF_FOLDER     = "textbooks"
STATS_FILE     = "stats.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# ══════════════════════════════════════════
# ── Language Instructions ──
# ══════════════════════════════════════════
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
# ── Agentic RAG System ──
# ══════════════════════════════════════════

def get_all_pdf_chunks(question: str) -> list:
    """PDF files වලින් relevant chunks හොයාගන්නවා"""
    if not PDF_SUPPORT:
        return []
    chunks = []
    keywords = [kw for kw in question.lower().split() if len(kw) > 3]
    for filename in os.listdir(PDF_FOLDER):
        if not filename.endswith(".pdf"):
            continue
        try:
            doc = fitz.open(os.path.join(PDF_FOLDER, filename))
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    continue
                score = sum(1 for kw in keywords if kw in text.lower())
                if score > 0:
                    chunks.append({
                        "filename": filename,
                        "page": page_num + 1,
                        "text": text[:1500],
                        "score": score
                    })
        except:
            pass
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:5]

def agent_decide_rag(question: str, subject: str, chunks: list) -> dict:
    """AI Agent එකම decide කරනවා — PDF use කරන්නද නැද්ද"""
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
    """Agentic RAG — agent decide කරලා relevant context return කරනවා"""
    chunks = get_all_pdf_chunks(question)
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

def check_admin(x_admin_password: str = "") -> bool:
    return x_admin_password == ADMIN_PASSWORD

# ══════════════════════════════════════════
# ── Health Check ──
# ══════════════════════════════════════════
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Server is running! 🚀",
        "stt_support": STT_SUPPORT,
        "tts_support": TTS_SUPPORT,
        "pdf_support": PDF_SUPPORT
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

        # Try Sinhala first
        try:
            text = recognizer.recognize_google(audio_data, language="si-LK")
            if text:
                detected_text = text
                detected_lang = "si"
        except:
            pass

        # Try Tamil
        if not detected_text:
            try:
                text = recognizer.recognize_google(audio_data, language="ta-LK")
                if text:
                    detected_text = text
                    detected_lang = "ta"
            except:
                pass

        # Try English
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
# ── Main Solve ──
# ══════════════════════════════════════════
@app.post("/solve")
async def solve_math(
    name:     str        = Form(default="User"),
    grade:    str        = Form(default="N/A"),
    subject:  str        = Form(default="Mathematics"),
    question: str        = Form(default=""),
    language: str        = Form(default="si"),
    image:    UploadFile = File(default=None),
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

        # 5. RAG
        rag_context, rag_used = get_rag_context(question, grade)

        # 6. Instruction
        instruction = f"""
        You are an expert {subject} teacher. Student: {name}, Grade {grade}.
        {lang_cfg['instruction']}
        Solve step by step. If a graph is needed, provide matplotlib code between [GRAPH_START] and [GRAPH_END]. Do NOT use plt.savefig().
        {f'Use this textbook reference: {rag_context}' if rag_context else ''}
        """

        content_list = [f"User: {name}, Grade: {grade}, Subject: {subject}. Question: {question}"]

        if image and image.filename:
            image_data = await image.read()
            content_list.append({"mime_type": "image/jpeg", "data": image_data})

        # 7. Gemini Flash - Main Answer
        response1 = gemini_flash.generate_content([instruction] + content_list)
        answer1   = response1.text

        # 8. Gemini Cross Check
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
            final_answer = cross_check
            verified     = False

        # 9. Graph
        graph_url   = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', final_answer, re.DOTALL)
        if graph_match:
            graph_code   = graph_match.group(1).strip()
            final_answer = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', final_answer, flags=re.DOTALL)
            try:
                plt.figure(figsize=(6, 4))
                exec(graph_code)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                plt.close()
            except Exception as e:
                print(f"Graph error: {e}")

        save_history(name, grade, subject, question, final_answer)
        update_stats(subject)

        return {
            "status": "success",
            "answer": rage_warning + final_answer,
            "graph_url": graph_url,
            "verified": verified,
            "rag_used": rag_used,
            "detected_language": language
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


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

@app.post("/admin/upload_pdf")
async def admin_upload_pdf(
    pdf: UploadFile = File(...),
    x_admin_password: str = Header(default="")
):
    if not check_admin(x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    contents = await pdf.read()
    with open(os.path.join(PDF_FOLDER, pdf.filename), "wb") as f:
        f.write(contents)
    return {"status": "success", "message": f"{pdf.filename} uploaded!"}

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
