import os
import io
import base64
import re
import json
import datetime
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
CORS(app)

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

# ── Language Instructions ──
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

# ── Language Detection ──
def detect_language(text):
    """Auto detect Sinhala / Tamil / English"""
    sinhala_count = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    tamil_count   = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if sinhala_count > 2:
        return "si"
    elif tamil_count > 2:
        return "ta"
    else:
        return "en"

# ── JSON Helpers ──
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
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

def contains_bad_words(text):
    text_lower = text.lower()
    return any(w in text_lower for w in BAD_WORDS)

def is_blacklisted(user_name):
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

def add_to_blacklist(user_name):
    blacklist = load_json(BLACKLIST_FILE, {})
    ban_until = datetime.datetime.now() + datetime.timedelta(hours=2)
    blacklist[user_name] = ban_until.isoformat()
    save_json(BLACKLIST_FILE, blacklist)

RAGE_WORDS = ["stupid", "idiot", "useless", "hate", "terrible", "worst",
              "මෝඩ", "නිකම්", "වැඩක් නෑ", "අකාරයි", "හොඳ නෑ"]

def is_rage(text):
    text_lower = text.lower()
    return sum(1 for w in RAGE_WORDS if w in text_lower) >= 2

def get_rag_context(question, grade):
    if not PDF_SUPPORT:
        return ""
    context_parts = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(os.path.join(PDF_FOLDER, filename))
                for page in doc:
                    text = page.get_text()
                    keywords = question.lower().split()
                    if any(kw in text.lower() for kw in keywords if len(kw) > 3):
                        context_parts.append(f"[{filename}]\n{text[:1000]}")
                        break
            except:
                pass
    return "\n\n".join(context_parts[:3])

def update_stats(subject):
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
# ── Health Check ──
# ══════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Server is running! 🚀",
        "stt_support": STT_SUPPORT,
        "tts_support": TTS_SUPPORT,
        "pdf_support": PDF_SUPPORT
    })

# ══════════════════════════════════════════
# ── Voice Input (STT) ──
# ══════════════════════════════════════════
@app.route('/voice-input', methods=['POST'])
def voice_input():
    """
    🎤 Record button → Frontend sends audio blob → Backend converts to text
    Returns: { text, detected_language }
    """
    try:
        if not STT_SUPPORT:
            return jsonify({
                "status": "error",
                "message": "STT library not installed. Run: pip install SpeechRecognition"
            }), 500

        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"status": "error", "message": "No audio file received!"}), 400

        # Save temp audio file
        temp_path = os.path.join("temp_audio", "input.wav")
        audio_file.save(temp_path)

        # Speech to Text
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)

        # Try all 3 languages
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

        # Cleanup temp file
        try:
            os.remove(temp_path)
        except:
            pass

        if not detected_text:
            return jsonify({
                "status": "error",
                "message": "Could not understand audio. Please try again."
            }), 400

        # Also verify language from text
        text_lang = detect_language(detected_text)
        if text_lang != "en":
            detected_lang = text_lang

        return jsonify({
            "status": "success",
            "text": detected_text,
            "detected_language": detected_lang
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════
# ── TTS (Text to Speech) ──
# ══════════════════════════════════════════
@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    ▶️ Play button click → Frontend sends text + language → Returns audio file
    Returns: MP3 audio stream
    """
    try:
        if not TTS_SUPPORT:
            return jsonify({
                "status": "error",
                "message": "TTS library not installed. Run: pip install gTTS"
            }), 500

        data     = request.json
        text     = data.get('text', '')
        language = data.get('language', 'si')

        if not text:
            return jsonify({"status": "error", "message": "No text provided!"}), 400

        if language not in LANG_INSTRUCTIONS:
            language = detect_language(text)

        gtts_lang = LANG_INSTRUCTIONS[language]['gtts_lang']

        # Clean text — remove markdown symbols for better TTS
        clean_text = re.sub(r'[*_#`\[\]()~>]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Limit text length for TTS (too long = slow)
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."

        # Generate TTS
        tts = gTTS(text=clean_text, lang=gtts_lang, slow=False)

        # Save to buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='response.mp3'
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════
# ── Main Solve ──
# ══════════════════════════════════════════
@app.route('/solve', methods=['POST'])
def solve_math():
    try:
        user_name  = request.form.get('name', 'User')
        user_grade = request.form.get('grade', 'N/A')
        subject    = request.form.get('subject', 'Mathematics')
        question   = request.form.get('question', '')
        language   = request.form.get('language', 'si')
        image_file = request.files.get('image')

        # Auto-detect language if not provided
        if language not in LANG_INSTRUCTIONS:
            language = detect_language(question)

        lang_cfg = LANG_INSTRUCTIONS[language]

        # 1. Blacklist
        banned, remaining = is_blacklisted(user_name)
        if banned:
            msgs = {
                "si": f"⛔ ඔයා {remaining} minutes ගෙවෙන තෙක් use කරන්න බෑ!",
                "en": f"⛔ You are blocked for {remaining} more minutes!",
                "ta": f"⛔ இன்னும் {remaining} நிமிடங்கள் பயன்படுத்த முடியாது!"
            }
            return jsonify({"status": "banned", "message": msgs.get(language, msgs["si"])})

        # 2. Creator
        creator_names = ["sanduni", "hansika", "sanduni hansika", "sanz", "sanz queen"]
        if any(name in question.lower() for name in creator_names):
            return jsonify({
                "status": "creator",
                "message": (
                    "අනේ... ඔයා Sanz Queen ගැන දන්නවාද? 👑\n\n"
                    "ඒ කෙනා තමයි මේ සම්පූර්ණ platform එකම හැදුවේ. "
                    "Backend, frontend, AI integration — ඔක්කොම තනියම. 💜\n\n"
                    "Please — මේ platform එක misuse කරන්න එපා. 🙏"
                )
            })

        # 3. Bad Words
        if contains_bad_words(question):
            add_to_blacklist(user_name)
            msgs = {
                "si": "⛔ නුසුදුසු වචන use කළා! ඔයා පැය 2ක් blacklist කරලා!",
                "en": "⛔ Inappropriate words! You are blacklisted for 2 hours!",
                "ta": "⛔ தகாத வார்த்தைகள்! 2 மணி நேரம் தடைசெய்யப்பட்டீர்கள்!"
            }
            return jsonify({"status": "banned", "message": msgs.get(language, msgs["si"])})

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
        rag_context = get_rag_context(question, user_grade)

        # 6. Instruction
        instruction = f"""
        You are an expert {subject} teacher. Student: {user_name}, Grade {user_grade}.
        {lang_cfg['instruction']}
        Solve step by step. If a graph is needed, provide matplotlib code between [GRAPH_START] and [GRAPH_END]. Do NOT use plt.savefig().
        {f'Use this textbook reference: {rag_context}' if rag_context else ''}
        """

        content_list = [f"User: {user_name}, Grade: {user_grade}, Subject: {subject}. Question: {question}"]
        if image_file:
            image_data = image_file.read()
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
            verified = True
        else:
            final_answer = cross_check
            verified = False

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

        save_history(user_name, user_grade, subject, question, final_answer)
        update_stats(subject)

        return jsonify({
            "status": "success",
            "answer": rage_warning + final_answer,
            "graph_url": graph_url,
            "verified": verified,
            "rag_used": bool(rag_context),
            "detected_language": language  # Frontend එකට language return කරනවා TTS වලට
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ══════════════════════════════════════════
# ── Admin Routes ──
# ══════════════════════════════════════════
def check_admin(req):
    return req.headers.get("X-Admin-Password", "") == ADMIN_PASSWORD

@app.route('/admin/stats', methods=['GET'])
def admin_stats():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    stats     = load_json(STATS_FILE, {"total": 0, "subjects": {}})
    history   = load_json(HISTORY_FILE, [])
    blacklist = load_json(BLACKLIST_FILE, {})
    return jsonify({
        "status": "success",
        "total_questions": stats["total"],
        "subjects": stats["subjects"],
        "recent_users": list(set([h["user"] for h in history[-50:]])),
        "blacklisted_users": list(blacklist.keys())
    })

@app.route('/admin/history', methods=['GET'])
def admin_history():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    history = load_json(HISTORY_FILE, [])
    return jsonify({"status": "success", "history": history[-100:]})

@app.route('/admin/blacklist/remove', methods=['POST'])
def admin_remove_blacklist():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    data      = request.json
    user_name = data.get("user_name")
    blacklist = load_json(BLACKLIST_FILE, {})
    if user_name in blacklist:
        del blacklist[user_name]
        save_json(BLACKLIST_FILE, blacklist)
    return jsonify({"status": "success", "message": f"{user_name} removed!"})

@app.route('/admin/upload_pdf', methods=['POST'])
def admin_upload_pdf():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return jsonify({"status": "error", "message": "No PDF!"})
    pdf_file.save(os.path.join(PDF_FOLDER, pdf_file.filename))
    return jsonify({"status": "success", "message": f"{pdf_file.filename} uploaded!"})

@app.route('/admin/pdfs', methods=['GET'])
def admin_list_pdfs():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    return jsonify({"status": "success", "pdfs": pdfs})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
