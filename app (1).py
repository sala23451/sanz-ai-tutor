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

# Environment Variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set!")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel('gemini-2.0-flash-lite')
gemini_check = genai.GenerativeModel('gemini-2.0-flash-lite')

HISTORY_FILE   = "history.json"
BLACKLIST_FILE = "blacklist.json"
PDF_FOLDER      = "textbooks"
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

# ── Helpers ──
def detect_language(text):
    sinhala_count = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    tamil_count   = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if sinhala_count > 2: return "si"
    elif tamil_count > 2: return "ta"
    else: return "en"

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

# ── Security & Stats ──
BAD_WORDS = ["fuck", "shit", "bitch", "ass", "හුත්ත", "පුකේ", "මෝඩ", "පට්ට", "හුකන"] # කෙටි කරන ලදි

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
    save_json(HISTORY_FILE, history[-500:])

def get_rag_context(question, grade):
    if not PDF_SUPPORT: return ""
    context_parts = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(os.path.join(PDF_FOLDER, filename))
                for page in doc:
                    text = page.get_text()
                    if any(kw in text.lower() for kw in question.lower().split() if len(kw) > 3):
                        context_parts.append(f"[{filename}]\n{text[:1000]}")
                        break
            except: pass
    return "\n\n".join(context_parts[:3])

# ── Routes ──
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "stt": STT_SUPPORT, "tts": TTS_SUPPORT})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'si')
        gtts_lang = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS['si'])['gtts_lang']
        clean_text = re.sub(r'[*_#`\[\]()~>]', '', text)
        tts = gTTS(text=clean_text[:1000], lang=gtts_lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return send_file(audio_buffer, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/solve', methods=['POST'])
def solve_math():
    try:
        # Handle both JSON and Form Data
        if request.is_json:
            data = request.json
        else:
            data = request.form

        user_name  = data.get('name', 'User')
        user_grade = data.get('grade', 'N/A')
        subject    = data.get('subject', 'Mathematics')
        question   = data.get('question', '')
        language   = data.get('language', 'si')
        is_summary = data.get('is_summary_request', False)
        image_file = request.files.get('image') if not request.is_json else None

        if language not in LANG_INSTRUCTIONS:
            language = detect_language(question)
        lang_cfg = LANG_INSTRUCTIONS[language]

        # 1. Summary logic (Quick response for voice)
        if is_summary:
            summary_prompt = f"Summarize this answer into 2 simple sentences in {lang_cfg['gtts_lang']}. No markdown."
            response = gemini_flash.generate_content([summary_prompt, question])
            return jsonify({"status": "success", "answer": response.text.strip(), "is_summary": True})

        # 2. Safety checks
        banned, remaining = is_blacklisted(user_name)
        if banned: return jsonify({"status": "banned", "message": f"Blocked for {remaining} mins"})
        if contains_bad_words(question):
            add_to_blacklist(user_name)
            return jsonify({"status": "banned", "message": "Bad words detected!"})

        # 3. Main AI Logic
        rag_context = get_rag_context(question, user_grade)
        instruction = f"You are an expert {subject} teacher for Grade {user_grade}. {lang_cfg['instruction']} Solve step by step. If a graph is needed, use matplotlib code between [GRAPH_START] and [GRAPH_END]."
        
        content = [f"Question: {question}\nContext: {rag_context}"]
        if image_file:
            content.append({"mime_type": "image/jpeg", "data": image_file.read()})

        response = gemini_flash.generate_content([instruction] + content)
        final_answer = response.text

        # 4. Graph Processing
        graph_url = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', final_answer, re.DOTALL)
        if graph_match:
            graph_code = graph_match.group(1).strip()
            final_answer = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', final_answer, flags=re.DOTALL)
            try:
                plt.figure(figsize=(6, 4))
                exec(graph_code)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                plt.close()
            except: pass

        save_history(user_name, user_grade, subject, question, final_answer)
        update_stats(subject)

        return jsonify({
            "status": "success",
            "answer": final_answer,
            "graph_url": graph_url,
            "detected_language": language
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
