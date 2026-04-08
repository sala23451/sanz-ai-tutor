import os
import io
import base64
import re
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import anthropic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# RAG (PDF) Support
# ============================================================
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

app = Flask(__name__)
CORS(app)

# ============================================================
# Environment Variables
# ============================================================
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ADMIN_PASSWORD    = os.environ.get("ADMIN_PASSWORD", "admin123")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable එක set කර නොමැත!")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable එක set කර නොමැත!")

# ============================================================
# AI Models
# ============================================================
genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel('gemini-2.5-flash')
gemini_pro   = genai.GenerativeModel('gemini-2.5-pro')
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ============================================================
# Data Files
# ============================================================
HISTORY_FILE   = "history.json"
BLACKLIST_FILE = "blacklist.json"
PDF_FOLDER     = "textbooks"
STATS_FILE     = "stats.json"

os.makedirs(PDF_FOLDER, exist_ok=True)

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
# Bad Words List (Sinhala + English)
# ============================================================
BAD_WORDS = [
    # English
    "fuck", "shit", "bitch", "ass", "bastard", "dick", "cock",
    "pussy", "cunt", "whore", "slut", "nigga", "faggot",
    "moron", "dumbass", "retard", "loser",

    # Sinhala
    "හුත්ත", "පුකේ", "මෝඩ", "හරක", "කුකු", "කපන්", "බේපන්",
    "ගනින්", "හූකර", "හාමු", "පට්ට හරක", "කෙල්ල", "කොල්ල",
    "ගොන්", "වාහල", "හුකන්", "හූකන", "පිස්සු",
    "උන්", "බල්ලො", "හූලං", "හූකල", "ගෑනු", "ගෑනි",

    # Singlish
    "hutta", "puke", "moda", "harak", "kuku", "kapan", "bepan",
    "ganin", "hookar", "pissu", "marana", "ballo", "hulan",
    "wahala", "goni", "kotiya"
]

def contains_bad_words(text):
    text_lower = text.lower()
    for word in BAD_WORDS:
        if word in text_lower:
            return True
    return False

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

# ============================================================
# Rage Detection
# ============================================================
RAGE_WORDS = [
    "stupid", "idiot", "useless", "hate", "terrible", "worst",
    "මෝඩ", "නිකම්", "වැඩක් නෑ", "අකාරයි", "හොඳ නෑ"
]

def is_rage(text):
    text_lower = text.lower()
    count = sum(1 for w in RAGE_WORDS if w in text_lower)
    return count >= 2

# ============================================================
# RAG — PDF Text Extraction
# ============================================================
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
                    # Simple keyword match
                    keywords = question.lower().split()
                    if any(kw in text.lower() for kw in keywords if len(kw) > 3):
                        context_parts.append(f"[{filename}]\n{text[:1000]}")
                        break
            except:
                pass
    return "\n\n".join(context_parts[:3])

# ============================================================
# Stats Update
# ============================================================
def update_stats(subject):
    stats = load_json(STATS_FILE, {"total": 0, "subjects": {}})
    stats["total"] += 1
    stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
    save_json(STATS_FILE, stats)

# ============================================================
# History Save
# ============================================================
def save_history(user_name, grade, subject, question, answer):
    history = load_json(HISTORY_FILE, [])
    history.append({
        "user": user_name,
        "grade": grade,
        "subject": subject,
        "question": question,
        "answer": answer[:500],
        "time": datetime.datetime.now().isoformat()
    })
    # Keep last 500 records
    history = history[-500:]
    save_json(HISTORY_FILE, history)

# ============================================================
# Main Solve Route
# ============================================================
@app.route('/solve', methods=['POST'])
def solve_math():
    try:
        user_name  = request.form.get('name', 'User')
        user_grade = request.form.get('grade', 'N/A')
        subject    = request.form.get('subject', 'ගණිතය')
        question   = request.form.get('question', '')
        image_file = request.files.get('image')

        # 1. Blacklist Check
        banned, remaining = is_blacklisted(user_name)
        if banned:
            return jsonify({
                "status": "banned",
                "message": f"⛔ ඔයා {remaining} minutes ගෙවෙන තෙක් use කරන්න බෑ!"
            })

        # 2. Creator Name Detection
        creator_names = ["sanduni", "hansika", "sanduni hansika", "sanz", "sanz queen"]
        question_lower = question.lower()
        if any(name in question_lower for name in creator_names):
            return jsonify({
                "status": "creator",
                "message": (
                    "අනේ... ඔයා Sanz Queen ගැන දන්නවාද? 👑\n\n"
                    "ඒ කෙනා තමයි මේ සම්පූර්ණ platform එකම හැදුවේ. "
                    "Backend, frontend, AI integration — ඔක්කොම තනියම. "
                    "ඇය හිතුවේ ලංකාවේ ළමයින්ට හොඳ AI teacher කෙනෙක් නෑ කියලා — "
                    "ඒ නිසා තනියම හදලා දුන්නා. 💜\n\n"
                    "ඉතින් please — මේ platform එක misuse කරන්න එපා. "
                    "ඇය ඔයාලාට උදව් කරන්න හැදුවා, ඔයාලා ඇයට respect දෙන්න. 🙏"
                )
            })

        # 3. Bad Words Check
        if contains_bad_words(question):
            add_to_blacklist(user_name)
            return jsonify({
                "status": "banned",
                "message": "⛔ නුසුදුසු වචන use කළා! ඔයා පැය 2ක් blacklist කරලා!"
            })

        # 3. Rage Detection
        rage_warning = ""
        if is_rage(question):
            rage_warning = "⚠️ කරුණාකර සංසුන්ව ප්‍රශ්නය අහන්න! "

        # 4. RAG Context
        rag_context = get_rag_context(question, user_grade)

        instruction = f"""
        ඔබ දක්ෂ {subject} ගුරුවරයෙකි. {user_grade} ශ්‍රේණියේ {user_name} ට පියවරෙන් පියවර සිංහලෙන් පැහැදිලි කරන්න.
        ගැටලුව විසඳීමට ප්‍රස්තාරයක් අවශ්‍ය නම්, matplotlib කෝඩ් එකක් ලෙස [GRAPH_START] plt... [GRAPH_END] අතර දෙන්න.
        plt.savefig() භාවිතා නොකරන්න.
        {f'පහත textbook reference භාවිතා කරන්න: {rag_context}' if rag_context else ''}
        """

        content_list = [f"User: {user_name}, Grade: {user_grade}, Subject: {subject}. Question: {question}"]

        if image_file:
            image_data = image_file.read()
            content_list.append({"mime_type": "image/jpeg", "data": image_data})

        # 5. Gemini Flash — First Answer
        response1 = gemini_flash.generate_content([instruction] + content_list)
        answer1 = response1.text

        # 6. Claude Haiku — Cross Check
        cross_check_prompt = f"""
        පහත ගණිත/විද්‍යා ප්‍රශ්නයට දෙන ලද පිළිතුර නිවැරදිද? 
        ප්‍රශ්නය: {question}
        පිළිතුර: {answer1}
        
        නිවැරදි නම් "✅ නිවැරදියි" කියා confirm කරන්න.
        වැරදි නම් නිවැරදි පිළිතුර සිංහලෙන් දෙන්න.
        """
        claude_response = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": cross_check_prompt}]
        )
        cross_check = claude_response.content[0].text

        # 7. Final Answer
        if "✅ නිවැරදියි" in cross_check:
            final_answer = answer1
            verified = True
        else:
            final_answer = cross_check
            verified = False

        # 8. Graph Generation
        graph_url = None
        graph_code_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', final_answer, re.DOTALL)
        if graph_code_match:
            graph_code = graph_code_match.group(1).strip()
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

        # 9. Save History & Stats
        save_history(user_name, user_grade, subject, question, final_answer)
        update_stats(subject)

        return jsonify({
            "status": "success",
            "answer": rage_warning + final_answer,
            "graph_url": graph_url,
            "verified": verified,
            "rag_used": bool(rag_context)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
# Admin Routes
# ============================================================
def check_admin(req):
    password = req.headers.get("X-Admin-Password", "")
    return password == ADMIN_PASSWORD

@app.route('/admin/stats', methods=['GET'])
def admin_stats():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    stats   = load_json(STATS_FILE, {"total": 0, "subjects": {}})
    history = load_json(HISTORY_FILE, [])
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
    data = request.json
    user_name = data.get("user_name")
    blacklist = load_json(BLACKLIST_FILE, {})
    if user_name in blacklist:
        del blacklist[user_name]
        save_json(BLACKLIST_FILE, blacklist)
    return jsonify({"status": "success", "message": f"{user_name} blacklist remove කරලා!"})

@app.route('/admin/upload_pdf', methods=['POST'])
def admin_upload_pdf():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return jsonify({"status": "error", "message": "PDF file එකක් නෑ!"})
    filename = pdf_file.filename
    pdf_file.save(os.path.join(PDF_FOLDER, filename))
    return jsonify({"status": "success", "message": f"{filename} upload කරලා!"})

@app.route('/admin/pdfs', methods=['GET'])
def admin_list_pdfs():
    if not check_admin(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    return jsonify({"status": "success", "pdfs": pdfs})

# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
