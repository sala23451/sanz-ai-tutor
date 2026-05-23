import os
import io
import re
import json
import secrets
import datetime
import asyncio
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import config
from backend.tutor_prompts import LANG_INSTRUCTIONS
from backend.database import load_json, save_json

# Import services
from backend.services.ai_service import (
    gemini_flash, gemini_check, cache_lookup, cache_store, get_rag_context
)
from backend.services.tracker import (
    detect_language, is_blacklisted, contains_bad_words, add_to_blacklist,
    is_rage, save_history, update_stats, track_user_tokens,
    update_progress, update_leaderboard, get_progress, estimate_tokens,
    track_api_call
)
from backend.services.draw_service import generate_graph_base64, execute_matplotlib_diagram
from backend.services.voice_service import process_voice_input, convert_text_to_speech, TTS_SUPPORT, STT_SUPPORT

router = APIRouter()

# ── Pydantic Request Models ──
class QuizStartRequest(BaseModel):
    user_name: str
    grade: str
    subject: str
    language: str = "si"

class QuizAnswerRequest(BaseModel):
    session_id: str
    user_name: str
    user_answer: str

class TTSRequest(BaseModel):
    text: str
    language: str = "si"

class ImageGenRequest(BaseModel):
    prompt: str
    subject: str = "General"
    language: str = "en"

# ── Quiz Endpoints ──

@router.post("/quiz/start")
async def start_quiz(body: QuizStartRequest):
    lang       = body.language if body.language in LANG_INSTRUCTIONS else "en"
    difficulty = "easy" if int(body.grade) <= 5 else "medium" if int(body.grade) <= 9 else "hard"
    
    prompt = f"""Generate exactly 5 multiple-choice quiz questions for:
Grade: {body.grade}, Subject: {body.subject}, Difficulty: {difficulty}, Language: {body.language}
Return ONLY valid JSON:
{{"questions":[{{"q":"Question","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"answer":"A","explanation":"Why A is correct"}}]}}"""
    try:
        response  = gemini_flash.generate_content(prompt)
        track_api_call("gemini-2.5-flash-lite", "quiz_generate")
        raw       = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
        data      = json.loads(raw)
        questions = data.get("questions", [])
        if not questions: 
            raise ValueError("No questions")
    except Exception as e:
        return {"status": "error", "message": f"Quiz generation failed: {e}"}
        
    session_id = secrets.token_hex(8)
    sessions   = load_json(config.QUIZ_FILE, {})
    sessions[session_id] = {
        "user_name": body.user_name, "grade": body.grade,
        "subject": body.subject, "language": lang,
        "questions": questions, "current": 0,
        "score": 0, "answers": [],
        "started": datetime.datetime.now().isoformat(), "completed": False
    }
    save_json(config.QUIZ_FILE, sessions)
    q = questions[0]
    return {
        "status": "success", "session_id": session_id,
        "total": len(questions), "current": 1,
        "question": q["q"], "options": q["options"], "progress_pct": 0
    }

@router.post("/quiz/answer")
async def answer_quiz(body: QuizAnswerRequest):
    sessions = load_json(config.QUIZ_FILE, {})
    if body.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Quiz not found")
    sess = sessions[body.session_id]
    if sess["completed"]:
        raise HTTPException(status_code=400, detail="Quiz already done")
        
    idx     = sess["current"]
    q       = sess["questions"][idx]
    correct = body.user_answer.strip().upper().startswith(q["answer"].upper())
    if correct: 
        sess["score"] += 1
        
    sess["answers"].append({
        "question": q["q"], "user_answer": body.user_answer,
        "correct": correct, "right_answer": q["answer"],
        "explanation": q.get("explanation", "")
    })
    sess["current"] += 1
    
    if sess["current"] >= len(sess["questions"]):
        sess["completed"] = True
        sessions[body.session_id] = sess
        save_json(config.QUIZ_FILE, sessions)
        
        xp_earned = sess["score"] * 15
        user      = body.user_name.lower()
        progress  = load_json(config.PROGRESS_FILE, {})
        if user not in progress:
            get_progress(user)
            progress = load_json(config.PROGRESS_FILE, {})
            
        p         = progress.get(user, {})
        p["xp"]   = p.get("xp", 0) + xp_earned
        p.setdefault("quiz_scores", []).append({
            "subject": sess["subject"], "score": sess["score"],
            "total": len(sess["questions"]), "xp": xp_earned,
            "date": datetime.date.today().isoformat()
        })
        progress[user] = p
        save_json(config.PROGRESS_FILE, progress)
        update_leaderboard(user, sess["grade"])
        
        return {
            "status": "completed", "correct": correct,
            "explanation": q.get("explanation", ""),
            "score": sess["score"], "total": len(sess["questions"]),
            "percent": round(sess["score"] / len(sess["questions"]) * 100),
            "xp_earned": xp_earned, "answers": sess["answers"]
        }
        
    sessions[body.session_id] = sess
    save_json(config.QUIZ_FILE, sessions)
    next_q = sess["questions"][sess["current"]]
    return {
        "status": "ongoing", "correct": correct,
        "explanation": q.get("explanation", ""),
        "current": sess["current"] + 1, "total": len(sess["questions"]),
        "question": next_q["q"], "options": next_q["options"],
        "score_so_far": sess["score"],
        "progress_pct": round(sess["current"] / len(sess["questions"]) * 100)
    }

# ── Voice & Audio Endpoints ──

@router.post("/voice-input")
async def voice_input(audio: UploadFile = File(...)):
    if not STT_SUPPORT:
        raise HTTPException(status_code=500, detail="STT not installed on server.")
    contents = await audio.read()
    res = process_voice_input(contents)
    if res["status"] == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@router.post("/tts")
async def text_to_speech(body: TTSRequest):
    if not TTS_SUPPORT:
        raise HTTPException(status_code=500, detail="TTS not installed on server.")
    try:
        audio_stream = convert_text_to_speech(body.text, body.language)
        return StreamingResponse(audio_stream, media_type="audio/mpeg",
                                 headers={"Content-Disposition": "inline; filename=response.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Image Diagram Generation Endpoint ──

@router.post("/generate-image")
async def generate_image(body: ImageGenRequest):
    description, graph_url = execute_matplotlib_diagram(body.prompt, body.subject)
    if graph_url:
        return {"status": "success", "description": description, "image_url": graph_url}
    else:
        return {"status": "error", "message": description}

# ── Main Socratic Solvers Endpoint ──

@router.post("/solve")
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
                "si": "⛔ නුසුදුසු වචන use කළა! ඔයා පැය 2ක් blacklist කරලා!",
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

        # 7. RAG lookup
        rag_context, rag_used = get_rag_context(question, grade)

        # 8. Conversation context builder
        conversation_context = ""
        if conv_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conv_history:
                role = msg.get("role", "")
                text = msg.get("text", "")[:300]
                if role == "user":  conversation_context += f"Student: {text}\n"
                elif role == "ai":  conversation_context += f"Tutor: {text}\n"
            conversation_context += "\nContinue naturally based on above context.\n"

        # 9. Prompt formatting
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

        # 10. Parallel generation and cross-check
        if is_math and not (image and image.filename):
            pre_cross_prompt = f"""Is this a valid math question that can be answered?
Question: {question}
Reply only YES or NO."""

            # Gather both tasks concurrently
            main_task   = asyncio.to_thread(gemini_flash.generate_content, [instruction] + content_list)
            pre_task    = asyncio.to_thread(gemini_check.generate_content, pre_cross_prompt)
            response1, _ = await asyncio.gather(main_task, pre_task)
            answer1 = response1.text
            track_api_call("gemini-2.5-flash-lite", "main_answer")

            # Perform post-verification check
            cross_check_prompt = f"""Is this answer correct?
Question: {question}
Answer: {answer1[:500]}
{lang_cfg['cross_check']}"""
            cross_response = await asyncio.to_thread(gemini_check.generate_content, cross_check_prompt)
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
                redirect_resp = await asyncio.to_thread(gemini_check.generate_content, redirect_prompt)
                final_answer  = redirect_resp.text
                track_api_call("gemini-2.5-flash-lite", "socratic_redirect")
                verified = False
        else:
            # Single Gemini execution step for images or non-math
            response1    = await asyncio.to_thread(gemini_flash.generate_content, [instruction] + content_list)
            answer1      = response1.text
            final_answer = answer1
            verified     = True
            track_api_call("gemini-2.5-flash-lite", "main_answer")

        # 11. Plot parser sandbox
        graph_url   = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', final_answer, re.DOTALL)
        if graph_match:
            graph_code   = graph_match.group(1).strip()
            final_answer = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', final_answer, flags=re.DOTALL)
            graph_url    = generate_graph_base64(graph_code, figsize=(6, 4), dpi=120)

        # 12. Save progression indexes
        save_history(name, grade, subject, question, final_answer)
        update_stats(subject)
        new_badge = update_progress(name.lower(), subject, verified, xp_earned=5 if verified else 2)
        update_leaderboard(name.lower(), grade)

        # Record metrics usage
        input_est  = estimate_tokens(question + (rag_context or "") + str(conv_history))
        output_est = estimate_tokens(final_answer)
        track_user_tokens(name.lower(), input_est, output_est)

        # Update cache
        if not conv_history and not (image and image.filename):
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
            # Standard single retry fallback
            import time
            time.sleep(5)
            try:
                fallback = gemini_check.generate_content(
                    f"Answer this briefly for a Grade {grade} student in language '{language}': {question}"
                )
                track_api_call("gemini-2.5-flash-lite", "main_answer")
                return {
                    "status": "success", "answer": fallback.text,
                    "graph_url": None, "verified": True, "rag_used": False,
                    "cache_hit": False, "detected_language": language,
                    "new_badge": None, "progress": get_progress(name.lower())
                }
            except Exception as e2:
                retry_msgs = {
                    "si": "⏳ AI server ටිකක් busy! තත්පර 10කින් නැවත try කරන්න. 😊",
                    "en": "⏳ AI server is busy! Please try again in 10 seconds. 😊",
                    "ta": "⏳ AI server பிஸியாக உள்ளது! 10 விநாடிகளில் மீண்டும் முயற்சிக்கவும். 😊"
                }
                return {"status": "rate_limit", "answer": retry_msgs.get(language, retry_msgs["en"]),
                        "graph_url": None, "verified": True, "rag_used": False,
                        "cache_hit": False, "detected_language": language}
        return {"status": "error", "message": error_msg}
