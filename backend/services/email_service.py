import os
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from backend import config
from backend.database import load_json, save_json
from backend.services.tracker import get_progress, track_api_call
from backend.services.ai_service import gemini_check

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_SUPPORT = True
except ImportError:
    SCHEDULER_SUPPORT = False

def get_child_week_stats(child_name: str) -> dict:
    progress   = load_json(config.PROGRESS_FILE, {})
    tokens_db  = load_json(config.USER_TOKENS_FILE, {})
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
    except Exception as e:
        print(f"Weekly recommendation generation failed: {e}")
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
        msg["From"]    = config.EMAIL_USER
        msg["To"]      = parent_email
        msg.attach(MIMEText(html_content, "html"))
        
        with smtplib.SMTP(config.EMAIL_HOST, config.EMAIL_PORT) as server:
            server.starttls()
            server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
            server.sendmail(config.EMAIL_USER, parent_email, msg.as_string())
            
        log = load_json(config.EMAIL_LOG_FILE, [])
        log.append({"parent": parent_email, "child": child_name,
                    "sent_at": datetime.datetime.now().isoformat(), "status": "success"})
        save_json(config.EMAIL_LOG_FILE, log)
        return True
    except Exception as e:
        log = load_json(config.EMAIL_LOG_FILE, [])
        log.append({"parent": parent_email, "child": child_name,
                    "sent_at": datetime.datetime.now().isoformat(), "status": "failed", "error": str(e)})
        save_json(config.EMAIL_LOG_FILE, log)
        print(f"Error sending email to {parent_email}: {e}")
        return False

def send_all_weekly_reports():
    parents     = load_json(config.PARENTS_FILE, {})
    children_db = load_json(config.CHILDREN_FILE, {})
    for email, parent in parents.items():
        if not parent.get("email_reports", True): 
            continue
        for cid in parent.get("children", []):
            if cid in children_db:
                send_weekly_email(email, children_db[cid]["name"])
                print(f"Weekly email sent: {email} -> {children_db[cid]['name']}")

scheduler = None
def init_scheduler():
    global scheduler
    if SCHEDULER_SUPPORT:
        try:
            scheduler = BackgroundScheduler()
            # Run every Sunday at 8:00 PM
            scheduler.add_job(send_all_weekly_reports, "cron", day_of_week="sun", hour=20, minute=0)
            scheduler.start()
            print("Background APScheduler successfully loaded and started.")
        except Exception as e:
            print(f"Error starting BackgroundScheduler: {e}")
