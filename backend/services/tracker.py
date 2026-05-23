import os
import re
import datetime
from backend import config
from backend.database import load_json, save_json

# ── Base Admin Helper ──
def check_admin(x_admin_password: str = "") -> bool:
    return x_admin_password == config.ADMIN_PASSWORD

# ── General Calculations ──
def calculate_age(birthday: str) -> int:
    try:
        dob   = datetime.datetime.strptime(birthday, "%Y-%m-%d")
        today = datetime.datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return 0

def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)

def detect_language(text: str) -> str:
    sinhala_count = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    tamil_count   = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if sinhala_count > 2:   return "si"
    elif tamil_count > 2:   return "ta"
    return "en"

# ── Moderation & Bad Words ──
def contains_bad_words(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in config.BAD_WORDS)

def is_rage(text: str) -> bool:
    text_lower = text.lower()
    return sum(1 for w in config.RAGE_WORDS if w in text_lower) >= 2

def is_blacklisted(user_name: str) -> tuple:
    blacklist = load_json(config.BLACKLIST_FILE, {})
    if user_name in blacklist:
        try:
            ban_time = datetime.datetime.fromisoformat(blacklist[user_name])
            if datetime.datetime.now() < ban_time:
                remaining = (ban_time - datetime.datetime.now()).seconds // 60
                return True, remaining
            else:
                del blacklist[user_name]
                save_json(config.BLACKLIST_FILE, blacklist)
        except Exception as e:
            print(f"Error parsing blacklist for {user_name}: {e}")
    return False, 0

def add_to_blacklist(user_name: str):
    blacklist             = load_json(config.BLACKLIST_FILE, {})
    ban_until             = datetime.datetime.now() + datetime.timedelta(hours=2)
    blacklist[user_name]  = ban_until.isoformat()
    save_json(config.BLACKLIST_FILE, blacklist)

# ── Token Tracking ──
def track_user_tokens(user_name: str, input_tokens: int, output_tokens: int):
    tokens = load_json(config.USER_TOKENS_FILE, {})
    today  = datetime.date.today().isoformat()
    week   = datetime.date.today().strftime("%Y-W%W")
    month  = datetime.date.today().strftime("%Y-%m")

    name_key = user_name.lower()
    if name_key not in tokens:
        tokens[name_key] = {
            "total_input": 0, "total_output": 0, "total_calls": 0,
            "today": {}, "weekly": {}, "monthly": {}
        }

    u = tokens[name_key]
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

    tokens[name_key] = u
    save_json(config.USER_TOKENS_FILE, tokens)

def track_api_call(model: str, call_type: str = "solve"):
    usage = load_json(config.API_USAGE_FILE, {})
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
    save_json(config.API_USAGE_FILE, usage)

def get_api_usage() -> dict:
    usage      = load_json(config.API_USAGE_FILE, {})
    today      = datetime.date.today().isoformat()
    today_data = usage.get(today, {"total": 0, "models": {}, "by_type": {}})
    remaining  = {}
    for model, limit in config.API_LIMITS.items():
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

# ── Student Progress & Badges ──
def get_progress(user_name: str) -> dict:
    progress = load_json(config.PROGRESS_FILE, {})
    name_key = user_name.lower()
    if name_key not in progress:
        progress[name_key] = {
            "total_questions": 0, "correct_answers": 0,
            "subjects": {}, "daily_counts": {},
            "streak": 0, "last_active_date": None,
            "xp": 0, "badges": [], "quiz_scores": []
        }
        save_json(config.PROGRESS_FILE, progress)
    return progress[name_key]

def update_progress(user_name: str, subject: str, correct: bool, xp_earned: int = 5) -> str:
    progress = load_json(config.PROGRESS_FILE, {})
    name_key = user_name.lower()
    if name_key not in progress:
        get_progress(name_key)
        progress = load_json(config.PROGRESS_FILE, {})
    
    p     = progress[name_key]
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
    progress[name_key]  = p
    save_json(config.PROGRESS_FILE, progress)
    return new_badge

def update_leaderboard(user_name: str, grade: str):
    lb       = load_json(config.LEADERBOARD_FILE, {})
    progress = load_json(config.PROGRESS_FILE, {})
    name_key = user_name.lower()
    if name_key in progress:
        p             = progress[name_key]
        lb[name_key] = {
            "xp": p.get("xp", 0), "streak": p.get("streak", 0),
            "total_q": p.get("total_questions", 0),
            "badges": len(p.get("badges", [])), "grade": grade,
            "updated": datetime.datetime.now().isoformat()
        }
        save_json(config.LEADERBOARD_FILE, lb)

def update_stats(subject: str):
    stats = load_json(config.STATS_FILE, {"total": 0, "subjects": {}})
    stats["total"] += 1
    stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
    save_json(config.STATS_FILE, stats)

def save_history(user_name: str, grade: str, subject: str, question: str, answer: str):
    history = load_json(config.HISTORY_FILE, [])
    history.append({
        "user": user_name, "grade": grade, "subject": subject,
        "question": question, "answer": answer[:500],
        "time": datetime.datetime.now().isoformat()
    })
    history = history[-500:]
    save_json(config.HISTORY_FILE, history)
