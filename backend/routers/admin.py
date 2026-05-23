import os
import re
import datetime
import tempfile
from fastapi import APIRouter, Header, HTTPException, File, UploadFile
from pydantic import BaseModel

from backend import config
from backend.database import load_json, save_json
from backend.services.tracker import (
    check_admin, get_api_usage, add_to_blacklist, get_progress
)
from backend.services.email_service import send_weekly_email, SCHEDULER_SUPPORT
from backend.services.voice_service import TTS_SUPPORT, STT_SUPPORT
from backend.services.ai_service import (
    pinecone_embed, pinecone_index, PINECONE_SUPPORT, PDF_SUPPORT
)

router = APIRouter()

# ── Cloudinary Setup ──
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    CLOUDINARY_SUPPORT = True
except ImportError:
    CLOUDINARY_SUPPORT = False

if CLOUDINARY_SUPPORT and config.CLOUDINARY_CLOUD_NAME:
    cloudinary.config(
        cloud_name=config.CLOUDINARY_CLOUD_NAME,
        api_key=config.CLOUDINARY_API_KEY,
        api_secret=config.CLOUDINARY_API_SECRET
    )

# ── fitz (PyMuPDF) Setup ──
try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Unified Secure Auth Helper ──
def check_auth(authorization: str = Header(default=""), x_admin_password: str = Header(default="")) -> bool:
    """Verifies access using secure Bearer session token or fallback legacy password header."""
    # 1. Bearer Token Check
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1].strip()
        sessions = load_json("admin_sessions.json", {})
        if token in sessions:
            try:
                expiry = datetime.datetime.fromisoformat(sessions[token]["expires_at"])
                if datetime.datetime.now() < expiry:
                    return True
            except Exception as e:
                print(f"Auth token validation error: {e}")
                pass
    # 2. Legacy Password Header Check
    if x_admin_password and check_admin(x_admin_password):
        return True
    return False

# ── Pydantic Request Models ──
class BlacklistRemoveRequest(BaseModel):
    user_name: str

# ── Legacy User Helper ──
def get_user_accounts():
    return load_json(config.USER_ACCOUNTS_FILE, {})

def save_user_accounts(data):
    save_json(config.USER_ACCOUNTS_FILE, data)

# ── Health Endpoint ──

@router.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "version": "3.0.0",
        "stt_support": STT_SUPPORT, 
        "tts_support": TTS_SUPPORT,
        "pdf_support": PDF_SUPPORT, 
        "cloudinary_support": CLOUDINARY_SUPPORT,
        "pinecone_support": PINECONE_SUPPORT, 
        "scheduler_support": SCHEDULER_SUPPORT,
        "github_storage": False,
        "github_repo": "not configured"
    }

# ── Stats & History Endpoints ──

@router.get("/admin/stats")
async def admin_stats(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    stats     = load_json(config.STATS_FILE, {"total": 0, "subjects": {}})
    history   = load_json(config.HISTORY_FILE, [])
    blacklist = load_json(config.BLACKLIST_FILE, {})
    parents   = load_json(config.PARENTS_FILE, {})
    children  = load_json(config.CHILDREN_FILE, {})
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

@router.get("/admin/history")
async def admin_history(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "success", "history": load_json(config.HISTORY_FILE, [])[-100:]}

@router.get("/admin/api-usage")
async def admin_api_usage(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "success", "usage": get_api_usage()}

@router.get("/admin/token-usage")
async def admin_token_usage(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tokens  = load_json(config.USER_TOKENS_FILE, {})
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

@router.get("/admin/token-usage/{user_name}")
async def admin_user_tokens(user_name: str, authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tokens = load_json(config.USER_TOKENS_FILE, {})
    if user_name not in tokens:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "user": user_name, "data": tokens[user_name]}

# ── Parent & Student Accounts Management Endpoints ──

@router.get("/admin/parents")
async def admin_parents(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    parents     = load_json(config.PARENTS_FILE, {})
    children_db = load_json(config.CHILDREN_FILE, {})
    accounts    = get_user_accounts()
    result      = []

    # Parent-system accounts
    for email, p in parents.items():
        children = [
            {"id": cid, "name": children_db[cid]["name"], "grade": children_db[cid]["grade"]}
            for cid in p.get("children", []) if cid in children_db
        ]
        result.append({
            "email": email, "name": p["name"],
            "children": children, "plan": p.get("plan", "free"),
            "created": p.get("created", ""),
            "type": "parent"
        })

    # Self-registered student accounts
    for uname, acc in accounts.items():
        email  = acc.get("email", "")
        phone  = acc.get("phone", "")
        contact = email or phone
        if not contact:
            continue
        result.append({
            "email":    contact,
            "name":     acc.get("full_name", uname),
            "children": [{"id": uname, "name": acc.get("full_name", uname), "grade": acc.get("grade", "?")}],
            "plan":     "student_account",
            "created":  acc.get("created", ""),
            "type":     "student_account"
        })

    return {"status": "success", "parents": result, "total": len(result)}

# ── User Accounts Disabling/Enabling Endpoints ──

@router.get("/admin/user-accounts")
async def admin_user_accounts(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    accounts = get_user_accounts()
    tokens   = load_json(config.USER_TOKENS_FILE, {})
    today    = datetime.date.today().isoformat()
    result   = []
    
    for uname, acc in accounts.items():
        safe = acc.copy()
        safe.pop("password_hash", None)
        # Add token usage
        tok_data   = tokens.get(uname, {})
        today_data = tok_data.get("today", {}).get(today, {})
        safe["tokens_today"] = today_data.get("input", 0) + today_data.get("output", 0)
        safe["total_tokens"] = tok_data.get("total_input", 0) + tok_data.get("total_output", 0)
        
        # Add progress summary
        prog = load_json(config.PROGRESS_FILE, {}).get(uname, {})
        safe["xp"]             = prog.get("xp", 0)
        safe["streak"]         = prog.get("streak", 0)
        safe["total_questions"] = prog.get("total_questions", 0)
        safe["badges_count"]   = len(prog.get("badges", []))
        result.append(safe)
        
    result.sort(key=lambda x: x.get("created", ""), reverse=True)
    return {"status": "success", "accounts": result, "total": len(result)}

@router.post("/admin/user-accounts/disable")
async def admin_disable_user(body: BlacklistRemoveRequest, authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    accounts = get_user_accounts()
    uname    = body.user_name.strip().lower()
    if uname not in accounts:
        raise HTTPException(status_code=404, detail="User not found!")
    accounts[uname]["active"] = False
    save_user_accounts(accounts)
    return {"status": "success", "message": f"{uname} disabled!"}

@router.post("/admin/user-accounts/enable")
async def admin_enable_user(body: BlacklistRemoveRequest, authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    accounts = get_user_accounts()
    uname    = body.user_name.strip().lower()
    if uname not in accounts:
        raise HTTPException(status_code=404, detail="User not found!")
    accounts[uname]["active"] = True
    save_user_accounts(accounts)
    return {"status": "success", "message": f"{uname} enabled!"}

# ── Blacklist Controls Endpoints ──

@router.post("/admin/blacklist/remove")
async def admin_remove_blacklist(body: BlacklistRemoveRequest, authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    blacklist = load_json(config.BLACKLIST_FILE, {})
    if body.user_name in blacklist:
        del blacklist[body.user_name]
        save_json(config.BLACKLIST_FILE, blacklist)
    return {"status": "success", "message": f"{body.user_name} removed!"}

@router.post("/admin/blacklist/add")
async def admin_add_blacklist(body: BlacklistRemoveRequest, authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    add_to_blacklist(body.user_name)
    return {"status": "success", "message": f"{body.user_name} banned for 2 hours!"}

# ── Email Scheduler Commands ──

@router.post("/admin/email/send-now")
async def admin_send_email_now(
    parent_email: str, child_name: str,
    authorization: str = Header(default=""),
    x_admin_password: str = Header(default="")
):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    success = send_weekly_email(parent_email, child_name)
    return {
        "status":  "success" if success else "failed",
        "message": f"Email {'sent' if success else 'failed'} to {parent_email}"
    }

@router.get("/admin/email/log")
async def admin_email_log(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "success", "log": load_json(config.EMAIL_LOG_FILE, [])[-50:]}

# ── PDF Textbook Management Endpoints ──

@router.post("/admin/upload_pdf")
async def admin_upload_pdf(pdf: UploadFile = File(...), authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    contents = await pdf.read()
    filename = pdf.filename
    with open(os.path.join(config.PDF_FOLDER, filename), "wb") as f:
        f.write(contents)
        
    cloud_url = ""
    if CLOUDINARY_SUPPORT and config.CLOUDINARY_CLOUD_NAME:
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
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
                
            doc     = fitz.open(tmp_path)
            vectors = []
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if not text or len(text) < 50: 
                    continue
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

@router.delete("/admin/pdfs/{filename}")
async def admin_delete_pdf(filename: str, authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    local_path = os.path.join(config.PDF_FOLDER, filename)
    if os.path.exists(local_path): 
        os.remove(local_path)
    if CLOUDINARY_SUPPORT and config.CLOUDINARY_CLOUD_NAME:
        try: 
            cloudinary.uploader.destroy(f"sanz_textbooks/{filename.replace('.pdf','')}", resource_type="raw")
        except: 
            pass
    return {"status": "success", "message": f"{filename} deleted!"}

@router.get("/admin/pdfs")
async def admin_list_pdfs(authorization: str = Header(default=""), x_admin_password: str = Header(default="")):
    if not check_auth(authorization, x_admin_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    pdfs = [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]
    return {"status": "success", "pdfs": pdfs}
