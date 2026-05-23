import hashlib
import secrets
import uuid
import datetime
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from backend import config
from backend.database import load_json, save_json
from backend.services.tracker import get_progress, calculate_age, check_admin

router = APIRouter()

# ── Pydantic Request Models ──
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

class UserRegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str
    birthday: str      # "2010-05-14"
    grade: str
    email: str = ""
    phone: str = ""
    language: str = "si"

class UserLoginRequest(BaseModel):
    username: str
    password: str

class UserUpdateRequest(BaseModel):
    username: str
    full_name: str = ""
    grade: str = ""
    email: str = ""
    phone: str = ""

# ── Legacy User Helpers ──
def get_users():
    default = {
        "sanduni": {
            "password_hash": hashlib.sha256(config.ADMIN_PASSWORD.encode()).hexdigest(),
            "role": "admin"
        }
    }
    users = load_json(config.USERS_FILE, None)
    if users is None:
        save_json(config.USERS_FILE, default)
        return default
    return users

def get_user_accounts():
    return load_json(config.USER_ACCOUNTS_FILE, {})

def save_user_accounts(data):
    save_json(config.USER_ACCOUNTS_FILE, data)

# ── Endpoints ──

@router.post("/auth/login")
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

@router.post("/auth/register")
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
    save_json(config.USERS_FILE, users)
    return {"status": "success", "message": f"{name} registered!"}

# ── PARENT PORTAL ENDPOINTS ──

@router.post("/parent/register")
async def parent_register(body: ParentRegisterRequest):
    parents = load_json(config.PARENTS_FILE, {})
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
    save_json(config.PARENTS_FILE, parents)
    return {"status": "success", "message": f"Welcome {body.name}! Account created.", "email": email}

@router.post("/parent/login")
async def parent_login(body: ParentLoginRequest):
    parents = load_json(config.PARENTS_FILE, {})
    email   = body.email.strip().lower()
    if email not in parents:
        raise HTTPException(status_code=401, detail="Email not found!")
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if parents[email]["password_hash"] != pw_hash:
        raise HTTPException(status_code=401, detail="Wrong password!")
        
    children_db = load_json(config.CHILDREN_FILE, {})
    children    = []
    for cid in parents[email]["children"]:
        if cid in children_db:
            child        = children_db[cid].copy()
            tokens       = load_json(config.USER_TOKENS_FILE, {})
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

@router.post("/parent/add-child")
async def add_child(body: AddChildRequest):
    parents     = load_json(config.PARENTS_FILE, {})
    children_db = load_json(config.CHILDREN_FILE, {})
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
    save_json(config.CHILDREN_FILE, children_db)
    save_json(config.PARENTS_FILE, parents)
    return {"status": "success", "message": f"{body.name}'s account created!", "child_id": child_id, "age": age}

@router.get("/parent/children")
async def get_parent_children(parent_email: str):
    parents     = load_json(config.PARENTS_FILE, {})
    children_db = load_json(config.CHILDREN_FILE, {})
    email       = parent_email.strip().lower()
    if email not in parents:
        raise HTTPException(status_code=404, detail="Parent not found!")
    children = []
    for cid in parents[email]["children"]:
        if cid in children_db:
            child        = children_db[cid].copy()
            tokens       = load_json(config.USER_TOKENS_FILE, {})
            today        = datetime.date.today().isoformat()
            child_tokens = tokens.get(child["name"].lower(), {})
            today_data   = child_tokens.get("today", {}).get(today, {})
            child["tokens_today"] = today_data.get("input", 0) + today_data.get("output", 0)
            child["tokens_total"] = child_tokens.get("total_input", 0) + child_tokens.get("total_output", 0)
            child["progress"]     = get_progress(child["name"].lower())
            children.append({"id": cid, **child})
    return {"status": "success", "children": children}

@router.put("/parent/child/update")
async def update_child(body: UpdateChildRequest):
    children_db = load_json(config.CHILDREN_FILE, {})
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
    save_json(config.CHILDREN_FILE, children_db)
    return {"status": "success", "message": "Profile updated!"}

@router.delete("/parent/child")
async def delete_child(body: DeleteChildRequest):
    children_db = load_json(config.CHILDREN_FILE, {})
    parents     = load_json(config.PARENTS_FILE, {})
    email       = body.parent_email.strip().lower()
    if body.child_id not in children_db:
        raise HTTPException(status_code=404, detail="Child not found!")
    if children_db[body.child_id]["parent_email"] != email:
        raise HTTPException(status_code=403, detail="Not your child's account!")
    del children_db[body.child_id]
    if body.child_id in parents[email]["children"]:
        parents[email]["children"].remove(body.child_id)
    save_json(config.CHILDREN_FILE, children_db)
    save_json(config.PARENTS_FILE, parents)
    return {"status": "success", "message": "Child account removed!"}

# ── STUDENT SELF-REGISTRATION SYSTEM ──

@router.post("/user/register")
async def user_register(body: UserRegisterRequest):
    accounts = get_user_accounts()
    uname    = body.username.strip().lower().replace(" ", "_")

    if not uname or len(uname) < 3:
        raise HTTPException(status_code=400, detail="Username must be 3+ characters!")
    if not body.password or len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be 6+ characters!")
    if uname in accounts:
        raise HTTPException(status_code=400, detail="Username already taken!")
    if not body.email and not body.phone:
        raise HTTPException(status_code=400, detail="Email or phone required!")

    age = calculate_age(body.birthday)

    accounts[uname] = {
        "username":       uname,
        "full_name":      body.full_name.strip(),
        "password_hash":  hashlib.sha256(body.password.encode()).hexdigest(),
        "birthday":       body.birthday,
        "age":            age,
        "grade":          body.grade,
        "email":          body.email.strip().lower(),
        "phone":          body.phone.strip(),
        "language":       body.language,
        "created":        datetime.datetime.now().isoformat(),
        "last_login":     None,
        "active":         True,
        "role":           "student"
    }
    save_user_accounts(accounts)

    return {
        "status":    "success",
        "message":   f"Account created for {body.full_name}!",
        "username":  uname,
        "password":  body.password   # show once to user
    }

@router.post("/user/login")
async def user_login(body: UserLoginRequest):
    accounts = get_user_accounts()
    uname    = body.username.strip().lower().replace(" ", "_")

    if uname not in accounts:
        raise HTTPException(status_code=401, detail="Username not found!")

    acc     = accounts[uname]
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if acc["password_hash"] != pw_hash:
        raise HTTPException(status_code=401, detail="Wrong password!")
    if not acc.get("active", True):
        raise HTTPException(status_code=403, detail="Account disabled!")

    # Update last login
    accounts[uname]["last_login"] = datetime.datetime.now().isoformat()
    save_user_accounts(accounts)

    token = secrets.token_hex(32)
    return {
        "status":    "success",
        "token":     token,
        "username":  uname,
        "full_name": acc["full_name"],
        "grade":     acc["grade"],
        "email":     acc["email"],
        "phone":     acc["phone"],
        "language":  acc["language"],
        "age":       acc["age"],
        "birthday":  acc["birthday"],
        "role":      acc.get("role", "student")
    }

@router.get("/user/profile/{username}")
async def user_profile(username: str):
    accounts = get_user_accounts()
    uname    = username.strip().lower()
    if uname not in accounts:
        raise HTTPException(status_code=404, detail="User not found!")
    acc = accounts[uname].copy()
    acc.pop("password_hash", None)
    # Add progress
    acc["progress"] = get_progress(uname)
    return {"status": "success", "profile": acc}

@router.put("/user/profile")
async def update_user_profile(body: UserUpdateRequest):
    accounts = get_user_accounts()
    uname    = body.username.strip().lower()
    if uname not in accounts:
        raise HTTPException(status_code=404, detail="User not found!")
    acc = accounts[uname]
    if body.full_name: acc["full_name"] = body.full_name
    if body.grade:     acc["grade"]     = body.grade
    if body.email:     acc["email"]     = body.email.strip().lower()
    if body.phone:     acc["phone"]     = body.phone.strip()
    accounts[uname] = acc
    save_user_accounts(accounts)
    return {"status": "success", "message": "Profile updated!"}

# ── Admin Secure Session Auth ──

class AdminLoginRequest(BaseModel):
    username: str
    password: str

@router.post("/admin/auth/login")
async def admin_auth_login(body: AdminLoginRequest):
    users = get_users()
    uname = body.username.strip().lower()
    if uname != "sanduni":
        raise HTTPException(status_code=401, detail="Unauthorized username")
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if users.get(uname, {}).get("password_hash") != pw_hash:
        raise HTTPException(status_code=401, detail="Wrong admin password")
        
    token = secrets.token_hex(32)
    sessions = load_json("admin_sessions.json", {})
    expiry = (datetime.datetime.now() + datetime.timedelta(hours=12)).isoformat()
    sessions[token] = {
        "username": uname,
        "expires_at": expiry
    }
    save_json("admin_sessions.json", sessions)
    return {"status": "success", "token": token, "expires_at": expiry}
