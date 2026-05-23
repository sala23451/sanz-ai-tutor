import os

# ── Environment Variables ──
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "momsanzdad2001#")

# ── Email Settings ──
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASS", "")

# ── Cloudinary Settings ──
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

# ── Pinecone Settings ──
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# ── Directory Layout mappings (routed inside data/) ──
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_FILE = os.path.join(DATA_DIR, "sanz_tutor.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
BLACKLIST_FILE = os.path.join(DATA_DIR, "blacklist.json")
PDF_FOLDER = os.path.join(DATA_DIR, "textbooks")
STATS_FILE = os.path.join(DATA_DIR, "stats.json")
CACHE_FILE = os.path.join(DATA_DIR, "semantic_cache.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
QUIZ_FILE = os.path.join(DATA_DIR, "quiz_sessions.json")
LEADERBOARD_FILE = os.path.join(DATA_DIR, "leaderboard.json")
API_USAGE_FILE = os.path.join(DATA_DIR, "api_usage.json")
PARENTS_FILE = os.path.join(DATA_DIR, "parents.json")
CHILDREN_FILE = os.path.join(DATA_DIR, "children.json")
USER_TOKENS_FILE = os.path.join(DATA_DIR, "user_tokens.json")
EMAIL_LOG_FILE = os.path.join(DATA_DIR, "email_log.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
USER_ACCOUNTS_FILE = os.path.join(DATA_DIR, "user_accounts.json")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# ── Cache Configs ──
CACHE_SIMILARITY_THRESHOLD = 0.75
CACHE_MAX_SIZE = 200
CACHE_TTL_HOURS = 48

# ── API Quota Limits ──
API_LIMITS = {
    "gemini-2.5-flash-lite": 1000,
    "gemini-2.5-flash": 50,
}

# ── Moderation Lists ──
BAD_WORDS = [
    "fuck","shit","bitch","ass","bastard","dick","cock","pussy","cunt","whore","slut",
    "nigga","faggot","moron","dumbass","retard","loser",
    "හුත්ත","පුකේ","මෝඩ","හරක","කුකු","කපන්","බේපන්","ගනින්","හූකර","හාමු",
    "පට්ට හරක","කෙල්ල","කොල්ල","ගොන්","වාහල","හුකන්","හූකන","පිස්සු","උන්","බල්ලො",
    "හූලං","හූකල","ගෑනු","ගෑනි",
    "hutta","puke","moda","harak","kuku","kapan","bepan","ganin","hookar","pissu",
    "marana","ballo","hulan","wahala","goni","kotiya"
]

RAGE_WORDS = [
    "stupid","idiot","useless","hate","terrible","worst",
    "මෝඩ","නිකම්","වැඩක් නෑ","අකාරයි","හොඳ නෑ"
]
