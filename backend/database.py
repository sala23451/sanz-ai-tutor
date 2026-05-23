import os
import json
from sqlalchemy import create_engine, text as sql_text
from backend import config

# ── SQLAlchemy Database Engine ──
db_engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})

def init_db():
    """Create the key-value store table on startup if it doesn't exist."""
    with db_engine.connect() as conn:
        conn.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

def load_json(path: str, default):
    """Load JSON data from SQLite database. Falls back to local file if not in DB."""
    key = os.path.basename(path)
    try:
        with db_engine.connect() as conn:
            result = conn.execute(
                sql_text("SELECT value FROM kv_store WHERE key = :key"),
                {"key": key}
            )
            row = result.fetchone()
            if row:
                return json.loads(row[0])
    except Exception as e:
        print(f"DB load error ({key}): {e}")
    
    # Fallback: try reading from local JSON file (useful during migration)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path: str, data):
    """Save JSON data to SQLite database."""
    key   = os.path.basename(path)
    value = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        with db_engine.connect() as conn:
            conn.execute(sql_text("""
                INSERT INTO kv_store (key, value, updated_at)
                VALUES (:key, :value, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE
                  SET value      = excluded.value,
                      updated_at = excluded.updated_at
            """), {"key": key, "value": value})
            conn.commit()
    except Exception as e:
        print(f"DB save error ({key}): {e}")
