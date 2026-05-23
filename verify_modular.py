import os
import sys

# Configure stdout to handle UTF-8 symbols cleanly on Windows consoles
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8')

# Add current workspace to path to resolve packages properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

print("==================================================")
print("🚀 SANZ AI TUTOR MODULAR DIAGNOSTICS & VERIFIER 🚀")
print("==================================================")

try:
    print("\n[1/4] Checking Backend Imports...")
    
    from backend import config
    print("  ✅ config.py loaded successfully.")
    
    from backend.database import init_db, load_json, save_json
    print("  ✅ database.py loaded successfully.")
    
    from backend.tutor_prompts import SOCRATIC_INSTRUCTIONS, LANG_INSTRUCTIONS
    print("  ✅ tutor_prompts.py loaded successfully.")
    
    from backend.services import ai_service, voice_service, draw_service, tracker, email_service
    print("  ✅ All backend services successfully imported.")
    
    from backend.routers import auth, tutor, admin
    print("  ✅ All APIRouters successfully imported.")
    
    from backend.main import app
    print("  ✅ main.py FastAPI app context initialized successfully.")

except Exception as e:
    print(f"  ❌ Import error encountered: {e}")
    sys.exit(1)

try:
    print("\n[2/4] Testing Database Initialization...")
    init_db()
    print("  ✅ SQLite table created/verified successfully.")
    
    # Test DB load/save fallbacks
    test_key = "test_run_metric"
    test_data = {"status": "ok", "timestamp": "now"}
    save_json(test_key, test_data)
    loaded = load_json(test_key, {})
    if loaded.get("status") == "ok":
        print("  ✅ SQLite Key-Value storage wrapper functioning perfectly.")
    else:
        raise ValueError("Loaded data does not match saved data!")

except Exception as e:
    print(f"  ❌ Database test failed: {e}")
    sys.exit(1)

try:
    print("\n[3/4] Validating Moderation & Tracker Logic...")
    from backend.services.tracker import (
        detect_language, contains_bad_words, is_rage, estimate_tokens
    )
    
    # Language detection checks
    assert detect_language("හොඳ ආරම්භයක්") == "si", "Sinhala detection failed"
    assert detect_language("வணக்கம்") == "ta", "Tamil detection failed"
    assert detect_language("Hello tutor") == "en", "English detection failed"
    print("  ✅ Language detection logic matches exactly.")
    
    # Bad words moderation checks
    assert contains_bad_words("hello fool cock") == True, "English bad words filter failed"
    assert contains_bad_words("අනේ මෝඩ හරකෙක්") == True, "Sinhala bad words filter failed"
    assert contains_bad_words("good student math") == False, "Clean sentence marked bad"
    print("  ✅ Bad words filter logic matches exactly.")
    
    # Rage warnings checks
    assert is_rage("stupid useless worst math") == True, "Rage count filter failed"
    assert is_rage("I love math tutor") == False, "Clean sentence marked rage"
    print("  ✅ Rage metrics logic matches exactly.")
    
    # Token estimation checks
    assert estimate_tokens("hello world") == 2, "Token estimator discrepancy"
    print("  ✅ Token estimation logic matches exactly.")

except AssertionError as e:
    print(f"  ❌ Moderation assertion failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Moderation validation failed: {e}")
    sys.exit(1)

print("\n[4/4] Verifying Frontend Asset Layout...")
frontend_files = [
    "frontend/index.html",
    "frontend/dashboard.html",
    "frontend/css/index.css",
    "frontend/css/dashboard.css",
    "frontend/js/index.js",
    "frontend/js/dashboard.js"
]

missing_assets = 0
for f in frontend_files:
    if os.path.exists(f):
        print(f"  ✅ Asset found: {f}")
    else:
        print(f"  ❌ Missing asset: {f}")
        missing_assets += 1

if missing_assets > 0:
    print(f"  ❌ Frontend assets verification failed: {missing_assets} files missing.")
    sys.exit(1)
else:
    print("  ✅ All split frontend files present.")

print("\n==================================================")
print("🎉 ALL DIAGNOSTIC CHECKS COMPLETED SUCCESSFULLY! 🎉")
print("🎉 Codebase is modularized and ready to deploy!  🎉")
print("==================================================")
