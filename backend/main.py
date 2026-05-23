import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend import config
from backend.database import init_db
from backend.services.email_service import init_scheduler

# Import Routers
from backend.routers.auth import router as auth_router
from backend.routers.tutor import router as tutor_router
from backend.routers.admin import router as admin_router

app = FastAPI(
    title="Sanz AI Tutor",
    description="Socratic Math & Science Multi-lingual Tutor Platform backend",
    version="3.0.0"
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Sub-Routers ──
app.include_router(auth_router)
app.include_router(tutor_router)
app.include_router(admin_router)

# ── Mounting Frontend Assets ──
# Serving static CSS & JS files under /css and /js
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    css_dir = os.path.join(frontend_dir, "css")
    js_dir = os.path.join(frontend_dir, "js")
    
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")

    # Serve student portal index at / or /student
    @app.get("/")
    @app.get("/student")
    async def serve_student_portal():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    # Serve parent dashboard at /dashboard or /parent
    @app.get("/dashboard")
    @app.get("/parent")
    async def serve_parent_dashboard():
        return FileResponse(os.path.join(frontend_dir, "dashboard.html"))

# ── Startup & Initialization ──
@app.on_event("startup")
def startup_event():
    # Initialize SQLite Database tables
    init_db()
    # Start APScheduler background weekly progress report dispatcher
    init_scheduler()
    print("Sanz AI Tutor Modular Backend successfully initialized and online!")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
