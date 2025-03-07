from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import scheduler
from app.api.routes import auth
from app.api.routes import chat_routes
from app.api.routes import schedule
from app.core.config import get_settings
import firebase_admin

firebase_admin.initialize_app()

settings = get_settings()

app = FastAPI(title=settings.PROJECT_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scheduler.router, prefix="/api", tags=["scheduler"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(schedule.router, prefix="/api", tags=["schedule"])
app.include_router(chat_routes.router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Scheduler Assistant API"} 