from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import routes  
from app.database import create_db_and_tables 

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    print("Startup: Database tables created.")
    yield
    print("Shutdown: App is closing.")

app = FastAPI(
    title="GlowFlow AI API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "GlowFlow Backend is Running 🚀"}