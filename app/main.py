from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.routes import router
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, create_engine

load_dotenv()

# Database Setup 
engine = create_engine(os.environ.get("DATABASE_URL"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup
    SQLModel.metadata.create_all(engine)
    yield

app = FastAPI(
    title="GlowFlow AI API",
    description="Backend for BeautyVibe/GlowFlow App",
    version="1.0.0",
    lifespan=lifespan
)

# Include Routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "GlowFlow Backend is Running"}