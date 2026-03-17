"""
Retrieval Service
=================
Pure retrieval microservice that:
1. Receives a question and user profile.
2. Vektorizes the question.
3. Performs a filtered vector search in PostgreSQL (library schema).
4. Returns a list of relevant content chunks.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Internal modules
from retrieval import vector_search

# Database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RetrievalService")

# Config
DATABASE_URL       = os.environ["DATABASE_URL"]
EMBEDDER_MODEL_NAME = os.environ.get("EMBEDDER_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")

# Database Engine
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Embedder initialization
try:
    from sentence_transformers import SentenceTransformer
    logger.info(f"⏳ Loading embedder: {EMBEDDER_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME)
    logger.info("✅ Embedder loaded")
except Exception as e:
    logger.error(f"❌ Failed to load embedder: {e}")
    embedder = None

app = FastAPI(title="HSE Retrieval Service", version="2.0.0")


class UserProfile(BaseModel):
    university_id: Optional[str] = None
    campus_id: Optional[str] = None
    faculty_id: Optional[str] = None
    program_id: Optional[str] = None
    year: Optional[int] = None
    role: Optional[str] = "student"
    group_name: Optional[str] = None


class RetrievalRequest(BaseModel):
    user_id: str
    message: str = Field(..., alias="question")
    user_profile: UserProfile
    top_k: int = 5

    class Config:
        populate_by_name = True


@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "service": "Retrieval Service", 
        "model": EMBEDDER_MODEL_NAME
    }


@app.post("/retrieve")
async def retrieve(req: RetrievalRequest):
    """
    Main endpoint for RAG retrieval.
    Returns scoped relevant chunks based on user profile.
    """
    if embedder is None:
        raise HTTPException(500, "Embedder model not available.")

    # 1. Vectorize the question
    try:
        query_vec = embedder.encode(req.message, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise HTTPException(500, f"Error generating embedding: {e}")

    # 2. Perform scoped retrieval
    async with async_session() as session:
        chunks = await vector_search(
            session=session,
            query_embedding=query_vec,
            user_profile=req.user_profile.model_dump(),
            top_k=req.top_k
        )
    
    logger.info(f"🔍 Retrieved {len(chunks)} chunks for user {req.user_id}")

    # 3. Return results
    return {
        "question": req.message,
        "user_id": req.user_id,
        "chunks": chunks
    }
