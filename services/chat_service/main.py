from fastapi import FastAPI
from services.chat_service.rag_logic import hybrid_search
from core.db import AsyncSessionLocal # Предположим, тут лежит сессия

app = FastAPI(title="HSE Auto-Mentor API")

@app.get("/ping")
async def health_check():
    return {"status": "ok", "message": "Chat Service is live"}

@app.post("/chat")
async def chat(user_id: str, message: str):
    # Здесь мы будем вызывать логику из rag_logic.py
    return {"answer": f"Привет, я получил твой вопрос: {message}. Мозги RAG в процессе настройки!"}
