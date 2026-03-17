# services/chat_service/main.py
"""
Chat Service — FastAPI с SSE-стримингом
=========================================
Полный RAG pipeline за один HTTP-запрос:

  POST /chat/stream
  ─────────────────
  1. Embed вопроса          → мxbai-embed-large-v1 (та же модель, что в воркере)
  2. Vector Retrieval        → pgvector top-5 чанков
  3. Context Construction    → склейка чанков с Header Injection
  4. History (Redis)         → подгружаем последние 5 пар
  5. Prompt Engineering      → системный промпт + история + контекст + вопрос
  6. LLM Streaming           → Server-Sent Events (текст по мере генерации)
  7. Сохраняем ответ в Redis → для следующего хода диалога

  GET /ping            — healthcheck
  DELETE /chat/{user_id}/history — очистка истории диалога

Переменные окружения (.env):
  DATABASE_URL=postgresql+asyncpg://admin:password@localhost:5432/hse_rag
  REDIS_URL=redis://localhost:6379
  LLM_PROVIDER=stub | openai | ollama
  OPENAI_API_KEY=sk-...          (если LLM_PROVIDER=openai)
  OPENAI_MODEL=gpt-4o-mini       (опционально)
  OLLAMA_URL=http://localhost:11434  (если LLM_PROVIDER=ollama)
  OLLAMA_MODEL=llama3            (опционально)
  EMBEDDER_NAME=mixedbread-ai/mxbai-embed-large-v1
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

# ── Внутренние модули ──────────────────────────────────────────────────────
from core.config import settings
from core.db import async_session
from services.api.retrieval import vector_search
from services.chat_service.history import ChatHistory
from services.chat_service.rag_logic import (
    build_context,
    build_messages,
    stream_llm_response,
)

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ChatService")

# ---------------------------------------------------------------------------
# Загрузка эмбеддинг-модели
# Используем ту же модель, что и ingest_worker, чтобы векторы совпадали!
# ---------------------------------------------------------------------------
EMBEDDER_NAME = settings.EMBEDDER_MODEL_NAME
REDIS_URL = settings.REDIS_URL

try:
    from sentence_transformers import SentenceTransformer
    logger.info(f"⏳ Загружаю эмбеддинг-модель: {EMBEDDER_NAME} ...")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    logger.info("✅ Модель загружена успешно")
except Exception as exc:
    logger.error(f"❌ Не удалось загрузить модель: {exc}")
    embedder = None

# Один экземпляр ChatHistory на всё приложение (пул Redis-соединений внутри)
chat_history = ChatHistory(redis_url=REDIS_URL, max_messages=10)

# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HSE Auto-Mentor — Chat API",
    description="RAG-based Q&A для студентов МИЭМ НИУ ВШЭ",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Схемы запросов
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str


# ---------------------------------------------------------------------------
# Dependency: сессия БД
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Вспомогательная функция: встраивание вопроса
# ---------------------------------------------------------------------------
def embed_query(text: str) -> list[float]:
    """Векторизует вопрос пользователя через ту же модель, что и voркер."""
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Эмбеддинг-модель не загружена. Проверь логи сервиса.",
        )
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Маршруты
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Корневой маршрут — для проверки в браузере."""
    return {
        "service": "HSE Auto-Mentor Chat API",
        "status": "running",
        "docs": "/docs",
        "health": "/ping",
    }


@app.get("/ping", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "embedder": EMBEDDER_NAME,
        "llm_provider": os.environ.get("LLM_PROVIDER", "stub"),
    }


@app.delete("/chat/{user_id}/history", tags=["Chat"])
async def clear_history(user_id: str):
    """Очищает историю диалога пользователя (например, для 'начать заново')."""
    await chat_history.clear(user_id)
    return {"status": "cleared", "user_id": user_id}


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Главный эндпоинт: принимает вопрос, запускает RAG pipeline,
    стримит ответ LLM через Server-Sent Events.

    Формат SSE-ответа:
      data: токен1\\n\\n
      data: токен2\\n\\n
      ...
      data: [DONE]\\n\\n
    """
    user_id = request.user_id
    question = request.message.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым.")

    logger.info(f"💬 user={user_id!r}: {question!r}")

    # ── 1. Семантическое векторное представление вопроса ───────────────────
    try:
        query_vec = embed_query(question)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"embed_query error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ошибка эмбеддинга: {exc}")

    # ── 2. Vector Retrieval: top-5 близких чанков из pgvector ─────────────
    try:
        chunks = await vector_search(db, query_vec, top_k=5)
        logger.info(f"🔍 Найдено чанков: {len(chunks)}")
        for i, c in enumerate(chunks):
            logger.debug(f"  [{i+1}] sim={c['similarity']:.3f} | {c['text'][:60]}...")
    except Exception as exc:
        logger.error(f"vector_search error: {exc}")
        chunks = []   # Деградируем: отвечаем без контекста

    # ── 3. Context Construction ────────────────────────────────────────────
    context = build_context(chunks)

    # ── 4. История диалога из Redis ────────────────────────────────────────
    history = await chat_history.get(user_id)
    logger.info(f"📜 История диалога: {len(history)} сообщений")

    # ── 5. Prompt Engineering ──────────────────────────────────────────────
    messages = build_messages(
        context=context,
        history=history,
        user_question=question,
    )

    # ── 6 + 7. Streaming LLM + сохранение в историю ───────────────────────
    async def sse_generator() -> AsyncGenerator[str, None]:
        """
        Генератор для StreamingResponse.
        Стримит ответ LLM и параллельно накапливает полный текст для Redis.
        """
        full_response_parts = []

        # Сначала сохраняем вопрос пользователя
        await chat_history.add(user_id, "user", question)

        try:
            async for token in stream_llm_response(messages):
                full_response_parts.append(token)
                # SSE-формат: каждый кусок — отдельное событие
                yield f"data: {token}\n\n"
        except Exception as exc:
            logger.error(f"LLM streaming error: {exc}")
            yield f"data: Ошибка генерации ответа: {exc}\n\n"

        # После окончания стрима — сохраняем полный ответ ассистента
        full_response = "".join(full_response_parts)
        if full_response:
            await chat_history.add(user_id, "assistant", full_response)
            logger.info(f"✅ Ответ сохранён в историю ({len(full_response)} симв.)")

        # Финальный маркер для клиента
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            # Отключаем буферизацию прокси и браузеров
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
