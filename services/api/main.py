from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
# from services.api.retrieval import hybrid_search # Пока закомментируем, чтобы не мешало тестам

app = FastAPI(title="HSE Auto-Mentor API")

# 1. Описываем структуру JSON, которую ждем от клиента
class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.get("/ping")
async def health_check():
    return {"status": "ok", "message": "RAG System is live"}

# 2. Обрабатываем POST запрос и возвращаем поток
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    # 3. Создаем асинхронный генератор для потоковой выдачи текста
    async def fake_video_streamer():
        response_text = f"Система готова. ID студента: {request.user_id}. Твой вопрос: '{request.message}'. Контекст подгрузится после индексации."

        # Разбиваем текст на слова и отправляем по одному с небольшой задержкой
        for word in response_text.split():
            yield f"{word} "
            await asyncio.sleep(0.1) # Имитация задержки генерации LLM

    # Возвращаем специальный ответ, который не закрывает соединение сразу
    return StreamingResponse(fake_video_streamer(), media_type="text/event-stream")
