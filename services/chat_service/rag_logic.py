# services/chat_service/rag_logic.py
"""
RAG Logic: Context Construction + Prompt Engineering + LLM вызов
=================================================================
Реализует шаги 3 и 4 из ТЗ:
  3. Context Construction — склейка чанков в один контекст.
  4. Prompt Engineering   — формирование системного промпта с историей диалога.

Поддерживает два режима LLM (выбирается через settings.LLM_PROVIDER):
  - "openai"  : GPT-4o / GPT-4 / GPT-3.5 через OpenAI API
  - "ollama"  : локальная Llama (llama3, mistral и т.д.) через Ollama REST API
  - "stub"    : заглушка без LLM (для тестов без ключей)
"""

from __future__ import annotations

import logging
import os
from typing import List, AsyncGenerator

logger = logging.getLogger("RagLogic")

# ---------------------------------------------------------------------------
# Системный промпт для ментора МИЭМ / ВШЭ
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — автоматизированный наставник студентов МИЭМ НИУ ВШЭ.
Твоя задача: помогать студентам разбираться в документах, положениях и правилах Вышки.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного КОНТЕКСТА.
2. Если в контексте нет ответа — честно скажи: "К сожалению, в моей базе знаний нет точного ответа на этот вопрос."
3. Не придумывай цифры, даты, суммы. Цитируй документ если важно.
4. Отвечай на русском языке, кратко и по делу.
5. Можешь учитывать ИСТОРИЮ ДИАЛОГА для понимания контекста вопроса."""


def build_context(chunks: List[dict]) -> str:
    """
    Context Construction: собирает релевантные чанки в единый контекст.
    
    Каждый чанк уже содержит Header Injection, напр.:
      Документ: Положение о стипендиях
      Категория: Про деньги > Стипендии
      ---
      Студент имеет право на получение...

    Чанки разделяются двойным переносом строки.
    """
    if not chunks:
        return "Контекст не найден."
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Добавляем номер источника для прозрачности
        parts.append(f"[Источник {i}]\n{chunk['text']}")
    
    return "\n\n".join(parts)


def build_messages(
    context: str,
    history: List[dict],
    user_question: str,
) -> List[dict]:
    """
    Prompt Engineering: собирает список сообщений для ChatCompletion API.

    Структура:
        [system_prompt]
        [history[-4:]]   ← последние 4 сообщения из истории (без текущего)
        [user: context + вопрос]
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Подмешиваем историю диалога (без самого нового вопроса)
    # Берём только последние 4 сообщения, чтобы не раздувать контекст
    recent_history = history[-4:] if history else []
    messages.extend(recent_history)

    # Финальный пользовательский запрос = контекст + вопрос
    user_message = (
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ВОПРОС СТУДЕНТА:\n{user_question}"
    )
    messages.append({"role": "user", "content": user_message})

    return messages


# ---------------------------------------------------------------------------
# Streaming вызов LLM
# ---------------------------------------------------------------------------

async def stream_llm_response(messages: List[dict]) -> AsyncGenerator[str, None]:
    """
    Отправляет промпт в LLM и стримит ответ токен за токеном.

    Режим выбирается через переменную окружения LLM_PROVIDER:
      - "openai"  (по умолчанию, если есть OPENAI_API_KEY)
      - "ollama"  (локальный запуск, если нет API ключа)
      - "stub"    (для тестов)

    Yields
    ------
    str : фрагменты текста ответа (токены / слова)
    """
    provider = os.environ.get("LLM_PROVIDER", "stub").lower()

    if provider == "openai":
        async for chunk in _stream_openai(messages):
            yield chunk
    elif provider == "ollama":
        async for chunk in _stream_ollama(messages):
            yield chunk
    else:
        # Stub-режим: детерминированный ответ-заглушка для тестов
        async for chunk in _stream_stub(messages):
            yield chunk


async def _stream_openai(messages: List[dict]) -> AsyncGenerator[str, None]:
    """Streaming через OpenAI ChatCompletion API."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.3,   # ниже = точнее, выше = свободнее
            max_tokens=1024,
        )
        async for event in stream:
            delta = event.choices[0].delta
            if delta.content:
                yield delta.content

    except ImportError:
        logger.error("openai не установлен. pip install openai")
        yield "Ошибка: библиотека openai не найдена."
    except KeyError:
        logger.error("OPENAI_API_KEY не задан в .env")
        yield "Ошибка: не задан OPENAI_API_KEY."
    except Exception as exc:
        logger.error(f"OpenAI streaming error: {exc}")
        yield f"Ошибка при обращении к LLM: {exc}"


async def _stream_ollama(messages: List[dict]) -> AsyncGenerator[str, None]:
    """Streaming через локальный Ollama (http://localhost:11434)."""
    try:
        import aiohttp
        import json as _json

        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3")

        # Ollama /api/chat — поддерживает stream: true
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.3},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for line in resp.content:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except Exception:
                        continue

    except Exception as exc:
        logger.error(f"Ollama streaming error: {exc}")
        yield f"Ошибка при обращении к Ollama: {exc}"


async def _stream_stub(messages: List[dict]) -> AsyncGenerator[str, None]:
    """
    Заглушка — не требует LLM.
    Извлекает вопрос из последнего сообщения и возвращает фиксированный ответ.
    Используй для разработки без ключей.
    """
    import asyncio

    # Вытаскиваем вопрос из последнего user-сообщения
    question = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            # Вопрос после "ВОПРОС СТУДЕНТА:\n"
            parts = msg["content"].split("ВОПРОС СТУДЕНТА:\n")
            question = parts[-1].strip() if len(parts) > 1 else msg["content"]
            break

    stub_answer = (
        f"[STUB-режим, LLM не подключён]\n\n"
        f"Получен вопрос: «{question}»\n\n"
        f"Для подключения реального LLM установи переменную окружения:\n"
        f"  LLM_PROVIDER=openai  +  OPENAI_API_KEY=sk-...\n"
        f"  LLM_PROVIDER=ollama  +  OLLAMA_MODEL=llama3"
    )

    # Имитируем streaming по словам
    for word in stub_answer.split():
        yield word + " "
        await asyncio.sleep(0.03)
