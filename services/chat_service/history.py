# services/chat_service/history.py
"""
Chat History через Redis
===========================
Сохраняет последние N сообщений диалога в Redis.
Каждый диалог — отдельный список (list) с ключом  chat:{user_id}.

Формат одного сообщения в Redis:
  JSON-строка: {"role": "user"|"assistant", "content": "текст"}

Пример:
    history = ChatHistory(redis_url="redis://localhost:6379")
    await history.add("user_42", "user", "Как получить стипендию?")
    await history.add("user_42", "assistant", "Для получения стипендии...")
    messages = await history.get("user_42")
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("ChatHistory")

MAX_HISTORY = 10   # храним 10 «половинок» = 5 пар вопрос/ответ


class ChatHistory:
    """
    Асинхронный менеджер истории разговора в Redis.

    Параметры
    ----------
    redis_url : str
        URL подключения, напр. "redis://localhost:6379"
    max_messages : int
        Максимальное количество хранимых сообщений (default: 10 → 5 пар)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", max_messages: int = MAX_HISTORY):
        self._client = None
        self._redis_url = redis_url
        self.max_messages = max_messages

    async def _get_client(self):
        """Лениво создаём клиент при первом вызове."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                logger.info(f"✅ Redis подключен: {self._redis_url}")
            except Exception as exc:
                logger.error(f"❌ Ошибка подключения к Redis: {exc}")
                self._client = None
        return self._client

    def _key(self, user_id: str) -> str:
        return f"chat:{user_id}"

    async def add(self, user_id: str, role: str, content: str) -> None:
        """
        Добавляет сообщение в историю.
        Если история превысила max_messages — обрезаем старые сообщения.

        role: "user" или "assistant"
        """
        client = await self._get_client()
        if client is None:
            return   # Redis недоступен — тихо пропускаем

        try:
            key = self._key(user_id)
            entry = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            # Добавляем в правый конец списка
            await client.rpush(key, entry)
            # Обрезаем с начала: оставляем только последние max_messages
            await client.ltrim(key, -self.max_messages, -1)
            # TTL 24 часа — старые диалоги не занимают память вечно
            await client.expire(key, 60 * 60 * 24)
        except Exception as exc:
            logger.warning(f"Redis write error для user '{user_id}': {exc}")

    async def get(self, user_id: str) -> list:
        """
        Возвращает историю диалога в виде списка словарей
        [{"role": "user"|"assistant", "content": str}, ...]
        """
        client = await self._get_client()
        if client is None:
            return []

        try:
            key = self._key(user_id)
            raw_messages = await client.lrange(key, 0, -1)
            return [json.loads(m) for m in raw_messages]
        except Exception as exc:
            logger.warning(f"Redis read error для user '{user_id}': {exc}")
            return []

    async def clear(self, user_id: str) -> None:
        """Очищает историю диалога (например, если студент начинает новый чат)."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(self._key(user_id))
        except Exception as exc:
            logger.warning(f"Redis delete error для user '{user_id}': {exc}")

    async def close(self) -> None:
        """Закрывает соединение с Redis."""
        if self._client:
            await self._client.aclose()
            self._client = None
