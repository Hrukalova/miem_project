# scripts/init_db.py
import asyncio
import sys
import os

# Добавляем корень проекта в путь, чтобы импорты работали
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import engine
from core.models import Base
from sqlalchemy import text

async def init_models():
    async with engine.begin() as conn:
        # 1. Активируем pgvector (требует прав суперпользователя в БД)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 2. Создаем таблицы
        # В режиме разработки можно использовать drop_all(), если нужно сбросить схему
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    print("🚀 База данных успешно инициализирована: таблицы созданы, pgvector активен.")

if __name__ == "__main__":
    asyncio.run(init_models())
