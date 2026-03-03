from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# URL твоей базы данных (можешь вынести в config.py позже)
DATABASE_URL = "postgresql+asyncpg://admin:password@localhost:5432/hse_rag"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()
