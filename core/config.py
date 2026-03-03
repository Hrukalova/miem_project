from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Настройки БД
    DATABASE_URL: str = "postgresql+asyncpg://admin:password@localhost:5432/hse_rag"

    # Настройки MinIO (локально в докере будут такие)
    S3_ENDPOINT: str = "localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "documents"

    # Модели (можешь менять на легкие для тестов)
    EMBEDDER_NAME: str = "BAAI/bge-m3"
    RERANKER_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"

settings = Settings()
