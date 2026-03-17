from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # База данных
    DATABASE_URL: str = "postgresql+asyncpg://admin:password@localhost:5432/hse_rag"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Google Drive (для загрузки DOCX с диска)
    GOOGLE_DRIVE_FOLDER_ID: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = "google_creds.json"

    # Эмбеддинг-модель — имя совпадает с .env
    EMBEDDER_MODEL_NAME: str = "mixedbread-ai/mxbai-embed-large-v1"

    # Параметры чанкования
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    # Reranker (опционально)
    RERANKER_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MinIO / S3 (опционально)
    S3_ENDPOINT: str = "localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "documents"

    class Config:
        env_file = ".env"


settings = Settings()
