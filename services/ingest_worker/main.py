# services/ingest_worker/main.py
import asyncio
import logging
from sqlalchemy import select, update
from sentence_transformers import SentenceTransformer

from core.db import async_session
from core.models import Document, Chunk
from .parsers import parse_url, get_content, parse_pdf, parse_docx
from .chunker import SemanticChunker

# Настройка логов, чтобы видеть, что происходит
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestWorker")

# Загружаем модель для векторизации (при первом запуске она скачается ~1ГБ)
# mxbai-embed-large-v1 - одна из лучших для русского языка
try:
    embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    logger.info("✅ Модель эмбеддингов успешно загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")

chunker = SemanticChunker()

async def process_one_document():
    """Функция обработки одного документа из очереди"""
    async with async_session() as session:
        # 1. Атомарно забираем документ со статусом pending
        stmt = (
            select(Document)
            .where(Document.status == "pending")
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        res = await session.execute(stmt)
        doc = res.scalar_one_or_none()

        if not doc:
            return False # Документов нет

        logger.info(f"🔄 Обработка документа: {doc.title}")
        doc.status = "processing"
        await session.commit()

        try:
            # 2. Определяем, как парсить контент
            link = doc.raw_content_link
            if link.startswith("http"):
                if link.endswith(".pdf"):
                    content_bytes = await get_content(link)
                    text = parse_pdf(content_bytes)
                elif link.endswith(".docx"):
                    content_bytes = await get_content(link)
                    text = parse_docx(content_bytes)
                else:
                    # По умолчанию считаем, что это веб-страница (как твоя ссылка)
                    text = await parse_url(link)
            else:
                # Локальный файл (если ссылка ведет на data/...)
                content_bytes = await get_content(link)
                if link.endswith(".pdf"): text = parse_pdf(content_bytes)
                elif link.endswith(".docx"): text = parse_docx(content_bytes)
                else: text = content_bytes.decode('utf-8')

            doc.content_text = text

            # 3. Нарезка на чанки (интерфейс для твоего напарника)
            chunks_data = await chunker.create_chunks(text, doc.title)
            logger.info(f"✂️ Создано чанков: {len(chunks_data)}")

            # 4. Векторизация и сохранение
            for c in chunks_data:
                # Генерируем вектор (эмбеддинг)
                embedding_vector = embedder.encode(c["text"]).tolist()

                new_chunk = Chunk(
                    document_id=doc.id,
                    text=c["text"],
                    embedding=embedding_vector,
                    chunk_metadata=c["meta"]
                )
                session.add(new_chunk)

            doc.status = "indexed"
            logger.info(f"✅ Документ '{doc.title}' успешно проиндексирован")

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {doc.title}: {e}")
            doc.status = "failed"

        await session.commit()
        return True

async def main_loop():
    """Основной цикл работы воркера"""
    logger.info("🚀 Воркер-пылесос запущен и опрашивает базу...")
    while True:
        was_processed = await process_one_document()
        if not was_processed:
            # Если документов нет, спим 5 секунд
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Воркер остановлен пользователем")
