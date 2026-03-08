# services/ingest_worker/main.py
"""
Новичок №2 — "The Indexer": Ingest Worker (main pipeline)
==========================================================
Полный pipeline:
  1. Берём документ со статусом «pending» из БД.
  2. Определяем тип контента и парсим (через parsers.py — зона Новичка №1).
  3. Получаем breadcrumb из иерархии topics (topic_helper.py).
  4. Нарезаем на семантические чанки с Header Injection (chunker.py).
  5. Генерируем эмбеддинги через BAAI/bge-m3.
  6. Database Ops:
       - Сначала удаляем все старые чанки документа (на случай переиндексации).
       - Сохраняем новые чанки с векторами.
  7. Устанавливаем статус «indexed» или «failed».

Запуск:
  python -m services.ingest_worker.main
"""

import asyncio
import logging
import os

from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sentence_transformers import SentenceTransformer

from core.db import async_session
from core.config import settings
from core.models import Document, Chunk
from .parsers import parse_url, get_content, parse_pdf, parse_docx
from .chunker import SemanticChunker
from .topic_helper import TopicHelper

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("IngestWorker")

# ---------------------------------------------------------------------------
# Загрузка эмбеддинг-модели (BAAI/bge-m3 — мощная многоязычная модель)
# При первом запуске скачивается ~1.1 GB
# ---------------------------------------------------------------------------
EMBEDDER_NAME = settings.EMBEDDER_MODEL_NAME

try:
    logger.info(f"⏳ Загружаю эмбеддинг-модель: {EMBEDDER_NAME} ...")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    EMBEDDING_DIM = embedder.get_sentence_embedding_dimension()
    logger.info(f"✅ Модель загружена. Размерность вектора: {EMBEDDING_DIM}")
except Exception as exc:
    logger.error(f"❌ Ошибка загрузки модели '{EMBEDDER_NAME}': {exc}")
    logger.warning("⚠️  Буду работать в fallback-режиме — без векторизации")
    embedder = None
    EMBEDDING_DIM = 1024  # Сохраняем дефолт для схемы БД

# ---------------------------------------------------------------------------
# Инициализация вспомогательных компонентов
# ---------------------------------------------------------------------------
chunker = SemanticChunker(
    embedder=embedder,
    t_sim=0.5,
    max_tokens=settings.CHUNK_SIZE,
    min_tokens=30,
    window_size=2,
)

topic_helper = TopicHelper()

# Путь к Excel-метаданным (можно переопределить через ENV)
METADATA_XLSX = os.environ.get(
    "METADATA_XLSX",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "documents.xlsx"),
)

def _ensure_topics_loaded():
    """Лениво загружаем топики при первом обращении."""
    if not topic_helper.is_loaded():
        if os.path.exists(METADATA_XLSX):
            logger.info(f"📂 Загружаю топики из '{METADATA_XLSX}'")
            topic_helper.load_from_excel(METADATA_XLSX, sheet_name="topics")
        else:
            logger.warning(
                f"⚠️  Файл метаданных не найден: '{METADATA_XLSX}'. "
                "Header Injection без breadcrumb-категорий."
            )


# ---------------------------------------------------------------------------
# Вспомогательные функции Database Ops
# ---------------------------------------------------------------------------

async def _delete_old_chunks(session, doc_id) -> int:
    """
    Database Ops: удаляем все чанки документа перед переиндексацией.
    Это гарантирует, что в БД не останется устаревших данных, если документ
    был обновлён.

    Возвращает количество удалённых чанков.
    """
    stmt = delete(Chunk).where(Chunk.document_id == doc_id)
    result = await session.execute(stmt)
    deleted_count = result.rowcount
    if deleted_count:
        logger.info(f"🗑️  Удалено {deleted_count} старых чанков для документа {doc_id}")
    return deleted_count


async def _save_chunks(session, doc_id, chunks_data: list) -> None:
    """
    Database Ops: батч-запись чанков в pgvector.

    chunks_data — список словарей {"text": str, "meta": dict, "embedding": list}
    """
    for chunk in chunks_data:
        new_chunk = Chunk(
            document_id=doc_id,
            text=chunk["text"],
            embedding=chunk["embedding"],
            chunk_metadata=chunk["meta"],
        )
        session.add(new_chunk)


def _generate_embeddings(texts: list) -> list:
    """
    Генерируем эмбеддинги для списка текстов.
    Возвращает список списков float.

    Для BAAI/bge-m3 рекомендуется encode с normalize_embeddings=True.
    """
    if embedder is None:
        logger.warning("Модель не загружена — возвращаю нулевые векторы")
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    try:
        # batch_size=32 — хороший баланс скорости / памяти на CPU
        vecs = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True,   # нормализация нужна для cosine_ops в pgvector
        )
        return [v.tolist() for v in vecs]
    except Exception as exc:
        logger.error(f"Ошибка векторизации батча: {exc}")
        return [[0.0] * EMBEDDING_DIM for _ in texts]


# ---------------------------------------------------------------------------
# Основная функция обработки документа
# ---------------------------------------------------------------------------

async def process_one_document() -> bool:
    """
    Атомарно забирает один pending-документ из БД и обрабатывает его.
    Возвращает True, если документ был обработан, False — если очередь пуста.
    """
    _ensure_topics_loaded()

    async with async_session() as session:
        # --- 1. Забираем документ (SELECT FOR UPDATE SKIP LOCKED) ---
        stmt = (
            select(Document)
            .where(Document.status == "pending")
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        res = await session.execute(stmt)
        doc = res.scalar_one_or_none()

        if not doc:
            return False  # Очередь пуста

        logger.info(f"🔄 Начинаю обработку: [{doc.id}] '{doc.title}'")
        doc.status = "processing"
        await session.commit()

        try:
            # --- 2. Парсинг контента (зона Новичка №1) ---
            text = await _extract_text(doc)
            doc.content_text = text
            logger.info(f"📄 Извлечён текст: {len(text)} символов")

            # --- 3. Получаем breadcrumb из листа topics ---
            topic_id = (doc.doc_metadata or {}).get("topic_id")
            breadcrumb = topic_helper.get_breadcrumb(topic_id) if topic_id else None
            if breadcrumb:
                logger.info(f"🔖 Breadcrumb: {breadcrumb}")

            # --- 4. Семантическая нарезка с Header Injection ---
            chunks_data = await chunker.create_chunks(
                text=text,
                doc_title=doc.title,
                topic_breadcrumb=breadcrumb,
            )
            logger.info(f"✂️  Создано чанков: {len(chunks_data)}")

            if not chunks_data:
                logger.warning(f"⚠️  Документ '{doc.title}' не дал чанков — пропускаем индексацию")
                doc.status = "indexed"
                await session.commit()
                return True

            # --- 5. Генерация эмбеддингов (BAAI/bge-m3) ---
            texts_to_embed = [c["text"] for c in chunks_data]
            embeddings = _generate_embeddings(texts_to_embed)
            for chunk, emb in zip(chunks_data, embeddings):
                chunk["embedding"] = emb

            # --- 6. Database Ops ---
            # 6a. Удаляем старые чанки (на случай переиндексации)
            await _delete_old_chunks(session, doc.id)
            # 6b. Сохраняем новые чанки
            await _save_chunks(session, doc.id, chunks_data)

            doc.status = "indexed"
            logger.info(f"✅ Документ '{doc.title}' успешно проиндексирован ({len(chunks_data)} чанков)")

        except Exception as exc:
            logger.error(f"❌ Ошибка при обработке '{doc.title}': {exc}", exc_info=True)
            doc.status = "failed"

        await session.commit()
        return True


async def _extract_text(doc: Document) -> str:
    """Определяем способ парсинга по ссылке и типу документа."""
    from .parsers import (
        get_content, parse_url, parse_pdf, parse_docx, parse_excel,
        extract_gdrive_id, get_gdrive_content
    )

    link = doc.raw_content_link or ""
    content_type = (doc.doc_metadata or {}).get("content_type", "html")
    doc_title_lower = doc.title.lower() if doc.title else ""

    # 1. Если это ссылка на Google Диск
    if "drive.google.com" in link:
        file_id = extract_gdrive_id(link)
        if not file_id:
            raise ValueError(f"Не удалось извлечь ID из ссылки Google Drive: {link}")

        logger.info(f"📥 Скачиваю файл с Google Drive (ID: {file_id})")
        content_bytes = await get_gdrive_content(file_id)

        # Определяем парсер по content_type или расширению в названии
        if content_type == "pdf" or doc_title_lower.endswith(".pdf"):
            return parse_pdf(content_bytes)
        elif content_type == "spreadsheet" or doc_title_lower.endswith(".xlsx"):
            return parse_excel(content_bytes)
        elif content_type in ["docx", "doc"] or doc_title_lower.endswith(".docx"):
            return parse_docx(content_bytes)
        else:
            return content_bytes.decode("utf-8", errors="replace")

    # 2. Обычная http/https ссылка
    elif link.startswith("http"):
        if link.endswith(".pdf") or content_type == "pdf":
            content_bytes = await get_content(link)
            return parse_pdf(content_bytes)
        elif link.endswith(".docx") or content_type == "docx":
            content_bytes = await get_content(link)
            return parse_docx(content_bytes)
        else:
            return await parse_url(link)

    # 3. Локальный путь к файлу
    elif link:
        content_bytes = await get_content(link)
        if link.endswith(".pdf"):
            return parse_pdf(content_bytes)
        elif link.endswith(".docx"):
            return parse_docx(content_bytes)
        else:
            return content_bytes.decode("utf-8", errors="replace")

    # 4. Текст уже загружен в БД напрямую
    elif doc.content_text:
        return doc.content_text

    else:
        raise ValueError(f"Нет ссылки и нет cached-текста для документа '{doc.title}'")

# ---------------------------------------------------------------------------
# Главный цикл воркера
# ---------------------------------------------------------------------------

async def main_loop():
    """Бесконечный цикл опроса очереди pending-документов."""
    logger.info("🚀 Ingest Worker запущен. Опрашиваю очередь...")
    while True:
        try:
            was_processed = await process_one_document()
        except Exception as exc:
            logger.error(f"Критическая ошибка в main_loop: {exc}", exc_info=True)
            was_processed = False

        if not was_processed:
            # Очередь пуста — ждём перед следующим опросом
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Воркер остановлен пользователем")
