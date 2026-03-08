# scripts/sync_drive_to_db.py
import asyncio
import os
import uuid
import logging
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from sqlalchemy import select

# Импортируем вашу БД и модели
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.db import async_session
from core.models import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DriveSync")

load_dotenv()

CREDS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "google_creds.json")

def get_drive_service():
    if not os.path.exists(CREDS_PATH):
        raise FileNotFoundError(f"Файл ключа {CREDS_PATH} не найден!")

    creds = service_account.Credentials.from_service_account_file(
        CREDS_PATH, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build('drive', 'v3', credentials=creds)

async def sync_drive():
    logger.info("🔄 Подключаюсь к Google Drive...")

    def fetch_all_files():
        service = get_drive_service()
        # Ищем ВСЕ файлы, кроме папок, которые не в корзине
        query = "mimeType != 'application/vnd.google-apps.folder' and trashed=false"

        all_files = []
        page_token = None

        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                pageSize=1000,
                pageToken=page_token
            ).execute()

            all_files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return all_files

    files = await asyncio.to_thread(fetch_all_files)
    logger.info(f"📂 Всего найдено файлов (внутри всех папок): {len(files)}")

    if not files:
        logger.warning("⚠️ Файлы не найдены! Проверь, дан ли доступ сервисному аккаунту к папкам.")
        return

    async with async_session() as session:
        added_count = 0
        for f in files:
            file_id = f.get("id")
            name = f.get("name")
            mime_type = f.get("mimeType")
            link = f.get("webViewLink")

            # Проверяем, есть ли уже этот файл в БД
            stmt = select(Document).where(Document.raw_content_link == link)
            res = await session.execute(stmt)
            existing_doc = res.scalar_one_or_none()

            # Определяем тип контента для Ingest Worker
            c_type = "other"
            if "pdf" in mime_type:
                c_type = "pdf"
            elif "word" in mime_type or "officedocument.wordprocessingml" in mime_type:
                c_type = "docx"
            elif "spreadsheet" in mime_type or "excel" in mime_type:
                c_type = "spreadsheet"

            if existing_doc:
                logger.info(f"⏩ Уже в БД: {name} ({c_type})")
            else:
                logger.info(f"➕ Добавляю: {name} (MIME: {mime_type})")
                new_doc = Document(
                    id=uuid.uuid4(),
                    title=name,
                    source_type="google_drive",
                    raw_content_link=link,
                    status="pending",
                    doc_metadata={
                        "google_drive_id": file_id,
                        "content_type": c_type,
                        "mime_type": mime_type
                    }
                )
                session.add(new_doc)
                added_count += 1

        if added_count > 0:
            await session.commit()
            logger.info(f"✅ Успешно добавлено {added_count} новых файлов в базу!")
        else:
            logger.info("✅ Все найденные файлы уже есть в базе.")

if __name__ == "__main__":
    asyncio.run(sync_drive())
