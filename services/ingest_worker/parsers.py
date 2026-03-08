# services/ingest_worker/parsers.py
import aiohttp
import fitz  # PyMuPDF
from docx import Document as DocxReader
from readability import Document as ReadabilityDoc
from bs4 import BeautifulSoup
import io
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Инициализация Google API (сразу при импорте модуля)
creds = service_account.Credentials.from_service_account_file(
    "google_creds.json", # Путь к твоему ключу
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build('drive', 'v3', credentials=creds)

def extract_gdrive_id(url: str) -> str | None:
    """Вытаскивает ID файла из стандартной ссылки Google Drive."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

async def get_gdrive_content(file_id: str) -> bytes:
    """Скачивает файл с Google Диска, правильно обрабатывая нативные форматы."""
    import asyncio
    from googleapiclient.http import MediaIoBaseDownload
    import io

    def _download():
        service = get_drive_service()

        # 1. Узнаем точный тип файла на Диске
        file_meta = service.files().get(fileId=file_id, fields="mimeType").execute()
        mime = file_meta.get("mimeType", "")

        # 2. Если это Google Документ -> экспортируем как DOCX
        if mime == "application/vnd.google-apps.document":
            request = service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        # 3. Если это Google Таблица -> экспортируем как XLSX
        elif mime == "application/vnd.google-apps.spreadsheet":
            request = service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        # 4. В противном случае (обычные pdf, загруженные docx) -> скачиваем напрямую
        else:
            request = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue()

    return await asyncio.to_thread(_download)

async def get_content(path_or_url: str) -> bytes:
    """Универсальный загрузчик: качает по HTTP или читает с диска."""
    if path_or_url.startswith("http"):
        async with aiohttp.ClientSession() as session:
            async with session.get(path_or_url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.read()
                raise Exception(f"Ошибка загрузки: статус {resp.status}")
    else:
        # Убираем возможные лишние пробелы из пути
        clean_path = path_or_url.strip()
        if os.path.exists(clean_path):
            with open(clean_path, "rb") as f:
                return f.read()
        raise FileNotFoundError(f"Файл не найден по пути: {clean_path}")

async def parse_url(url: str) -> str:
    """Парсинг HTML через Readability для очистки от мусора."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            html = await resp.text()
            doc = ReadabilityDoc(html)
            clean_html = doc.summary()
            soup = BeautifulSoup(clean_html, "lxml")
            return f"{doc.title()}\n{soup.get_text(separator=' ', strip=True)}"

def parse_pdf(file_bytes: bytes) -> str:
    """Парсинг PDF через PyMuPDF."""
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def parse_docx(file_bytes: bytes) -> str:
    """Парсинг DOCX."""
    doc = DocxReader(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

import io
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os

# Ленивая инициализация клиента Google Drive
_drive_service = None

def get_drive_service():
    global _drive_service
    if _drive_service is None:
        # Убедитесь, что путь к ключу правильный
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "google_creds.json")
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        _drive_service = build('drive', 'v3', credentials=creds)
    return _drive_service

def extract_gdrive_id(url: str) -> str | None:
    """Вытаскивает ID файла из стандартной ссылки Google Drive."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    # Если ссылка вида ?id=XXXXX
    match_id = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    return match_id.group(1) if match_id else None

async def get_gdrive_content(file_id: str) -> bytes:
    """Скачивает файл с Google Диска напрямую в оперативную память."""
    import asyncio

    def _download():
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue()

    # Запускаем синхронную скачку в отдельном потоке, чтобы не блокировать async
    return await asyncio.to_thread(_download)
