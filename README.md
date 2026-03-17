# HSE Auto-Mentor RAG

**Автонаставник для студентов МИЭМ НИУ ВШЭ** — RAG-система, которая отвечает на вопросы по документам университета.

---

## 🏗️ Архитектура

```
┌──────────────────────┐     ┌───────────────────────┐
│    Ingest Worker     │     │     Chat Service       │
│                      │     │                        │
│  URL → parse → chunk │     │  вопрос → embed        │
│  → embed → pgvector  │     │  → pgvector top-5      │
│                      │     │  → LLM → SSE stream    │
└──────────────────────┘     └───────────────────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
      ┌─────────────┴──────────────┐
      │  PostgreSQL + pgvector     │  ← хранит чанки + векторы (HNSW)
      │  Redis                     │  ← история диалогов (5 пар, 24ч TTL)
      └────────────────────────────┘
```

---

## ✅ Текущий статус

### Ingest Worker — `services/ingest_worker/`
| Компонент | Файл | Статус |
|---|---|---|
| Парсинг HTML/PDF/DOCX | `parsers.py` | ✅ |
| Semantic Chunking (t_sim + sliding window) | `chunker.py` | ✅ |
| Header Injection из иерархии topics | `chunker.py` | ✅ |
| Хлебные крошки из Excel (topic_helper) | `topic_helper.py` | ✅ |
| Embeddings — `BAAI/bge-m3` (batch, normalized) | `main.py` | ✅ |
| Database Ops: удаление старых чанков | `main.py` | ✅ |

### Chat Service — `services/chat_service/`
| Компонент | Файл | Статус |
|---|---|---|
| Vector search (pgvector `<=>`) | `services/api/retrieval.py` | ✅ |
| Context Construction | `rag_logic.py` | ✅ |
| Prompt Engineering (системный промпт + история) | `rag_logic.py` | ✅ |
| LLM Streaming — OpenAI / Ollama / stub | `rag_logic.py` | ✅ |
| Chat History (Redis, 5 пар, TTL 24ч) | `history.py` | ✅ |
| SSE-эндпоинт `POST /chat/stream` | `main.py` | ✅ |

---

## 🚀 Быстрый старт (Windows PowerShell)

> ⚠️ **На Windows** переменные окружения нельзя задавать через `KEY=value` перед командой.
> Всё настраивается через файл `.env` в корне проекта — библиотека читает его автоматически.

### Шаг 1 — Создай файл `.env`

Создай файл `.env` в папке `miem_project-main` со следующим содержимым:

```env
# Если БД запущена в Docker (docker-compose up -d) — используй localhost:
DATABASE_URL=postgresql+asyncpg://admin:password@localhost:5432/hse_rag

# Redis
REDIS_URL=redis://localhost:6379/0

# Google Drive
GOOGLE_DRIVE_FOLDER_ID=1hRjlqG4OtOuTe4UI0CIQsCe0z2gfZalz
GOOGLE_APPLICATION_CREDENTIALS=google_creds.json

# AI модели
EMBEDDER_MODEL_NAME=mixedbread-ai/mxbai-embed-large-v1
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# LLM (stub = без ключей, для теста)
LLM_PROVIDER=stub
```

> **Важно:** в `docker-compose.yaml` хост БД — `db`, но с твоего компьютера (снаружи Docker) нужно `localhost`.

---

### Шаг 2 — Подними инфраструктуру

```powershell
docker-compose up -d
```

Проверь, что контейнеры запустились:

```powershell
docker ps
```

Должны быть: `db` (PostgreSQL), `redis`.

---

### Шаг 3 — Установи зависимости и инициализируй БД

```powershell
pip install -r requirements.txt
python scripts/init_db.py
```

---

### Шаг 4 — Запусти Chat API

```powershell
python -m uvicorn services.chat_service.main:app --reload --port 8001
```

Открой в браузере:
- `http://localhost:8001/` — статус сервиса
- `http://localhost:8001/docs` — интерактивный Swagger (тестировать эндпоинты здесь!)

---

### Шаг 5 — Запусти Ingest Worker (в отдельном окне PowerShell)

```powershell
python -m services.ingest_worker.main
```

---

## 🔌 API

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/ping` | Healthcheck |
| `POST` | `/chat/stream` | Потоковый ответ (SSE) |
| `DELETE` | `/chat/{user_id}/history` | Очистить историю пользователя |

### Пример запроса
```bash
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_42", "message": "Как получить академический отпуск?"}'
```

Ответ стримится в формате Server-Sent Events:
```
data: Для оформления ...
data:  академического ...
data: [DONE]
```

---

## 🧪 Тесты

```bash
# Unit-тесты SemanticChunker и TopicHelper (без БД и ML-модели)
python tests/test_indexer.py
```

---

## � Структура проекта

```
├── core/
│   ├── config.py          # настройки через pydantic-settings
│   ├── db.py              # SQLAlchemy async engine
│   └── models.py          # ORM: Document, Chunk (с pgvector)
├── services/
│   ├── api/
│   │   └── retrieval.py   # pgvector ANN-поиск
│   ├── chat_service/
│   │   ├── main.py        # FastAPI: /chat/stream (SSE)
│   │   ├── rag_logic.py   # Prompt Engineering + LLM streaming
│   │   └── history.py     # Redis: история диалогов
│   └── ingest_worker/
│       ├── main.py        # Основной pipeline индексации
│       ├── chunker.py     # SemanticChunker + Header Injection
│       ├── parsers.py     # HTML / PDF / DOCX парсеры
│       └── topic_helper.py# Хлебные крошки из иерархии топиков
├── scripts/
│   └── init_db.py         # Создание схемы БД
├── tests/
│   └── test_indexer.py    # 17 unit-тестов (без зависимостей)
├── docker-compose.yaml
└── requirements.txt
```

---

## 📈 Следующие шаги
- [ ] Reranker-модель (`cross-encoder/ms-marco-MiniLM`) для переранжирования чанков
- [ ] Hybrid search (BM25 + vector) для лучшего recall
- [ ] Загрузка документов через API (POST /documents)
- [ ] Telegram-бот как фронтенд для студентов
