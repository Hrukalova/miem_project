FROM python:3.11-slim

# Устанавливаем системные зависимости для сборки (если нужны для библиотек)
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта в контейнер
COPY . .

# Указываем переменную окружения, чтобы Python корректно видел модули
ENV PYTHONPATH=/app

# По умолчанию запускаем Chat Service (API)
CMD ["uvicorn", "services.chat_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
