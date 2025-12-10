# Базовый образ с Python
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота
COPY telegram_bot.py .
COPY lama_integration.py .

# Создаем директории
RUN mkdir -p /tmp/bot_images /app/models

# Скачиваем LaMa модель (опционально, можно скачать при первом запуске)
 RUN wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
     unzip big-lama.zip -d /app/models/ && \
     rm big-lama.zip

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "-u", "telegram_bot.py"]
