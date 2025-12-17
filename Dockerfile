# Dockerfile

# Базовый образ Python 3.10
FROM python:3.10-slim

# Установка системных зависимостей + шрифты
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Создание директории для шрифтов
RUN mkdir -p /app/fonts

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода бота
COPY telegram_bot.py .
COPY lama_integration.py .

# Копирование шрифта (поместите fonts/WaffleSoft.otf в репозиторий)
COPY fonts/ /app/fonts/

# Обновление кэша шрифтов
RUN fc-cache -f -v

# Создание временной директории
RUN mkdir -p /tmp/bot_images

# Запуск бота
CMD ["python", "telegram_bot.py"]
