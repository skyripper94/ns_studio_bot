# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота
COPY telegram_bot.py .
COPY lama_integration.py .

# Создаём директории для изображений и модели
RUN mkdir -p /tmp/bot_images /app/models

# Загружаем предобученную модель LaMa
RUN wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
    unzip big-lama.zip -d /app/models/ && \
    rm big-lama.zip

# Запускаем бота
CMD ["python", "-u", "telegram_bot.py"]
