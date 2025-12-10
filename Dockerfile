# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
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

# Создаём директорию для временных изображений
RUN mkdir -p /tmp/bot_images

# Запускаем бота
CMD ["python", "-u", "telegram_bot.py"]
