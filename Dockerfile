## Use Python 3.10 slim
FROM python:3.10-slim

# Install system dependencies including font support
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

# Set working directory
WORKDIR /app

# Create fonts directory
RUN mkdir -p /app/fonts

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY telegram_bot.py .
COPY lama_integration.py .

# Copy font file (user must add this manually to GitHub)
# The font file must be at: fonts/WaffleSoft.otf in repository
COPY fonts/WaffleSoft.otf /app/fonts/ 2>/dev/null || echo "Warning: Font file not found, will use fallback"

# Update font cache
RUN fc-cache -f -v

# Create temp directory
RUN mkdir -p /tmp/bot_images

# Run bot
CMD ["python", "telegram_bot.py"]
