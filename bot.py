import logging
import os
import asyncio
import sys
import json
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# --- ИМПОРТЫ VERTEX AI (СТАБИЛЬНЫЙ SDK) ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmCategory, HarmBlockThreshold
from google.oauth2 import service_account

# Настройка логов
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

model = None

EDIT_PROMPT = """
Task: Generate a modified version of this image.
Changes required:
1. Remove ALL yellow text and typography.
2. Remove yellow lines.
3. Change yellow arrows to forest green (#228B22).
4. Remove logos/watermarks.
5. Keep the background and other elements exactly as they are.
Output: A high-quality image.
"""

def init_vertex():
    global model
    key_base64 = os.getenv("GOOGLE_KEY_BASE64")
    project_id = os.getenv("GOOGLE_PROJECT_ID", "tough-shard-479214-t2")
    location = os.getenv("GOOGLE_LOCATION", "us-central1")

    try:
        if not key_base64:
            logger.error("❌ GOOGLE_KEY_BASE64 not found!")
            sys.exit(1)

        # 1. Декодируем ключ
        key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
        creds_json = base64.b64decode(key_clean).decode('utf-8')
        creds_dict = json.loads(creds_json)
        
        # 2. Создаем Credentials с правами
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # 3. Инициализируем Vertex AI (Стабильный SDK)
        vertexai.init(
            project=creds_dict.get("project_id", project_id),
            location=location,
            credentials=credentials
        )
        
        # 4. Загружаем модель
        # Используем gemini-2.0-flash-exp (она умеет рисовать)
        model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info("✅ Vertex AI Initialized (Standard SDK)")

    except Exception as e:
        logger.error(f"Vertex Init Error: {e}")
        sys.exit(1)

def process_image(img_bytes: bytes) -> bytes:
    global model
    try:
        # 1. Подготовка контента
        image_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")
        
        # 2. Конфигурация генерации (как словарь, чтобы обойти типизацию)
        # response_modalities=["IMAGE"] заставляет Gemini 2.0 рисовать
        generation_config = {
            "temperature": 1.0,
            "max_output_tokens": 8192,
            "response_modalities": ["IMAGE"],
        }

        # 3. Настройки безопасности (отключаем всё)
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]

        # 4. Генерация
        response = model.generate_content(
            [image_part, EDIT_PROMPT],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # 5. Извлечение картинки из ответа Vertex AI
        # У Vertex AI картинки лежат в parts[].inline_data
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # В Vertex SDK это атрибут _raw_part или просто data в зависимости от версии
                # Но стандартный метод - проверить inline_data
                if part.inline_data:
                    return part.inline_data.data
                
    except Exception as e:
        logger.error(f"Vertex Gen Error: {e}")
        # Если придет ошибка валидации здесь - Vertex SDK обычно просто пишет WARNING, а не крашится
        return None
    return None

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"⚠️ Telegram Error: {context.error}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍌 *Nano Banana Pro (Vertex)*\n\n"
        "Система переведена на стабильный Vertex SDK. Жду фото.",
        parse_mode="Markdown"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Генерирую (Vertex)...")
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        result = await asyncio.to_thread(process_image, bytes(img_bytes))
        
        if result:
            await msg.delete()
            await update.message.reply_photo(result, caption="✅ Готово")
        else:
            await msg.edit_text("❌ Ошибка (проверьте логи)")
    except Exception as e:
        logger.error(f"Bot Error: {e}")
        await msg.edit_text("❌ Сбой")

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        sys.exit(1)
    
    init_vertex()
    
    request = HTTPXRequest(http_version="1.1", connection_pool_size=10, read_timeout=60)
    app = Application.builder().token(token).request(request).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(error_handler)

    logger.info("🍌 Bot Started (Vertex Mode)")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
