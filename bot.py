import logging
import os
import asyncio
import sys
import json
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# Импорты Google
from google import genai
from google.genai import types
from google.oauth2 import service_account

# Настройка логов
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

client = None

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

def init_client():
    global client
    # 1. Ищем секретный ключ
    key_base64 = os.getenv("GOOGLE_KEY_BASE64")
    
    # Резерв
    project_id = os.getenv("GOOGLE_PROJECT_ID", "tough-shard-479214-t2")
    location = os.getenv("GOOGLE_LOCATION", "us-central1")

    try:
        if key_base64:
            # Декодируем
            key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
            creds_json = base64.b64decode(key_clean).decode('utf-8')
            creds_dict = json.loads(creds_json)
            
            # --- ГЛАВНЫЙ ФИКС (Scope) ---
            # Мы явно говорим Google: "Дай этому ключу доступ к облачной платформе"
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict, 
                scopes=scopes  # <--- ВОТ ЧТО ИСПРАВИТ ОШИБКУ INVALID_SCOPE
            )
            
            # Инициализируем клиент
            client = genai.Client(
                vertexai=True,
                project=creds_dict.get("project_id", project_id),
                location=location,
                credentials=credentials
            )
            logger.info("✅ Gemini Client Ready (Vertex AI Mode + Scopes)")
            
        else:
            # Fallback на API Key (но он скорее всего не сработает для картинок)
            api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
            if not api_key:
                logger.error("❌ Auth Error: No GOOGLE_KEY_BASE64 found")
                sys.exit(1)
            client = genai.Client(api_key=api_key)
            logger.warning("⚠️ Gemini Client Ready (API Key Mode)")

    except Exception as e:
        logger.error(f"Client Init Error: {e}")
        sys.exit(1)

def process_image(img_bytes: bytes) -> bytes:
    global client
    try:
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg",
        )
        text_part = types.Part.from_text(text=EDIT_PROMPT)

        # Конфиг для Gemini 2.0
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["IMAGE"], 
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", 
            contents=[
                types.Content(
                    role="user",
                    parts=[image_part, text_part]
                )
            ],
            config=generate_content_config,
        )

        # Извлекаем результат
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
                if hasattr(part, 'image_bytes'):
                     return part.image_bytes
                     
    except Exception as e:
        logger.error(f"GenAI Error: {e}")
        return None
    return None

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"⚠️ Telegram Error: {context.error}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍌 *Nano Banana Pro (Vertex AI)*\n\n"
        "Система авторизована. Отправляй фото!",
        parse_mode="Markdown"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Генерирую (Vertex AI)...")
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        result = await asyncio.to_thread(process_image, bytes(img_bytes))
        
        if result:
            await msg.delete()
            await update.message.reply_photo(result, caption="✅ Готово")
        else:
            await msg.edit_text("❌ Ошибка генерации (Смотри логи)")
    except Exception as e:
        logger.error(f"Bot Error: {e}")
        await msg.edit_text("❌ Сбой")

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        sys.exit(1)
    
    init_client()
    
    # Сеть
    request = HTTPXRequest(http_version="1.1", connection_pool_size=10, read_timeout=60, write_timeout=60, connect_timeout=60)
    app = Application.builder().token(token).request(request).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(error_handler)

    logger.info("🍌 Bot Started (Vertex Mode)")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
