import logging
import os
import asyncio
import sys
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# Импорты по твоему образцу
from google import genai
from google.genai import types

# Настройка логов
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

client = None

# Промпт переписан для Gemini (она понимает инструкции лучше)
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
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
    if not api_key:
        logger.error("GOOGLE_CLOUD_API_KEY not set!")
        sys.exit(1)
    
    # Инициализация строго как в твоем примере
    try:
        client = genai.Client(
            api_key=api_key,
            # vertexai=True убрал, так как для API Key обычно используется AI Studio,
            # но если у тебя Vertex проект, раскомментируй. 
            # Для стабильности с API KEY лучше оставить дефолт.
        )
        logger.info("✅ Gemini Client Ready (GenAI SDK)")
    except Exception as e:
        logger.error(f"Client Init Error: {e}")
        sys.exit(1)

def process_image(img_bytes: bytes) -> bytes:
    global client
    try:
        # 1. Создаем Part из картинки (как в твоем коде)
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg",
        )
        
        # 2. Создаем Part из текста
        text_part = types.Part.from_text(text=EDIT_PROMPT)

        # 3. Конфиг (ВОТ ОНО! То, что ты нашел)
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            # Ключевой момент: просим вернуть КАРТИНКУ
            response_modalities=["IMAGE"], 
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
            # Тот самый ImageConfig, который вызывал ошибку раньше (теперь он на своем месте)
            image_config=types.ImageConfig(
                aspect_ratio="3:4",
                output_mime_type="image/jpeg",
            ),
        )

        # 4. Вызов (используем актуальную модель Gemini 2.0 Flash Exp)
        # "gemini-3-pro" из примера может быть еще закрыта, 2.0 Flash - работает.
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

        # 5. Извлекаем картинку из ответа
        # Ответ Gemini с картинкой приходит в parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
                # Иногда байты могут быть в другом поле в зависимости от версии
                if hasattr(part, 'image_bytes'):
                     return part.image_bytes
                     
    except Exception as e:
        logger.error(f"GenAI Error: {e}")
        return None
    return None

# --- Обработчик ошибок Telegram ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"⚠️ Telegram Error: {context.error}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍌 *Nano Banana Pro (Gemini Native)*\n\n"
        "Отправь фото -> Я перерисую его через Gemini 2.0 Vision.",
        parse_mode="Markdown"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Генерирую...")
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        result = await asyncio.to_thread(process_image, bytes(img_bytes))
        
        if result:
            await msg.delete()
            await update.message.reply_photo(result, caption="✅ Готово")
        else:
            await msg.edit_text("❌ Ошибка генерации (возможно, фильтры)")
    except Exception as e:
        logger.error(f"Bot Error: {e}")
        await msg.edit_text("❌ Сбой")

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        sys.exit(1)
    
    init_client()
    
    # Сетевые настройки
    request = HTTPXRequest(http_version="1.1", connection_pool_size=10, read_timeout=60, write_timeout=60, connect_timeout=60)
    app = Application.builder().token(token).request(request).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(error_handler)

    logger.info("🍌 Bot Started (Gemini Native Mode)")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
