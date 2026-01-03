import logging
import os
import asyncio
import sys
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# Импорты Google GenAI SDK
from google import genai
from google.genai import types

# 1. Настройка логов
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

client = None

EDIT_PROMPT = """Edit this image:
1. Remove ALL yellow text and typography from the image (especially bottom 40%)
2. Remove the yellow horizontal lines above the text
3. Remove any logos or watermarks
4. Change ALL yellow arrows to forest green color (#228B22)
5. Restore the original background where text/elements were removed
6. Keep everything else exactly the same

Return the edited image."""

def init_client():
    global client
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
    
    if not api_key:
        logger.error("GOOGLE_CLOUD_API_KEY not set!")
        sys.exit(1)
    
    try:
        # Инициализация для AI Studio
        client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client ready (AI Studio Mode)")
    except Exception as e:
        logger.error(f"Client Init Error: {e}")
        sys.exit(1)

def process_image(img_bytes: bytes) -> bytes:
    global client
    
    try:
        # 1. Создаем объект Image через конструктор (правильный метод для SDK)
        my_image = types.Image(image_bytes=img_bytes)

        # 2. Оборачиваем в RawReferenceImage
        ref_image = types.RawReferenceImage(
            reference_id=1,
            reference_image=my_image
        )
        
        # 3. Конфиг (EditImageConfig)
        config = types.EditImageConfig(
            edit_mode="inpainting-insert",
            number_of_images=1,
            safety_filter_level="block_some",
            person_generation="allow_adult",
            include_rai_reason=True,
            output_mime_type="image/jpeg"
        )
        
        # 4. Вызов
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt=EDIT_PROMPT,
            reference_images=[ref_image],
            config=config
        )
        
        # 5. Результат
        if response.generated_images:
            return response.generated_images[0].image.image_bytes
            
    except Exception as e:
        logger.error(f"Imagen API Error: {e}")
        return None
    return None

# --- ГЛАВНЫЙ ФИКС СТАБИЛЬНОСТИ ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ловит ошибки и не дает боту упасть"""
    logger.error(f"⚠️ Telegram Error: {context.error}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍌 *Nano Banana Pro Bot*\n\n"
        "Отправь картинку — я уберу жёлтый текст и перекрашу стрелки в зелёный.",
        parse_mode="Markdown"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Обрабатываю (Imagen 3)...")
    
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        result = await asyncio.to_thread(process_image, bytes(img_bytes))
        
        if result:
            await msg.delete()
            await update.message.reply_photo(result, caption="✅ Готово")
        else:
            await msg.edit_text("❌ Ошибка обработки (Google вернул пустоту)")
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        await msg.edit_text("❌ Сбой бота")

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_TOKEN not set")
        sys.exit(1)

    init_client()

    # Настройки сети (HTTP 1.1 + тайм-ауты)
    request = HTTPXRequest(
        http_version="1.1",
        connection_pool_size=10,
        read_timeout=60.0,
        write_timeout=60.0,
        connect_timeout=60.0
    )
    
    app = Application.builder().token(token).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Добавляем обработчик ошибок, чтобы бот не крашился
    app.add_error_handler(error_handler)

    logger.info("🍌 Bot Started")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
