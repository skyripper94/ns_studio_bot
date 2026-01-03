import logging
import os
import asyncio
import sys
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

from google import genai
from google.genai import types

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
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
    
    client = genai.Client(
        vertexai=True,
        api_key=api_key,
    )
    logger.info("✅ Gemini client ready")


def process_image(img_bytes: bytes) -> bytes:
    global client
    
    image_part = types.Part.from_bytes(
        data=img_bytes,
        mime_type="image/jpeg",
    )
    text_part = types.Part.from_text(text=EDIT_PROMPT)
    
    contents = [
        types.Content(
            role="user",
            parts=[image_part, text_part]
        )
    ]
    
    config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=32768,
        response_modalities=["TEXT", "IMAGE"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        image_config=types.ImageConfig(
            aspect_ratio="3:4",
            image_size="1K",
            output_mime_type="image/png",
        ),
    )
    
    model = "gemini-3-pro-image-preview"
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                logger.info("✅ Image received from API")
                return base64.standard_b64decode(part.inline_data.data)
    
    logger.warning("No image in response")
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍌 *Nano Banana Pro Bot*\n\n"
        "Отправь картинку — я уберу:\n"
        "• Жёлтый текст и типографику\n"
        "• Логотипы и полоски\n"
        "• Заменю жёлтые стрелки на зелёные",
        parse_mode="Markdown"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Обрабатываю через Nano Banana Pro...")
    
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        result = await asyncio.to_thread(process_image, bytes(img_bytes))
        
        if result:
            await msg.delete()
            await update.message.reply_photo(result, caption="✅ Готово")
        else:
            await msg.edit_text("❌ Не удалось обработать")
    except Exception as e:
        logger.error(f"Error: {e}")
        await msg.edit_text(f"❌ Ошибка: {str(e)[:200]}")


def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_TOKEN not set")
        sys.exit(1)

    init_client()

    request = HTTPXRequest(http_version="1.1", read_timeout=120, write_timeout=120, connect_timeout=30)
    app = Application.builder().token(token).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("🍌 Nano Banana Pro Bot Started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
