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

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

client = None

EDIT_PROMPT = """Edit this image:
1. Remove ALL yellow text and typography from the image (especially bottom part)
2. Remove yellow horizontal lines and any logos/watermarks
3. Change ALL yellow arrows to forest green color (#228B22)
4. Restore the original background where text was removed
5. Keep everything else exactly the same

Return the edited image."""


def init_client():
    global client
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_API_KEY")
    
    if api_key:
        client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client ready (API key)")
    else:
        project_id = os.getenv("GOOGLE_PROJECT_ID", "tough-shard-479214-t2")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        key_base64 = os.getenv("GOOGLE_KEY_BASE64")
        if key_base64:
            import json
            key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
            creds_json = base64.b64decode(key_clean).decode('utf-8')
            creds_dict = json.loads(creds_json)
            
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials
            )
            logger.info("✅ Gemini client ready (Vertex AI + Service Account)")
        else:
            client = genai.Client(vertexai=True, project=project_id, location=location)
            logger.info("✅ Gemini client ready (Vertex AI default)")


def process_image(img_bytes: bytes) -> bytes:
    global client
    
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    image_part = types.Part.from_bytes(
        data=img_bytes,
        mime_type="image/png"
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
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-image-preview")
    logger.info(f"Using model: {model_name}")
    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    
    logger.info(f"Response received: {response}")
    
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                logger.info("Found inline_data in response")
                if hasattr(part.inline_data, 'data'):
                    return part.inline_data.data
                return part.inline_data
            if hasattr(part, 'image') and part.image:
                logger.info("Found image in response")
                return part.image
    
    logger.warning("No image found in response")
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📷 Отправь картинку\n\n"
        "Я уберу:\n"
        "• Жёлтый текст и типографику\n"
        "• Логотипы и полоски\n"
        "• Заменю жёлтые стрелки на зелёные"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Обрабатываю...")
    
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        result = await asyncio.to_thread(process_image, bytes(img_bytes))
        
        if result:
            await msg.delete()
            await update.message.reply_photo(result, caption="✅")
        else:
            await msg.edit_text("❌ Не удалось обработать")
    except Exception as e:
        logger.error(f"Error: {e}")
        await msg.edit_text(f"❌ {str(e)[:100]}")


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

    print("✅ Bot ready")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
