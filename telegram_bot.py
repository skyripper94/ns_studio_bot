"""
Telegram Bot with 2 main modes:
1. REMOVE - Only remove text (existing)
2. FULL - Full workflow with 3 submodes (1/2/3)
"""

import os
import logging
from io import BytesIO

import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv

from lama_integration import flux_kontext_inpaint, process_full_workflow

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TEMP_DIR = '/tmp/bot_images'
os.makedirs(TEMP_DIR, exist_ok=True)

# User state storage
user_states = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    user_id = update.effective_user.id
    user_states[user_id] = {'mode': None, 'submode': None}
    
    keyboard = [
        [
            InlineKeyboardButton("🗑️ УДАЛИТЬ ТЕКСТ", callback_data="mode_remove"),
            InlineKeyboardButton("🔄 FULL WORKFLOW", callback_data="mode_full")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "👋 **Бот для работы с изображениями**\n\n"
        "**🗑️ УДАЛИТЬ ТЕКСТ:**\n"
        "Только удаление текста и градиента (FLUX Kontext Pro)\n\n"
        "**🔄 FULL WORKFLOW:**\n"
        "OCR → Удаление → Перевод → Нанесение текста\n"
        "3 режима: Лого / Текст / Контент\n\n"
        "Выберите режим:",
        reply_markup=reply_markup
    )


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle mode selection"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        user_states[user_id] = {'mode': None, 'submode': None}
    
    if query.data == "mode_remove":
        user_states[user_id]['mode'] = 'remove'
        await query.edit_message_text(
            "✅ **Режим: УДАЛИТЬ ТЕКСТ**\n\n"
            "Просто отправьте изображение.\n"
            "Бот удалит текст и градиент (35% снизу)."
        )
    
    elif query.data == "mode_full":
        user_states[user_id]['mode'] = 'full'
        
        keyboard = [
            [
                InlineKeyboardButton("1️⃣ ЛОГО", callback_data="submode_1"),
                InlineKeyboardButton("2️⃣ ТЕКСТ", callback_data="submode_2"),
                InlineKeyboardButton("3️⃣ КОНТЕНТ", callback_data="submode_3")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "✅ **Режим: FULL WORKFLOW**\n\n"
            "Выберите подрежим:\n\n"
            "**1️⃣ ЛОГО** - Лого + полоски + заголовок\n"
            "**2️⃣ ТЕКСТ** - Только заголовок\n"
            "**3️⃣ КОНТЕНТ** - Заголовок + подзаголовок\n\n"
            "После выбора отправьте изображение.",
            reply_markup=reply_markup
        )
    
    elif query.data.startswith("submode_"):
        submode = int(query.data.split("_")[1])
        user_states[user_id]['submode'] = submode
        
        mode_names = {
            1: "ЛОГО (лого + заголовок)",
            2: "ТЕКСТ (только заголовок)",
            3: "КОНТЕНТ (заголовок + подзаголовок)"
        }
        
        await query.edit_message_text(
            f"✅ **Выбран режим {submode}: {mode_names[submode]}**\n\n"
            f"Теперь отправьте изображение для обработки.\n\n"
            f"Бот выполнит:\n"
            f"1. OCR (Google Vision)\n"
            f"2. Удаление текста (FLUX)\n"
            f"3. Перевод (OpenAI)\n"
            f"4. Нанесение русского текста"
        )


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process received image"""
    user_id = update.effective_user.id
    
    # Check state
    if user_id not in user_states or user_states[user_id]['mode'] is None:
        await update.message.reply_text(
            "⚠️ Сначала выберите режим командой /start"
        )
        return
    
    mode = user_states[user_id]['mode']
    submode = user_states[user_id].get('submode')
    
    if mode == 'full' and submode is None:
        await update.message.reply_text(
            "⚠️ Сначала выберите подрежим (1/2/3)"
        )
        return
    
    try:
        # Download image
        photo = await update.message.photo[-1].get_file()
        image_bytes = await photo.download_as_bytearray()
        
        logger.info(f"✅ Image from user {user_id}, mode: {mode}, submode: {submode}")
        
        # Convert to OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mode == 'remove':
            # REMOVE MODE: Just remove text
            status_msg = await update.message.reply_text("⏳ Удаление текста...")
            
            # Create mask for bottom 35%
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask_start = int(height * 0.65)
            mask[mask_start:, :] = 255
            
            # Remove text
            result = flux_kontext_inpaint(image, mask)
            
            # Send result
            success, buffer = cv2.imencode('.png', result)
            if success:
                await update.message.reply_photo(
                    photo=BytesIO(buffer.tobytes()),
                    caption="✅ **Текст удалён!**\n🎨 FLUX Kontext Pro"
                )
                await status_msg.delete()
        
        elif mode == 'full':
            # FULL MODE: Complete workflow
            status_msg = await update.message.reply_text(
                f"⏳ **Обработка (режим {submode})...**\n\n"
                f"1. OCR...\n"
                f"2. Удаление...\n"
                f"3. Перевод...\n"
                f"4. Нанесение текста..."
            )
            
            # Process with full workflow
            result, ocr_data = process_full_workflow(image, submode)
            
            # Send result
            success, buffer = cv2.imencode('.png', result)
            if success:
                mode_names = {
                    1: "ЛОГО",
                    2: "ТЕКСТ",
                    3: "КОНТЕНТ"
                }
                
                await update.message.reply_photo(
                    photo=BytesIO(buffer.tobytes()),
                    caption=(
                        f"✅ **Готово! (Режим {submode}: {mode_names[submode]})**\n\n"
                        f"📝 Распознано текста: {len(ocr_data.get('lines', []))} строк\n"
                        f"🌐 Переведено и адаптировано\n"
                        f"🎨 FLUX Kontext Pro + OpenAI GPT-4"
                    )
                )
                await status_msg.delete()
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


def main():
    """Start bot"""
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not set!")
        return
    
    logger.info("🚀 Starting bot...")
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(mode_callback))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))
    
    logger.info("✅ Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
