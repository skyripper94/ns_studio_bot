"""
Telegram Bot with Full Workflow
OCR -> Remove -> Translate -> Add Text
"""

import os
import logging
from io import BytesIO

import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv

from lama_integration import process_image_full_workflow, flux_kontext_inpaint, recognize_text, create_text_mask

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TEMP_DIR = '/tmp/bot_images'
os.makedirs(TEMP_DIR, exist_ok=True)

user_modes = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = {
        'mode': 'full_workflow',
        'gradient_percent': 40
    }
    
    await update.message.reply_text(
        "👋 Привет! Я бот для работы с текстом на изображениях.\n\n"
        "📋 **Режимы:**\n\n"
        "1️⃣ **FULL WORKFLOW** (по умолчанию):\n"
        "   • OCR → Удаление → Перевод → Нанесение\n\n"
        "2️⃣ **ONLY REMOVE**:\n"
        "   • Только удаление текста\n\n"
        "📸 Отправь изображение!\n\n"
        "⚙️ /mode - Выбрать режим\n"
        "/gradient <30-50> - Область градиента"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 **Справка:**\n\n"
        "/mode - Режим работы\n"
        "/gradient 40 - Область градиента (30-50%)\n\n"
        "**FULL WORKFLOW:**\n"
        "OCR → Remove → Translate EN→RU → Add\n\n"
        "**ONLY REMOVE:**\n"
        "Только удаление\n\n"
        "Отправь изображение для обработки!"
    )

async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_modes:
        user_modes[user_id] = {'mode': 'full_workflow', 'gradient_percent': 40}
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 FULL", callback_data="mode_full"),
            InlineKeyboardButton("🗑️ REMOVE", callback_data="mode_remove")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    current = user_modes[user_id]
    await update.message.reply_text(
        f"⚙️ Режим: `{current['mode']}`\n"
        f"Градиент: `{current['gradient_percent']}%`",
        reply_markup=reply_markup
    )

async def gradient_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_modes:
        user_modes[user_id] = {'mode': 'full_workflow', 'gradient_percent': 40}
    
    if not context.args:
        await update.message.reply_text(
            f"📐 Текущее: `{user_modes[user_id]['gradient_percent']}%`\n"
            f"Использование: `/gradient 40`"
        )
        return
    
    try:
        percent = int(context.args[0])
        if 30 <= percent <= 50:
            user_modes[user_id]['gradient_percent'] = percent
            await update.message.reply_text(f"✅ Установлено: `{percent}%`")
        else:
            await update.message.reply_text("❌ Диапазон: 30-50")
    except ValueError:
        await update.message.reply_text("❌ Неверный формат")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if user_id not in user_modes:
        user_modes[user_id] = {'mode': 'full_workflow', 'gradient_percent': 40}
    
    if query.data == "mode_full":
        user_modes[user_id]['mode'] = 'full_workflow'
        await query.edit_message_text("✅ Режим: **FULL WORKFLOW**\nOCR → Remove → Translate → Add")
    elif query.data == "mode_remove":
        user_modes[user_id]['mode'] = 'only_remove'
        await query.edit_message_text("✅ Режим: **ONLY REMOVE**\nТолько удаление")

async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_modes:
        user_modes[user_id] = {'mode': 'full_workflow', 'gradient_percent': 40}
    
    mode = user_modes[user_id]['mode']
    gradient_percent = user_modes[user_id]['gradient_percent']
    
    try:
        photo = await update.message.photo[-1].get_file()
        image_bytes = await photo.download_as_bytearray()
        
        logger.info(f"✅ Image from user {user_id}")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        status_msg = await update.message.reply_text(
            f"⏳ Обработка...\nРежим: `{mode}`\nГрадиент: `{gradient_percent}%`"
        )
        
        if mode == 'full_workflow':
            result, text_data = process_image_full_workflow(
                image, 
                gradient_percent=gradient_percent,
                translate=True
            )
            
            success, buffer = cv2.imencode('.png', result)
            if success:
                await update.message.reply_photo(
                    photo=BytesIO(buffer.tobytes()),
                    caption=(
                        f"✅ **Готово!**\n\n"
                        f"📝 Распознано: {len(text_data)}\n"
                        f"🌐 Переведено: {len([t for t in text_data if 'translated_text' in t])}\n"
                        f"🎨 FLUX Kontext Pro"
                    )
                )
                await status_msg.delete()
        
        else:
            text_data = recognize_text(image)
            mask = create_text_mask(image, text_data, gradient_percent)
            result = flux_kontext_inpaint(image, mask)
            
            success, buffer = cv2.imencode('.png', result)
            if success:
                await update.message.reply_photo(
                    photo=BytesIO(buffer.tobytes()),
                    caption=(
                        f"✅ **Текст удалён!**\n\n"
                        f"📝 Распознано: {len(text_data)}\n"
                        f"🎨 FLUX Kontext Pro"
                    )
                )
                await status_msg.delete()
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not set!")
        return
    
    logger.info("🚀 Starting bot...")
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("gradient", gradient_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))
    
    logger.info("✅ Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
