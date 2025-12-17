# telegram_bot.py

"""
Telegram бот с 2 основными режимами:
1. УДАЛИТЬ - только удаление текста (существующий функционал)
2. FULL - полный workflow с 3 подрежимами (1/2/3)
"""

import os
import logging
from io import BytesIO

import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv

from lama_integration import flux_kontext_inpaint, process_full_workflow, MASK_BOTTOM_PERCENT

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Временная директория для изображений
TEMP_DIR = '/tmp/bot_images'
os.makedirs(TEMP_DIR, exist_ok=True)

# Хранилище состояний пользователей
user_states = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /start - показывает меню выбора режима
    """
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
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка выбора режима через inline кнопки
    """
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    # Инициализация состояния если нужно
    if user_id not in user_states:
        user_states[user_id] = {'mode': None, 'submode': None}
    
    if query.data == "mode_remove":
        # Режим "только удаление"
        user_states[user_id]['mode'] = 'remove'
        await query.edit_message_text(
            "✅ **Режим: УДАЛИТЬ ТЕКСТ**\n\n"
            "Просто отправьте изображение.\n"
            f"Бот удалит текст и градиент ({MASK_BOTTOM_PERCENT}% снизу).",
            parse_mode='Markdown'
        )
    
    elif query.data == "mode_full":
        # Режим "полный workflow" - показываем подрежимы
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
            "**1️⃣ ЛОГО** - Лого + линии + заголовок\n"
            "**2️⃣ ТЕКСТ** - Только заголовок\n"
            "**3️⃣ КОНТЕНТ** - Заголовок + подзаголовок\n\n"
            "После выбора отправьте изображение.",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    elif query.data.startswith("submode_"):
        # Выбор подрежима (1, 2 или 3)
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
            f"4. Нанесение русского текста",
            parse_mode='Markdown'
        )


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка полученного изображения
    """
    user_id = update.effective_user.id
    
    # Проверка что выбран режим
    if user_id not in user_states or user_states[user_id]['mode'] is None:
        await update.message.reply_text(
            "⚠️ Сначала выберите режим командой /start"
        )
        return
    
    mode = user_states[user_id]['mode']
    submode = user_states[user_id].get('submode')
    
    # Для full режима нужен подрежим
    if mode == 'full' and submode is None:
        await update.message.reply_text(
            "⚠️ Сначала выберите подрежим (1/2/3)"
        )
        return
    
    try:
        # Скачивание изображения
        photo = await update.message.photo[-1].get_file()
        image_bytes = await photo.download_as_bytearray()
        
        logger.info(f"✅ Изображение от пользователя {user_id}, режим: {mode}, подрежим: {submode}")
        
        # Конвертация в OpenCV формат
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mode == 'remove':
            # РЕЖИМ УДАЛЕНИЯ: только убираем текст
            status_msg = await update.message.reply_text("⏳ Удаление текста...")
            
            # Создаем маску для нижних 35%
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask_start = int(height * (1 - MASK_BOTTOM_PERCENT / 100))
            mask[mask_start:, :] = 255
            
            # Удаляем текст через FLUX
            result = flux_kontext_inpaint(image, mask)
            
            # Отправка результата
            success, buffer = cv2.imencode('.png', result)
            if success:
                await update.message.reply_photo(
                    photo=BytesIO(buffer.tobytes()),
                    caption="✅ **Текст удалён!**\n🎨 FLUX Kontext Pro",
                    parse_mode='Markdown'
                )
                await status_msg.delete()
        
        elif mode == 'full':
            # ПОЛНЫЙ РЕЖИМ: весь workflow
            status_msg = await update.message.reply_text(
                f"⏳ **Обработка (режим {submode})...**\n\n"
                f"1. OCR...\n"
                f"2. Удаление...\n"
                f"3. Перевод...\n"
                f"4. Нанесение текста...",
                parse_mode='Markdown'
            )
            
            # Обработка через полный workflow
            result, ocr_data = process_full_workflow(image, submode)
            
            # Отправка результата
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
                    ),
                    parse_mode='Markdown'
                )
                await status_msg.delete()
    
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


def main():
    """
    Запуск бота
    """
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN не установлен!")
        return
    
    logger.info("🚀 Запуск бота...")
    
    # Создание приложения
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(mode_callback))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))
    
    logger.info("✅ Бот запущен!")
    
    # Запуск polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
