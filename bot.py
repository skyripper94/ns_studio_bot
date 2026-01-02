import logging
import os
import asyncio
import sys
import traceback
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)

# Проверка наличия сервисов Google
try:
    from google_services import GoogleBrain
except ImportError:
    print("CRITICAL: Файл google_services.py не найден!")
    sys.exit(1)

# --- НАСТРОЙКА ЛОГОВ (ЧИСТАЯ) ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Заглушаем шумные библиотеки (чтобы токен не лез в логи)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

# Состояния диалога
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

# Инициализация Мозга
try:
    brain = GoogleBrain()
except Exception as e:
    logger.critical(f"Ошибка старта GoogleBrain: {e}")
    sys.exit(1)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit=False):
    """Отправляет или обновляет главное меню"""
    text = (
        "🚀 **Nano Banana AI: Контент-Завод**\n\n"
        "Я создаю вирусные карусели и чищу фото.\n"
        "Выбери задачу:"
    )
    keyboard = [
        [InlineKeyboardButton("🎡 Создать Карусель (Gemini + Imagen)", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Удалить текст с фото", callback_data='mode_cleaner')],
        [InlineKeyboardButton("🔄 Обновить статус", callback_data='check_status')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if edit and update.callback_query:
        # Если можно отредактировать старое сообщение
        try:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
        except:
            await update.callback_query.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        # Отправляем новое
        chat_id = update.effective_chat.id
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode="Markdown")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ловит ошибки, чтобы бот не падал"""
    logger.error("Exception while handling an update:", exc_info=context.error)
    if update and isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("⚠️ Произошла ошибка. Попробуйте нажать /start")

# --- ХЕНДЛЕРЫ: СТАРТ И МЕНЮ ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_main_menu(update, context)
    return ConversationHandler.END

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await send_main_menu(update, context, edit=True)
    return ConversationHandler.END

# --- ХЕНДЛЕРЫ: ОЧИСТКА ФОТО ---

async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "📷 **Режим очистки**\n\n"
        "Пришли мне фото, и я удалю текст с нижней части изображения.\n"
        "Поддерживаются: JPG, PNG.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_to_main")]])
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return

    # Защита: сообщаем пользователю, что процесс идет
    status_msg = await update.message.reply_text("⏳ Скачиваю и обрабатываю фото...")

    try:
        photo_file = await update.message.photo[-1].get_file()
        img_bytes = await photo_file.download_as_bytearray()
        
        await status_msg.edit_text("🧹 Nano Banana стирает текст (Imagen 3)...")
        
        # Запускаем в отдельном потоке, чтобы не блокировать бота
        cleaned_bytes = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
        
        if cleaned_bytes:
            await status_msg.delete()
            await update.message.reply_photo(cleaned_bytes, caption="✅ Готово! Текст удален.")
        else:
            await status_msg.edit_text("❌ Не удалось очистить фото. Попробуйте другое.")
            
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        await status_msg.edit_text("⚠️ Ошибка обработки.")
    
    # Возвращаем меню
    await send_main_menu(update, context)

# --- ХЕНДЛЕРЫ: КАРУСЕЛИ (СЛОЖНАЯ ЛОГИКА) ---

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("🧠 Gemini анализирует тренды и ищет темы...")
    
    # Генерация тем
    topics = await asyncio.to_thread(brain.generate_topics)
    
    keyboard = []
    for topic in topics:
        # Обрезаем callback_data до 60 символов, иначе Телеграм ругается
        safe_data = f"ts_{topic[:40]}"
        keyboard.append([InlineKeyboardButton(topic, callback_data=safe_data)])
    
    keyboard.append([InlineKeyboardButton("✍️ Написать свою тему", callback_data="topic_custom")])
    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
    
    await query.edit_message_text(
        "🔥 **Выбери тему для карусели:**\n"
        "Я подобрал актуальные темы на основе трендов.", 
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
    return CHOOSING_MODE

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "topic_custom":
        await query.edit_message_text(
            "✍️ Введи тему карусели (например: *Биткоин против Золота*):", 
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]),
            parse_mode="Markdown"
        )
        return ENTERING_TOPIC
    
    # Пытаемся восстановить полное название темы из кнопки
    chosen_topic = "Unknown Topic"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == data:
                chosen_topic = btn.text
                break
    
    return await generate_plan_step(update, context, chosen_topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text
    return await generate_plan_step(update, context, topic)

async def generate_plan_step(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    # Определяем, куда отвечать (на кнопку или на сообщение)
    msg_func = update.message.reply_text if update.message else update.callback_query.message.reply_text
    
    status_msg = await msg_func(f"📝 Пишу сценарий (минимум слов, только факты) для: **{topic}**...", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    
    if not plan:
        await status_msg.edit_text("❌ Gemini не смогла создать план. Попробуй другую тему.")
        return ConversationHandler.END

    context.user_data['current_plan'] = plan
    
    # Красивое превью
    preview_text = f"📋 **Сценарий:** {topic}\n\n"
    for slide in plan:
        num = slide.get('slide_number', '-')
        # Показываем только текст слайда
        caption = slide.get('ru_caption', '...') 
        preview_text += f"🔹 **Слайд {num}:** {caption}\n"
        
    keyboard = [
        [InlineKeyboardButton("🚀 Генерировать (Imagen 3)", callback_data="confirm_gen")],
        [InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]
    ]
    
    await status_msg.edit_text(preview_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    plan = context.user_data.get('current_plan')
    total = len(plan)
    
    await query.edit_message_text(
        f"🎨 **Начинаю производство!**\n"
        f"Всего слайдов: {total}.\n"
        f"⚠️ Делаю паузы между слайдами, чтобы Google не ругался. Ждите.",
        parse_mode="Markdown"
    )
    
    for i, slide in enumerate(plan):
        num = i + 1
        prompt = slide.get('image_prompt')
        caption = slide.get('ru_caption')
        
        # Информируем пользователя
        status = await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=f"🎨 Рисую слайд {num}/{total}..."
        )
        
        # ГЕНЕРАЦИЯ
        img_bytes = await asyncio.to_thread(brain.generate_image, prompt)
        
        if img_bytes:
            await status.delete()
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_bytes,
                caption=f"{caption}\n\n#{num}", # Короткая подпись
            )
        else:
            await status.edit_text(f"⚠️ Слайд {num} пропущен (ошибка генерации).")

        # === ВАЖНАЯ ЗАЩИТА ОТ БАНА ===
        if num < total:
            await asyncio.sleep(10) # 10 секунд отдыха между картинками
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ **Карусель готова!**", parse_mode="Markdown")
    await send_main_menu(update, context)
    return ConversationHandler.END

# --- ЗАПУСК БОТА ---

def main():
    # 1. Получаем и чистим токен
    raw_token = os.getenv("TELEGRAM_TOKEN", "")
    token = raw_token.strip().replace('"', '').replace("'", "")
    
    if not token:
        print("❌ CRITICAL: Переменная TELEGRAM_TOKEN пустая!")
        sys.exit(1)
        
    print(f"✅ Бот запускается... Токен OK (длина {len(token)})")

    # 2. Строим приложение
    application = Application.builder().token(token).build()
    
    # 3. Добавляем обработчик ошибок
    application.add_error_handler(error_handler)

    # 4. Сценарий карусели
    carousel_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(handle_topic_selection, pattern='^ts_')],
            ENTERING_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_topic_input)],
            CONFIRMING_PLAN: [CallbackQueryHandler(run_final_generation, pattern='^confirm_gen$')]
        },
        fallbacks=[
            CallbackQueryHandler(back_to_main, pattern='^back_to_main$'),
            CommandHandler('start', start)
        ]
    )

    # 5. Регистрируем хендлеры
    application.add_handler(CommandHandler("start", start))
    application.add_handler(carousel_handler)
    
    # Обработчики кнопок меню
    application.add_handler(CallbackQueryHandler(mode_cleaner_start, pattern='^mode_cleaner$'))
    application.add_handler(CallbackQueryHandler(back_to_main, pattern='^back_to_main$'))
    application.add_handler(CallbackQueryHandler(start, pattern='^check_status$'))
    
    # Обработчик фото (для очистки)
    application.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    # 6. Поехали
    print("🚀 Polling started...")
    application.run_polling()

if __name__ == '__main__':
    main()
