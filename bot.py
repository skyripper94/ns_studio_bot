import logging
import os
import asyncio
import sys

# Импорты Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ConversationHandler, filters, ContextTypes

# Импортируем наш модуль Google
# Если файл называется google_services.py, то импорт верный
try:
    from google_services import GoogleBrain
except ImportError:
    print("CRITICAL ERROR: Не найден файл google_services.py!")
    sys.exit(1)

# Настройка логов
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# Убираем шум от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)

# Состояния диалога
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

# --- Инициализация Мозга (Google) ---
try:
    brain = GoogleBrain()
except Exception as e:
    logger.critical(f"FATAL: Не удалось запустить Google Brain: {e}")
    sys.exit(1)

# --- Клавиатуры ---
def get_start_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎡 Создать Карусель (Nano Banana)", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото от текста", callback_data='mode_cleaner_info')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_topic_keyboard(topics):
    keyboard = []
    for topic in topics:
        # Обрезаем колбек, чтобы не превысить лимит телеграма (64 байта)
        safe_topic = topic[:30]
        keyboard.append([InlineKeyboardButton(topic, callback_data=f'topic_{safe_topic}')])
    keyboard.append([InlineKeyboardButton("✍️ Написать свою тему", callback_data='topic_custom')])
    return InlineKeyboardMarkup(keyboard)

# --- Хендлеры ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **Привет! Я бот-редактор на базе Google AI.**\n\n"
        "🔥 **Мои возможности:**\n"
        "1. Генерирую *Wealth-карусели* с текстом и правильным дизайном (зеленые кружочки, стрелочки).\n"
        "2. Удаляю текст с картинок и чищу фон.\n\n"
        "Выбери режим:",
        reply_markup=get_start_keyboard(),
        parse_mode="Markdown"
    )
    return ConversationHandler.END

# --- ЛОГИКА ОЧИСТКИ ФОТО ---
async def mode_cleaner_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Отправь мне фото, с которого нужно убрать текст (я очищу нижнюю часть).")

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
        
    photo_file = await update.message.photo[-1].get_file()
    img_bytes = await photo_file.download_as_bytearray()
    
    msg = await update.message.reply_text("⏳ Nano Banana (Imagen 3) удаляет текст...\nЭто займет 10-15 сек.")
    
    # Запускаем тяжелую задачу в отдельном потоке
    cleaned_bytes = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
    
    if cleaned_bytes:
        await msg.delete()
        await update.message.reply_photo(cleaned_bytes, caption="✅ Текст удален!")
    else:
        await msg.edit_text("❌ Ошибка при обработке. Попробуйте другое фото.")

# --- ЛОГИКА КАРУСЕЛЕЙ ---

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("🧠 Gemini придумывает хайповые темы...")
    topics = await asyncio.to_thread(brain.generate_topics)
    
    await query.message.reply_text(
        "Выбери тему из списка или предложи свою:",
        reply_markup=get_topic_keyboard(topics)
    )
    return CHOOSING_MODE

async def topic_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'topic_custom':
        await query.message.reply_text("Введи свою тему:")
        return ENTERING_TOPIC
    
    # Пытаемся достать текст кнопки, которую нажал юзер
    # Это костыль, но рабочий для инлайн кнопок
    chosen_topic = "Тема из списка"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == data:
                chosen_topic = btn.text
                break
    
    return await generate_plan_step(update, context, chosen_topic)

async def topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text
    return await generate_plan_step(update, context, topic)

async def generate_plan_step(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    msg_source = update.callback_query.message if update.callback_query else update.message
    status_msg = await msg_source.reply_text(f"🧠 Разрабатываю сценарий для: *{topic}*...\nGemini пишет тексты и промпты...", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    
    if not plan:
        await status_msg.edit_text("❌ Не удалось сгенерировать план. Попробуй другую тему.")
        return ConversationHandler.END
    
    context.user_data['carousel_plan'] = plan
    context.user_data['carousel_topic'] = topic
    
    # Превью плана
    preview = f"📝 **План карусели ({len(plan)} слайдов):**\n\n"
    for slide in plan:
        # Показываем только начало текста, чтобы не спамить
        caption_preview = slide.get('ru_caption', 'Без текста')[:80]
        preview += f"🔹 **Слайд {slide.get('slide_number', '?')}:**\n{caption_preview}...\n\n"
        
    keyboard = [
        [InlineKeyboardButton("🚀 Генерировать картинки", callback_data='generate_go')],
        [InlineKeyboardButton("🔄 Пересобрать текст", callback_data='regen_plan')],
        [InlineKeyboardButton("❌ Отмена", callback_data='cancel')]
    ]
    
    await status_msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def generate_execution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'cancel':
        await query.edit_message_text("Отменено.")
        return ConversationHandler.END
        
    if data == 'regen_plan':
        topic = context.user_data.get('carousel_topic')
        return await generate_plan_step(update, context, topic)
    
    # ЗАПУСК ГЕНЕРАЦИИ
    plan = context.user_data['carousel_plan']
    total = len(plan)
    await query.message.reply_text(f"🚀 Начинаю производство {total} слайдов через Nano Banana (Imagen 3)...\nБуду присылать по мере готовности.")
    
    for i, slide in enumerate(plan):
        prompt = slide.get('image_prompt')
        caption = slide.get('ru_caption', '')
        num = slide.get('slide_number', i+1)
        
        # Генерируем
        img_bytes = await asyncio.to_thread(brain.generate_image, prompt)
        
        if img_bytes:
            # Форматируем подпись
            full_caption = f"📄 **Слайд {num}/{total}**\n\n{caption}\n\n_#WealthAI_"
            try:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=img_bytes,
                    caption=full_caption[:1024], # Лимит телеграма
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Ошибка отправки фото: {e}")
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⚠️ Слайд {num}: Картинка сгенерирована, но не отправилась.")
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"⚠️ Ошибка генерации картинки для слайда {num}"
            )
            
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Карусель полностью готова!")
    return ConversationHandler.END

def main():
    # 1. ЗАБИРАЕМ ТОКЕН ТУТ (с очисткой от пробелов)
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    
    # 2. ПРОВЕРЯЕМ
    if not token:
        print("❌ ОШИБКА: Переменная TELEGRAM_TOKEN пустая! Проверьте Railway Variables.")
        sys.exit(1)
    else:
        print(f"✅ Токен найден (длина: {len(token)}). Запускаю бота...")

    # 3. СТРОИМ БОТА
    application = Application.builder().token(token).build()

    # Conversation для карусели
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(topic_choice, pattern='^topic_')],
            ENTERING_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, topic_input)],
            CONFIRMING_PLAN: [CallbackQueryHandler(generate_execution, pattern='^(generate_go|regen_plan|cancel)$')]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(mode_cleaner_info, pattern='^mode_cleaner_info$'))
    application.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    logger.info("Бот переходит в режим polling...")
    application.run_polling()

if __name__ == '__main__':
    main()
