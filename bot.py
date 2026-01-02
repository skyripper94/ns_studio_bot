import logging
import os
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ConversationHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# Импортируем наш новый модуль Google
from google_services import GoogleBrain

# Настройка логов
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Токен ТГ
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Состояния диалога
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

# Инициализация Мозга
try:
    brain = GoogleBrain()
except Exception as e:
    logger.critical(f"FATAL: Не удалось запустить Google Brain: {e}")
    exit(1)

# --- Клавиатуры ---
def get_start_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎡 Создать Карусель", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner_info')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_topic_keyboard(topics):
    keyboard = []
    for topic in topics:
        keyboard.append([InlineKeyboardButton(topic, callback_data=f'topic_{topic[:30]}')]) # обрезаем длинные callback
    keyboard.append([InlineKeyboardButton("✍️ Написать свою тему", callback_data='topic_custom')])
    return InlineKeyboardMarkup(keyboard)

# --- Хендлеры ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я твой AI-редактор.\n\n"
        "Я умею:\n"
        "1. Генерировать **Wealth-карусели** (3-12 слайдов) через Google Imagen 3.\n"
        "2. Удалять текст с картинок.\n\n"
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
    photo_file = await update.message.photo[-1].get_file()
    img_bytes = await photo_file.download_as_bytearray()
    
    msg = await update.message.reply_text("⏳ Nano Banana (Imagen 3) удаляет текст...")
    
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
    
    await query.edit_message_text("Думаю над темами...")
    topics = await asyncio.to_thread(brain.generate_topics)
    
    await query.message.reply_text(
        "Выбери тему или предложи свою:",
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
    
    # Пользователь выбрал тему из списка (нужно найти полное название, здесь упростим)
    # В реальном проекте лучше сохранять список тем в context.user_data
    topic = query.message.reply_markup.inline_keyboard[[x.callback_data for row in query.message.reply_markup.inline_keyboard for x in row].index(data)][0].text
    
    return await generate_plan_step(update, context, topic)

async def topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text
    return await generate_plan_step(update, context, topic)

async def generate_plan_step(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    msg_source = update.callback_query.message if update.callback_query else update.message
    status_msg = await msg_source.reply_text(f"🧠 Разрабатываю сценарий для: *{topic}*...\nЭто может занять 15-30 сек.", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    
    if not plan:
        await status_msg.edit_text("Ошибка генерации плана. Попробуй другую тему.")
        return ConversationHandler.END
    
    context.user_data['carousel_plan'] = plan
    context.user_data['carousel_topic'] = topic
    
    # Формируем превью текста
    preview = f"📝 **План карусели ({len(plan)} слайдов):**\n\n"
    for slide in plan:
        preview += f"🔹 **Слайд {slide['slide_number']}:**\n{slide['ru_caption'][:100]}...\n\n"
        
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
    
    # START GENERATION
    plan = context.user_data['carousel_plan']
    total = len(plan)
    await query.message.reply_text(f"🚀 Начинаю производство {total} слайдов через Nano Banana...\nБуду присылать по готовности.")
    
    for slide in plan:
        prompt = slide['image_prompt']
        caption = slide['ru_caption']
        num = slide['slide_number']
        
        # Генерация (в потоке)
        img_bytes = await asyncio.to_thread(brain.generate_image, prompt)
        
        if img_bytes:
            # Формируем красивую подпись
            full_caption = f"📄 **Слайд {num}/{total}**\n\n{caption}"
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_bytes,
                caption=full_caption,
                parse_mode="Markdown"
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"⚠️ Ошибка генерации слайда {num}"
            )
            
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Карусель готова!")
    return ConversationHandler.END

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

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

    logger.info("Бот запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()
