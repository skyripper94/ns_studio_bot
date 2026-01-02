import logging
import os
import asyncio
import sys
import io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)

# Пытаемся импортировать GoogleBrain
try:
    from google_services import GoogleBrain
except ImportError:
    print("CRITICAL: google_services.py not found!")
    sys.exit(1)

# Логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния диалога
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

# Инициализация мозга
brain = GoogleBrain()

# --- Вспомогательные функции интерфейса ---

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit=False):
    text = (
        "🚀 **Nano Banana AI v2.1**\n\n"
        "Выбери режим работы:"
    )
    keyboard = [
        [InlineKeyboardButton("🎡 Создать Карусель", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото от текста", callback_data='mode_cleaner')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=reply_markup, parse_mode="Markdown")

# --- Хендлеры управления ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_main_menu(update, context)
    return ConversationHandler.END

async def cancel_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query:
        await query.answer()
    await send_main_menu(update, context, edit=True)
    return ConversationHandler.END

# --- ЛОГИКА ОЧИСТКИ ФОТО ---

async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "📷 Пришли мне фото, с которого нужно удалить текст.\n"
        "Я автоматически очищу нижнюю область изображения.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]])
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
        
    photo_file = await update.message.photo[-1].get_file()
    img_bytes = await photo_file.download_as_bytearray()
    
    msg = await update.message.reply_text("⏳ Nano Banana чистит фон... Подождите.")
    
    # Вызываем очистку из GoogleBrain
    cleaned_bytes = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
    
    if cleaned_bytes:
        await msg.delete()
        await update.message.reply_photo(cleaned_bytes, caption="✅ Готово! Текст удален.")
    else:
        await msg.edit_text("❌ Не удалось очистить это фото.")
    
    await send_main_menu(update, context)

# --- ЛОГИКА КАРУСЕЛЕЙ ---

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    msg = await query.edit_message_text("🧠 Gemini подбирает актуальные темы...")
    
    try:
        topics = await asyncio.to_thread(brain.generate_topics)
        keyboard = []
        for t in topics:
            keyboard.append([InlineKeyboardButton(t, callback_data=f"topic_select_{t[:25]}")])
        
        keyboard.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
        
        await msg.edit_text("Выбери тему для карусели:", reply_markup=InlineKeyboardMarkup(keyboard))
        return CHOOSING_MODE
    except Exception as e:
        logger.error(f"Error getting topics: {e}")
        await msg.edit_text("❌ Ошибка связи с Google. Попробуйте позже.", 
                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]))
        return ConversationHandler.END

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "topic_custom":
        await query.edit_message_text("Введите вашу тему текстом:", 
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]))
        return ENTERING_TOPIC
    
    # Ищем текст кнопки
    topic = "Выбранная тема"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == query.data:
                topic = btn.text
    
    return await start_generation_plan(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text
    return await start_generation_plan(update, context, topic)

async def start_generation_plan(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ Генерирую план для: *{topic}*...", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    if not plan:
        await status_msg.edit_text("❌ Ошибка при создании сценария.")
        return ConversationHandler.END

    context.user_data['current_plan'] = plan
    
    preview = "📝 **Сценарий готов:**\n\n"
    for i, slide in enumerate(plan, 1):
        preview += f"{i}. {slide.get('ru_caption', '')[:45]}...\n"

    keyboard = [
        [InlineKeyboardButton("🚀 Создать картинки", callback_data="confirm_gen")],
        [InlineKeyboardButton("🔄 Другие темы", callback_data="mode_carousel")],
        [InlineKeyboardButton("⬅️ В меню", callback_data="back_to_main")]
    ]
    await status_msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    plan = context.user_data.get('current_plan')
    await query.edit_message_text(f"🎨 Начинаю генерацию {len(plan)} слайдов. Это может занять пару минут...")
    
    for slide in plan:
        img_bytes = await asyncio.to_thread(brain.generate_image, slide.get('image_prompt'))
        if img_bytes:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_bytes,
                caption=slide.get('ru_caption')
            )
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ Ошибка генерации одного из слайдов.")
            
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Карусель готова!")
    await send_main_menu(update, context)
    return ConversationHandler.END

# --- ЗАПУСК ---

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token: 
        print("TELEGRAM_TOKEN is missing!")
        sys.exit(1)

    app = Application.builder().token(token).build()

    carousel_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(handle_topic_selection, pattern='^topic_')],
            ENTERING_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_topic_input)],
            CONFIRMING_PLAN: [CallbackQueryHandler(run_final_generation, pattern='^confirm_gen$')]
        },
        fallbacks=[
            CallbackQueryHandler(cancel_action, pattern='^back_to_main$'),
            CommandHandler('start', start)
        ]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(carousel_handler)
    app.add_handler(CallbackQueryHandler(mode_cleaner_start, pattern='^mode_cleaner$'))
    app.add_handler(CallbackQueryHandler(cancel_action, pattern='^back_to_main$'))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    print("✅ Бот запущен и готов к работе!")
    app.run_polling()

if __name__ == '__main__':
    main()
