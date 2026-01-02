import logging
import os
import asyncio
import sys
import traceback
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

# ОТКЛЮЧАЕМ ШУМ БИБЛИОТЕК (Скрываем токен)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

# Состояния диалога
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

# Инициализация мозга
try:
    brain = GoogleBrain()
except Exception as e:
    logger.critical(f"Ошибка запуска Google Brain: {e}")
    sys.exit(1)

# --- Глобальный обработчик ошибок ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    # Пытаемся сообщить пользователю
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("⚠️ Произошла внутренняя ошибка. Попробуйте позже.")

# --- Вспомогательные функции интерфейса ---

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit=False):
    text = (
        "🚀 **Nano Banana AI v2.2 (Stable)**\n\n"
        "Выбери режим работы:"
    )
    keyboard = [
        [InlineKeyboardButton("🎡 Создать Карусель", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')],
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
        "Я автоматически очищу нижнюю область (30% снизу).",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]])
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
        
    photo_file = await update.message.photo[-1].get_file()
    img_bytes = await photo_file.download_as_bytearray()
    
    msg = await update.message.reply_text("⏳ Nano Banana чистит фон... Подождите 10-15 сек.")
    
    # Вызываем очистку
    cleaned_bytes = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
    
    if cleaned_bytes:
        await msg.delete()
        await update.message.reply_photo(cleaned_bytes, caption="✅ Готово! Текст удален.")
    else:
        await msg.edit_text("❌ Ошибка обработки. Возможно, формат фото не поддерживается или сбой API.")
    
    await send_main_menu(update, context)

# --- ЛОГИКА КАРУСЕЛЕЙ ---

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    msg = await query.edit_message_text("🧠 Gemini подбирает темы...")
    
    try:
        topics = await asyncio.to_thread(brain.generate_topics)
        keyboard = []
        for t in topics:
            # Обрезаем callback_data до 64 байт
            keyboard.append([InlineKeyboardButton(t, callback_data=f"ts_{t[:20]}")])
        
        keyboard.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
        
        await msg.edit_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(keyboard))
        return CHOOSING_MODE
    except Exception as e:
        logger.error(f"Topics Error: {e}")
        await msg.edit_text("❌ Ошибка связи с Google.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]))
        return ConversationHandler.END

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "topic_custom":
        await query.edit_message_text("Напиши тему текстом:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]))
        return ENTERING_TOPIC
    
    # Пытаемся найти текст кнопки
    topic = "Тема"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == data:
                topic = btn.text
                break
    
    return await start_generation_plan(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text
    return await start_generation_plan(update, context, topic)

async def start_generation_plan(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ Пишу сценарий для: *{topic}*...", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    if not plan:
        await status_msg.edit_text("❌ Не удалось создать план. Попробуйте другую тему.")
        return ConversationHandler.END

    context.user_data['current_plan'] = plan
    
    preview = "📝 **Сценарий:**\n\n"
    for slide in plan:
        num = slide.get('slide_number', '-')
        caption = slide.get('ru_caption', '')[:40]
        preview += f"{num}. {caption}...\n"

    keyboard = [
        [InlineKeyboardButton("🚀 Генерировать фото", callback_data="confirm_gen")],
        [InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]
    ]
    await status_msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    plan = context.user_data.get('current_plan')
    await query.edit_message_text(f"🎨 Рисую {len(plan)} слайдов. Это займет ~1 минуту...")
    
    for slide in plan:
        prompt = slide.get('image_prompt')
        caption = slide.get('ru_caption')
        
        img_bytes = await asyncio.to_thread(brain.generate_image, prompt)
        
        if img_bytes:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_bytes,
                caption=caption
            )
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⚠️ Слайд не получился (Google Filter).")
            
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Карусель готова!")
    await send_main_menu(update, context)
    return ConversationHandler.END

# --- ЗАПУСК ---

def main():
    # 1. ЧИСТКА ТОКЕНА (Удаляем пробелы, кавычки и переносы)
    raw_token = os.getenv("TELEGRAM_TOKEN", "")
    token = raw_token.strip().replace('"', '').replace("'", "")
    
    if not token: 
        print("CRITICAL: TELEGRAM_TOKEN пустой!")
        sys.exit(1)
        
    print(f"✅ Токен найден: {token[:5]}... (длина {len(token)})")

    # 2. Сборка приложения
    app = Application.builder().token(token).build()

    # 3. Добавление обработчика ошибок
    app.add_error_handler(error_handler)

    # 4. Сценарии
    carousel_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(handle_topic_selection, pattern='^ts_')],
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

    print("✅ Бот запущен (Stable v2.2)!")
    app.run_polling()

if __name__ == '__main__':
    main()
