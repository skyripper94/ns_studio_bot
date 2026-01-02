import logging
import os
import asyncio
import sys
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)

try:
    from google_services import GoogleBrain
except ImportError:
    print("CRITICAL: google_services.py not found!")
    sys.exit(1)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

brain = None

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception: {context.error}")
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("⚠️ Ошибка. Попробуй позже.")

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit=False):
    text = "🚀 **Nano Banana AI v3.0**\n\nВыбери режим:"
    keyboard = [
        [InlineKeyboardButton("🎡 Карусель", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=reply_markup, parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_main_menu(update, context)
    return ConversationHandler.END

async def cancel_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query:
        await query.answer()
    await send_main_menu(update, context, edit=True)
    return ConversationHandler.END

async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "📷 Пришли фото для очистки нижней части.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]])
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
    photo_file = await update.message.photo[-1].get_file()
    img_bytes = await photo_file.download_as_bytearray()
    msg = await update.message.reply_text("⏳ Обрабатываю...")
    cleaned_bytes = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
    if cleaned_bytes:
        await msg.delete()
        await update.message.reply_photo(cleaned_bytes, caption="✅ Готово!")
    else:
        await msg.edit_text("❌ Ошибка обработки.")
    await send_main_menu(update, context)

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    msg = await query.edit_message_text("🧠 Gemini генерирует темы...")
    try:
        topics = await asyncio.to_thread(brain.generate_topics)
        keyboard = []
        for t in topics:
            keyboard.append([InlineKeyboardButton(t[:30], callback_data=f"ts_{t[:20]}")])
        keyboard.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
        await msg.edit_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(keyboard))
        return CHOOSING_MODE
    except Exception as e:
        logger.error(f"Topics Error: {e}")
        await msg.edit_text("❌ Ошибка API.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]))
        return ConversationHandler.END

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "topic_custom":
        await query.edit_message_text("Напиши тему:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]))
        return ENTERING_TOPIC
    topic = "Тема"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == data:
                topic = btn.text
                break
    return await start_generation_plan(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await start_generation_plan(update, context, update.message.text)

async def start_generation_plan(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ Создаю план: *{topic}*...", parse_mode="Markdown")
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    if not plan:
        await status_msg.edit_text("❌ Ошибка плана.")
        return ConversationHandler.END
    context.user_data['current_plan'] = plan
    preview = "📝 **План:**\n\n"
    for s in plan:
        preview += f"{s.get('slide_number', '-')}. {s.get('ru_caption', '')[:40]}...\n"
    keyboard = [
        [InlineKeyboardButton("🚀 Генерировать", callback_data="confirm_gen")],
        [InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]
    ]
    await status_msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    plan = context.user_data.get('current_plan')
    await query.edit_message_text(f"🎨 Рисую {len(plan)} слайдов...")
    for i, slide in enumerate(plan, 1):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ {i}/{len(plan)}...")
        img_bytes = await asyncio.to_thread(brain.generate_image, slide.get('image_prompt'))
        if img_bytes:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=img_bytes, caption=slide.get('ru_caption'))
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⚠️ Слайд {i} не получился")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Готово!")
    await send_main_menu(update, context)
    return ConversationHandler.END

def main():
    global brain
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        print("TELEGRAM_TOKEN missing!")
        sys.exit(1)
    print(f"✅ Token: {token[:8]}...")
    
    try:
        brain = GoogleBrain()
    except Exception as e:
        print(f"❌ GoogleBrain init failed: {e}")
        sys.exit(1)

    app = Application.builder().token(token).build()
    app.add_error_handler(error_handler)

    carousel_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(handle_topic_selection, pattern='^ts_|^topic_custom$')],
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

    print("✅ Bot started (v3.0)!")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
