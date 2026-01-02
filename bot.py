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

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

brain = GoogleBrain()

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit=False):
    text = (
        "👋 **Главное меню Nano Banana AI**\n\n"
        "Я помогу тебе создать контент с помощью моделей Google Gemini & Imagen 3.\n\n"
        "Выбери, что хочешь сделать:"
    )
    keyboard = [
        [InlineKeyboardButton("🎡 Создать Карусель", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data='mode_help')]
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
        await query.answer("Возвращаемся...")
    await send_main_menu(update, context, edit=True)
    return ConversationHandler.END

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    msg = await query.edit_message_text("🧠 Gemini генерирует идеи тем...")
    
    try:
        topics = await asyncio.to_thread(brain.generate_topics)
        keyboard = []
        for t in topics[:5]:
            keyboard.append([InlineKeyboardButton(t, callback_data=f"topic_select_{t[:25]}")])
        
        keyboard.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
        
        await msg.edit_text("Выбери тему для карусели:", reply_markup=InlineKeyboardMarkup(keyboard))
        return CHOOSING_MODE
    except Exception as e:
        logger.error(f"Ошибка тем: {e}")
        await msg.edit_text("❌ Ошибка связи с Google. Попробуй позже.", 
                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]))
        return ConversationHandler.END

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "topic_custom":
        await query.edit_message_text("Напиши тему для карусели (например: 'Как заработать на ИИ'):", 
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]))
        return ENTERING_TOPIC
    
    topic = query.data.replace("topic_select_", "")
    return await start_generation_plan(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text
    return await start_generation_plan(update, context, topic)

async def start_generation_plan(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ Работаю над планом для темы: *{topic}*...", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    if not plan:
        await status_msg.edit_text("❌ Ошибка генерации плана.")
        return ConversationHandler.END

    context.user_data['current_plan'] = plan
    
    preview = "📝 **Ваш сценарий готов:**\n\n"
    for i, slide in enumerate(plan, 1):
        preview += f"{i}. {slide.get('ru_caption', '')[:50]}...\n"

    keyboard = [
        [InlineKeyboardButton("🚀 Запустить генерацию фото", callback_data="confirm_gen")],
        [InlineKeyboardButton("🔄 Другой вариант", callback_data="mode_carousel")],
        [InlineKeyboardButton("⬅️ В меню", callback_data="back_to_main")]
    ]
    await status_msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    plan = context.user_data.get('current_plan')
    await query.edit_message_text(f"🎨 Начинаю отрисовку {len(plan)} слайдов. Это займет около 1-2 минут...")
    
    for i, slide in enumerate(plan, 1):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ Генерация {i}/{len(plan)}...")
        img_bytes = await asyncio.to_thread(brain.generate_image, slide.get('image_prompt'))
        if img_bytes:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_bytes,
                caption=slide.get('ru_caption')
            )
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ Пропуск слайда из-за ошибки генерации.")
            
    await send_main_menu(update, context)
    return ConversationHandler.END

async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("📷 Пришли мне фото, которое нужно очистить от текста.", 
                                   reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]))

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    img_bytes = await photo_file.download_as_bytearray()
    
    msg = await update.message.reply_text("⏳ Очищаю фото от текста...")
    cleaned = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
    
    if cleaned:
        await msg.delete()
        await update.message.reply_photo(cleaned, caption="✅ Готово! Текст удален.")
    else:
        await msg.edit_text("❌ Не удалось очистить фото. Попробуй другое изображение.")
    
    await send_main_menu(update, context)

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    help_text = (
        "ℹ️ **Помощь**\n\n"
        "🎡 **Карусель** — создаю план из 5 слайдов и генерирую картинки через Imagen 3\n\n"
        "🧹 **Очистка фото** — удаляю текст и водяные знаки с изображений\n\n"
        "Просто выбери режим и следуй инструкциям!"
    )
    await query.edit_message_text(help_text, parse_mode="Markdown",
                                   reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]))

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        print("TELEGRAM_TOKEN is missing!")
        sys.exit(1)

    app = Application.builder().token(token).build()

    carousel_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(handle_topic_selection, pattern='^topic_')],
            ENTERING_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_topic_input)],
            CONFIRMING_PLAN: [CallbackQueryHandler(run_final_generation, pattern='^confirm_gen$')]
        },
        fallbacks=[
            CallbackQueryHandler(cancel_action, pattern='^back_to_main$'),
            CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$'),
            CommandHandler('start', start)
        ]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(carousel_conv)
    app.add_handler(CallbackQueryHandler(mode_cleaner_start, pattern='^mode_cleaner$'))
    app.add_handler(CallbackQueryHandler(help_handler, pattern='^mode_help$'))
    app.add_handler(CallbackQueryHandler(cancel_action, pattern='^back_to_main$'))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    print("✅ Nano Banana AI запущен!")
    app.run_polling()

if __name__ == '__main__':
    main()
