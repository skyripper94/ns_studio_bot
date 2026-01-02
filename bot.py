import logging
import os
import asyncio
import sys
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)
from telegram.request import HTTPXRequest # <-- Для настройки тайм-аутов

# Проверка сервисов
try:
    from google_services import GoogleBrain
except ImportError:
    print("CRITICAL: google_services.py не найден!")
    sys.exit(1)

# Логирование (без мусора)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Состояния
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

try:
    brain = GoogleBrain()
except Exception:
    sys.exit(1)

# --- МЕНЮ ---
async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit=False):
    text = "💎 **Wealth AI Creator**\n\nСоздаю факты, сравнения и чищу фото.\nЧто делаем?"
    keyboard = [
        [InlineKeyboardButton("📊 Создать Карусель (Facts/Vs)", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup, parse_mode="Markdown")
    else:
        await context.bot.send_message(update.effective_chat.id, text, reply_markup=markup, parse_mode="Markdown")

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query: await update.callback_query.answer()
    await send_main_menu(update, context, edit=True)
    return ConversationHandler.END

# --- ОЧИСТКА ---
async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query: await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        "📷 Пришли фото, я очищу текст снизу.", 
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]])
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo: return
    msg = await update.message.reply_text("⏳ Чищу...")
    try:
        f = await update.message.photo[-1].get_file()
        b = await f.download_as_bytearray()
        res = await asyncio.to_thread(brain.remove_text_from_image, bytes(b))
        if res:
            await msg.delete()
            await update.message.reply_photo(res, caption="✅ Чисто.")
        else:
            await msg.edit_text("❌ Ошибка.")
    except:
        await msg.edit_text("⚠️ Сбой.")
    await send_main_menu(update, context)

# --- КАРУСЕЛИ ---
async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query: await update.callback_query.answer()
    await update.callback_query.edit_message_text("🧠 Ищу хайповые факты...")
    
    topics = await asyncio.to_thread(brain.generate_topics)
    kb = [[InlineKeyboardButton(t, callback_data=f"ts_{t[:30]}")] for t in topics]
    kb.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
    kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")])
    
    await update.callback_query.edit_message_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(kb))
    return CHOOSING_MODE

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "topic_custom":
        await q.edit_message_text("Напиши тему (например: iPhone vs Android):", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]))
        return ENTERING_TOPIC
    
    topic = "Тема"
    for row in q.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == q.data: topic = btn.text
    return await generate_plan_step(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await generate_plan_step(update, context, update.message.text)

async def generate_plan_step(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    msg_func = update.message.reply_text if update.message else update.callback_query.message.reply_text
    msg = await msg_func(f"📝 Собираю факты: {topic}...", parse_mode="Markdown")
    
    plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
    if not plan:
        await msg.edit_text("❌ Не вышло.")
        return ConversationHandler.END

    context.user_data['plan'] = plan
    preview = f"📊 **План:**\n" + "\n".join([f"- {s.get('ru_caption')}" for s in plan])
    kb = [[InlineKeyboardButton("🚀 Генерировать", callback_data="go")], [InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]]
    await msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    return CONFIRMING_PLAN

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    plan = context.user_data.get('plan')
    await q.edit_message_text(f"🎨 Рисую {len(plan)} слайдов (с паузами)...")
    
    for i, slide in enumerate(plan):
        prompt = slide.get('image_prompt')
        caption = slide.get('ru_caption')
        
        status = await context.bot.send_message(update.effective_chat.id, f"Слайд {i+1}...")
        img = await asyncio.to_thread(brain.generate_image, prompt)
        
        if img:
            await status.delete()
            await context.bot.send_photo(update.effective_chat.id, img, caption=f"**{caption}**\n\n#{i+1}", parse_mode="Markdown")
        else:
            await status.edit_text("⚠️ Ошибка.")
        
        if i < len(plan) - 1: await asyncio.sleep(8) # Пауза 8 сек
            
    await context.bot.send_message(update.effective_chat.id, "✅ Готово!")
    await send_main_menu(update, context)
    return ConversationHandler.END

# --- ЗАПУСК ---
def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip().replace('"', '').replace("'", "")
    if not token: sys.exit(1)

    # ВАЖНО: Настройка тайм-аутов для стабильности
    request = HTTPXRequest(connection_pool_size=8, read_timeout=30.0, write_timeout=30.0, connect_timeout=30.0)
    app = Application.builder().token(token).request(request).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [CallbackQueryHandler(handle_topic_selection, pattern='^ts_')],
            ENTERING_TOPIC: [MessageHandler(filters.TEXT, handle_custom_topic_input)],
            CONFIRMING_PLAN: [CallbackQueryHandler(run_final_generation, pattern='^go$')]
        },
        fallbacks=[CallbackQueryHandler(back_to_main, pattern='^back_to_main$'), CommandHandler('start', start)]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(mode_cleaner_start, pattern='^mode_cleaner$'))
    app.add_handler(CallbackQueryHandler(back_to_main, pattern='^back_to_main$'))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    print("✅ Бот запущен (Stable + Wealth Style)!")
    app.run_polling()

if __name__ == '__main__':
    main()
