import logging
import os
import asyncio
import sys
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)
from telegram.request import HTTPXRequest

try:
    from google_services import GoogleBrain, CATEGORIES
except ImportError:
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
(CHOOSING_CATEGORY, CHOOSING_TOPIC, ENTERING_CUSTOM, CHOOSING_COUNT, CONFIRMING_PLAN) = range(5)

brain = GoogleBrain()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    kb = [[InlineKeyboardButton("📊 Создать Карусель", callback_data='mode_carousel')],
          [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')]]
    text = "💎 **Nano Banana AI v10**\nВыберите режим:"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    return ConversationHandler.END

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    kb = [[InlineKeyboardButton(v["name"], callback_data=f"cat_{k}")] for k, v in CATEGORIES.items()]
    kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="back_main")])
    await update.callback_query.edit_message_text("📂 Выберите категорию:", reply_markup=InlineKeyboardMarkup(kb))
    return CHOOSING_CATEGORY

async def handle_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    cat_key = query.data.replace("cat_", "")
    context.user_data["cat_key"] = cat_key
    
    topics = await asyncio.to_thread(brain.generate_topics, cat_key)
    kb = [[InlineKeyboardButton(t, callback_data=f"ts_{t[:30]}")] for t in topics]
    kb.append([InlineKeyboardButton("✍️ Своя тема", callback_data="custom")])
    kb.append([InlineKeyboardButton("⬅️ Назад", callback_data="mode_carousel")])
    
    await query.edit_message_text(f"🔥 Темы для {CATEGORIES[cat_key]['name']}:", reply_markup=InlineKeyboardMarkup(kb))
    return CHOOSING_TOPIC

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "custom":
        await query.edit_message_text("✍️ Введите вашу тему:")
        return ENTERING_CUSTOM
    
    # Извлекаем текст кнопки
    topic = next(b.text for r in query.message.reply_markup.inline_keyboard for b in r if b.callback_data == query.data)
    context.user_data["topic"] = topic
    return await ask_count(query, context)

async def handle_custom_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["topic"] = update.message.text
    return await ask_count(update, context)

async def ask_count(event, context):
    kb = [[InlineKeyboardButton(f"{i} слайдов", callback_data=f"cnt_{i}")] for i in [1, 3, 5, 10]]
    text = f"📌 Тема: {context.user_data['topic']}\nСколько слайдов делаем?"
    if hasattr(event, "edit_message_text"):
        await event.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))
    else:
        await event.reply_text(text, reply_markup=InlineKeyboardMarkup(kb))
    return CHOOSING_COUNT

async def handle_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    count = int(query.data.replace("cnt_", ""))
    context.user_data["count"] = count
    
    await query.edit_message_text("🧠 Составляю план премиум-карусели...")
    plan = await asyncio.to_thread(brain.generate_carousel_plan, context.user_data["topic"], count)
    context.user_data["plan"] = plan
    
    preview = f"📋 План ({count} слайдов):\n" + "\n".join([f"• {s['ru_caption']}" for s in plan])
    kb = [[InlineKeyboardButton("🚀 ПОЕХАЛИ", callback_data="go")], [InlineKeyboardButton("⬅️ Отмена", callback_data="back_main")]]
    await query.edit_message_text(preview, reply_markup=InlineKeyboardMarkup(kb))
    return CONFIRMING_PLAN

async def run_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    plan = context.user_data["plan"]
    
    await query.edit_message_text("🎨 Начинаю генерацию. Это займет пару минут...")
    
    for i, slide in enumerate(plan):
        msg = await context.bot.send_message(update.effective_chat.id, f"🖼 Слайд {i+1}/{len(plan)}: Генерирую...")
        img = await asyncio.to_thread(brain.generate_image, slide["image_prompt"])
        
        if img:
            await msg.delete()
            await context.bot.send_photo(
                update.effective_chat.id, 
                img, 
                caption=f"**{slide['ru_caption']}**", 
                parse_mode="Markdown"
            )
        else:
            await msg.edit_text(f"❌ Ошибка на слайде {i+1}")
        
        await asyncio.sleep(2) # Защита от лимитов

    await context.bot.send_message(update.effective_chat.id, "✅ Карусель готова!")
    await start(update, context)
    return ConversationHandler.END

async def mode_cleaner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.edit_message_text("📷 Отправьте фото, и я уберу лишнее снизу.")

async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    img_bytes = await photo.download_as_bytearray()
    res = await asyncio.to_thread(brain.remove_text_from_image, bytes(img_bytes))
    if res: await update.message.reply_photo(res, caption="✨ Очищено")
    await start(update, context)

def main():
    token = os.getenv("TELEGRAM_TOKEN").strip().replace('"', '')
    app = Application.builder().token(token).build()
    
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern="^mode_carousel$")],
        states={
            CHOOSING_CATEGORY: [CallbackQueryHandler(handle_category, pattern="^cat_")],
            CHOOSING_TOPIC: [CallbackQueryHandler(handle_topic_selection, pattern="^ts_"), CallbackQueryHandler(handle_topic_selection, pattern="^custom$")],
            ENTERING_CUSTOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_input)],
            CHOOSING_COUNT: [CallbackQueryHandler(handle_count, pattern="^cnt_")],
            CONFIRMING_PLAN: [CallbackQueryHandler(run_generation, pattern="^go$")]
        },
        fallbacks=[CallbackQueryHandler(start, pattern="^back_main$"), CommandHandler("start", start)]
    )
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(mode_cleaner, pattern="^mode_cleaner$"))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo))
    app.run_polling()

if __name__ == "__main__": main()
