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

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

(CHOOSING_CATEGORY, CHOOSING_TOPIC, ENTERING_CUSTOM_TOPIC, 
 EDITING_TOPIC, CHOOSING_COUNT, CONFIRMING_PLAN, 
 GENERATING, AWAITING_FEEDBACK) = range(8)

try:
    brain = GoogleBrain()
except Exception as e:
    logger.error(f"Brain init failed: {e}")
    sys.exit(1)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception: {context.error}")


async def safe_edit(msg, text: str, markup=None):
    try:
        await msg.edit_text(text, reply_markup=markup, parse_mode="Markdown")
    except:
        try:
            await msg.edit_text(text, reply_markup=markup)
        except:
            pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    kb = [
        [InlineKeyboardButton("📊 Карусель", callback_data="mode_carousel")],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data="mode_cleaner")]
    ]
    text = "💎 *Wealth AI v9*\n\nВыберите режим:"
    if update.callback_query:
        await update.callback_query.answer()
        await safe_edit(update.callback_query.message, text, InlineKeyboardMarkup(kb))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    return ConversationHandler.END


async def show_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    kb = []
    for cat_id, cat_data in CATEGORIES.items():
        kb.append([InlineKeyboardButton(cat_data["name"], callback_data=f"cat_{cat_id}")])
    kb.append([InlineKeyboardButton("✍️ Своя тема", callback_data="custom_topic")])
    kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="back_main")])
    
    await safe_edit(query.message, "📂 *Выберите категорию:*", InlineKeyboardMarkup(kb))
    return CHOOSING_CATEGORY


async def handle_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "custom_topic":
        await safe_edit(query.message, "✍️ Напишите тему:", 
                       InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_categories")]]))
        return ENTERING_CUSTOM_TOPIC
    
    cat_id = query.data.replace("cat_", "")
    context.user_data["category"] = cat_id
    
    await safe_edit(query.message, "🧠 Генерирую темы...")
    
    try:
        topics = await asyncio.wait_for(
            asyncio.to_thread(brain.generate_topics_by_category, cat_id), 
            timeout=15.0
        )
    except:
        topics = ["Ошибка генерации"]
    
    context.user_data["topics"] = topics
    
    kb = []
    for i, t in enumerate(topics):
        kb.append([InlineKeyboardButton(t[:45], callback_data=f"top_{i}")])
    kb.append([InlineKeyboardButton("🔄 Другие темы", callback_data=f"cat_{cat_id}")])
    kb.append([InlineKeyboardButton("✍️ Своя тема", callback_data="custom_topic")])
    kb.append([InlineKeyboardButton("⬅️ Категории", callback_data="back_categories")])
    
    cat_name = CATEGORIES.get(cat_id, {}).get("name", "")
    await safe_edit(query.message, f"{cat_name}\n\n*Выберите тему:*", InlineKeyboardMarkup(kb))
    return CHOOSING_TOPIC


async def handle_topic_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    idx = int(query.data.replace("top_", ""))
    topics = context.user_data.get("topics", [])
    topic = topics[idx] if idx < len(topics) else "Тема"
    context.user_data["topic"] = topic
    
    return await show_topic_confirm(update, context, topic)


async def handle_custom_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text.strip()
    context.user_data["topic"] = topic
    return await show_topic_confirm(update, context, topic)


async def show_topic_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str):
    kb = [
        [InlineKeyboardButton("✅ Подтвердить", callback_data="confirm_topic")],
        [InlineKeyboardButton("✏️ Изменить", callback_data="edit_topic")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_categories")]
    ]
    text = f"📌 *Тема:*\n{topic}\n\nПодтвердить или изменить?"
    
    if update.callback_query:
        await safe_edit(update.callback_query.message, text, InlineKeyboardMarkup(kb))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    return EDITING_TOPIC


async def handle_edit_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    current = context.user_data.get("topic", "")
    await safe_edit(query.message, f"Текущая тема: {current}\n\n✍️ Напишите новую:", 
                   InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="cancel_edit")]]))
    return ENTERING_CUSTOM_TOPIC


async def handle_confirm_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await show_count_selection(update, context)


async def show_count_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = context.user_data.get("topic", "Тема")
    kb = [
        [
            InlineKeyboardButton("1", callback_data="cnt_1"),
            InlineKeyboardButton("3", callback_data="cnt_3"),
            InlineKeyboardButton("6", callback_data="cnt_6")
        ],
        [
            InlineKeyboardButton("10", callback_data="cnt_10"),
            InlineKeyboardButton("12", callback_data="cnt_12")
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_topic")]
    ]
    text = f"📌 {topic}\n\n*Количество слайдов:*\n\n1 = только обложка\n3-6 = короткая карусель\n10-12 = лонгрид"
    
    if update.callback_query:
        await safe_edit(update.callback_query.message, text, InlineKeyboardMarkup(kb))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    return CHOOSING_COUNT


async def handle_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    count = int(query.data.replace("cnt_", ""))
    context.user_data["count"] = count
    topic = context.user_data.get("topic", "Тема")
    
    await safe_edit(query.message, f"📝 Создаю план ({count} сл.)...")
    
    try:
        plan = await asyncio.wait_for(
            asyncio.to_thread(brain.generate_carousel_plan, topic, count),
            timeout=20.0
        )
    except:
        plan = []
    
    if not plan:
        await safe_edit(query.message, "❌ Ошибка плана.", 
                       InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Повторить", callback_data=f"cnt_{count}"),
                                             InlineKeyboardButton("⬅️ Меню", callback_data="back_main")]]))
        return CHOOSING_COUNT
    
    context.user_data["plan"] = plan
    
    preview = f"📊 *{topic}*\n\n"
    for s in plan:
        num = s.get("slide_number", "?")
        cap = s.get("ru_caption", "...")[:50]
        cover = " (коллаж)" if s.get("is_cover") else ""
        preview += f"{num}. {cap}{cover}\n"
    
    kb = [
        [InlineKeyboardButton("🚀 Генерировать", callback_data="gen_start")],
        [InlineKeyboardButton("🔄 Новый план", callback_data=f"cnt_{count}")],
        [InlineKeyboardButton("⬅️ Кол-во", callback_data="back_count")]
    ]
    await safe_edit(query.message, preview, InlineKeyboardMarkup(kb))
    return CONFIRMING_PLAN


async def start_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    plan = context.user_data.get("plan", [])
    if not plan:
        await start(update, context)
        return ConversationHandler.END
    
    context.user_data["current_slide"] = 0
    context.user_data["generated"] = []
    
    await query.message.delete()
    return await generate_next_slide(update, context)


async def generate_next_slide(update: Update, context: ContextTypes.DEFAULT_TYPE):
    plan = context.user_data.get("plan", [])
    idx = context.user_data.get("current_slide", 0)
    
    if idx >= len(plan):
        await context.bot.send_message(
            update.effective_chat.id,
            "✅ *Готово!*",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Меню", callback_data="back_main")]])
        )
        return ConversationHandler.END
    
    slide = plan[idx]
    total = len(plan)
    caption = slide.get("ru_caption", "")
    prompt = slide.get("image_prompt", "")
    is_cover = slide.get("is_cover", False)
    
    status = await context.bot.send_message(
        update.effective_chat.id,
        f"⏳ Слайд {idx+1}/{total}..."
    )
    
    try:
        img = await asyncio.wait_for(
            asyncio.to_thread(brain.generate_image, prompt, is_cover),
            timeout=35.0
        )
    except:
        img = None
    
    await status.delete()
    
    if not img:
        kb = [
            [InlineKeyboardButton("🔄 Повторить", callback_data="retry_slide")],
            [InlineKeyboardButton("⏭ Пропустить", callback_data="skip_slide")],
            [InlineKeyboardButton("❌ Стоп", callback_data="stop_gen")]
        ]
        await context.bot.send_message(
            update.effective_chat.id,
            f"⚠️ Слайд {idx+1} не получился",
            reply_markup=InlineKeyboardMarkup(kb)
        )
        return GENERATING
    
    context.user_data["last_image"] = img
    context.user_data["last_prompt"] = prompt
    
    cover_tag = " 🎨" if is_cover else ""
    kb = [
        [
            InlineKeyboardButton("✅", callback_data="accept_slide"),
            InlineKeyboardButton("🔄", callback_data="retry_slide"),
            InlineKeyboardButton("✏️", callback_data="edit_slide")
        ],
        [
            InlineKeyboardButton("⏭ Пропустить", callback_data="skip_slide"),
            InlineKeyboardButton("❌ Стоп", callback_data="stop_gen")
        ]
    ]
    
    await context.bot.send_photo(
        update.effective_chat.id,
        img,
        caption=f"*{idx+1}/{total}{cover_tag}*\n\n{caption}",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return GENERATING


async def handle_slide_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data
    
    if action == "accept_slide":
        img = context.user_data.get("last_image")
        if img:
            context.user_data.setdefault("generated", []).append(img)
        context.user_data["current_slide"] = context.user_data.get("current_slide", 0) + 1
        return await generate_next_slide(update, context)
    
    elif action == "retry_slide":
        return await generate_next_slide(update, context)
    
    elif action == "edit_slide":
        await context.bot.send_message(
            update.effective_chat.id,
            "✏️ Что изменить? (напишите текстом)",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="cancel_feedback")]])
        )
        return AWAITING_FEEDBACK
    
    elif action == "skip_slide":
        context.user_data["current_slide"] = context.user_data.get("current_slide", 0) + 1
        return await generate_next_slide(update, context)
    
    elif action == "stop_gen":
        await context.bot.send_message(
            update.effective_chat.id,
            "🛑 Генерация остановлена.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Меню", callback_data="back_main")]])
        )
        return ConversationHandler.END
    
    return GENERATING


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    feedback = update.message.text.strip()
    original_prompt = context.user_data.get("last_prompt", "")
    
    plan = context.user_data.get("plan", [])
    idx = context.user_data.get("current_slide", 0)
    is_cover = plan[idx].get("is_cover", False) if idx < len(plan) else False
    
    status = await update.message.reply_text("🔄 Применяю изменения...")
    
    try:
        new_prompt, img = await asyncio.wait_for(
            asyncio.to_thread(brain.regenerate_with_feedback, original_prompt, feedback, is_cover),
            timeout=35.0
        )
    except:
        new_prompt, img = original_prompt, None
    
    await status.delete()
    
    if not img:
        kb = [[InlineKeyboardButton("🔄 Ещё раз", callback_data="retry_slide"),
               InlineKeyboardButton("⏭ Пропустить", callback_data="skip_slide")]]
        await update.message.reply_text("⚠️ Не получилось", reply_markup=InlineKeyboardMarkup(kb))
        return GENERATING
    
    context.user_data["last_image"] = img
    context.user_data["last_prompt"] = new_prompt
    
    slide = plan[idx] if idx < len(plan) else {}
    caption = slide.get("ru_caption", "")
    total = len(plan)
    
    kb = [
        [
            InlineKeyboardButton("✅", callback_data="accept_slide"),
            InlineKeyboardButton("🔄", callback_data="retry_slide"),
            InlineKeyboardButton("✏️", callback_data="edit_slide")
        ],
        [
            InlineKeyboardButton("⏭ Пропустить", callback_data="skip_slide"),
            InlineKeyboardButton("❌ Стоп", callback_data="stop_gen")
        ]
    ]
    
    await context.bot.send_photo(
        update.effective_chat.id,
        img,
        caption=f"*{idx+1}/{total}* (изменено)\n\n{caption}",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return GENERATING


async def cancel_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    plan = context.user_data.get("plan", [])
    idx = context.user_data.get("current_slide", 0)
    slide = plan[idx] if idx < len(plan) else {}
    caption = slide.get("ru_caption", "")
    img = context.user_data.get("last_image")
    
    if img:
        kb = [
            [
                InlineKeyboardButton("✅", callback_data="accept_slide"),
                InlineKeyboardButton("🔄", callback_data="retry_slide"),
                InlineKeyboardButton("✏️", callback_data="edit_slide")
            ],
            [
                InlineKeyboardButton("⏭ Пропустить", callback_data="skip_slide"),
                InlineKeyboardButton("❌ Стоп", callback_data="stop_gen")
            ]
        ]
        await context.bot.send_message(
            update.effective_chat.id,
            f"Текущий слайд: {caption}",
            reply_markup=InlineKeyboardMarkup(kb)
        )
    return GENERATING


async def back_to_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await show_categories(update, context)


async def back_to_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    topic = context.user_data.get("topic", "Тема")
    return await show_topic_confirm(update, context, topic)


async def back_to_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await show_count_selection(update, context)


async def mode_cleaner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await safe_edit(query.message, "📷 Пришлите фото для очистки:",
                   InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Меню", callback_data="back_main")]]))


async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
    
    msg = await update.message.reply_text("⏳ Обработка...")
    try:
        f = await update.message.photo[-1].get_file()
        b = await f.download_as_bytearray()
        res = await asyncio.to_thread(brain.remove_text_from_image, bytes(b))
        if res:
            await msg.delete()
            await update.message.reply_photo(res, caption="✅ Готово")
        else:
            await msg.edit_text("❌ Ошибка")
    except Exception as e:
        logger.error(f"Photo error: {e}")
        await msg.edit_text("⚠️ Сбой")


def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        sys.exit(1)

    request = HTTPXRequest(http_version="1.1", connection_pool_size=10,
                          read_timeout=60, write_timeout=60, connect_timeout=30)
    app = Application.builder().token(token).request(request).build()
    app.add_error_handler(error_handler)

    conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(show_categories, pattern="^mode_carousel$")
        ],
        states={
            CHOOSING_CATEGORY: [
                CallbackQueryHandler(handle_category, pattern="^cat_"),
                CallbackQueryHandler(handle_category, pattern="^custom_topic$"),
                CallbackQueryHandler(start, pattern="^back_main$")
            ],
            CHOOSING_TOPIC: [
                CallbackQueryHandler(handle_topic_select, pattern="^top_"),
                CallbackQueryHandler(handle_category, pattern="^cat_"),
                CallbackQueryHandler(handle_category, pattern="^custom_topic$"),
                CallbackQueryHandler(back_to_categories, pattern="^back_categories$")
            ],
            ENTERING_CUSTOM_TOPIC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_topic),
                CallbackQueryHandler(back_to_categories, pattern="^back_categories$"),
                CallbackQueryHandler(back_to_topic, pattern="^cancel_edit$")
            ],
            EDITING_TOPIC: [
                CallbackQueryHandler(handle_confirm_topic, pattern="^confirm_topic$"),
                CallbackQueryHandler(handle_edit_topic, pattern="^edit_topic$"),
                CallbackQueryHandler(back_to_categories, pattern="^back_categories$")
            ],
            CHOOSING_COUNT: [
                CallbackQueryHandler(handle_count, pattern="^cnt_"),
                CallbackQueryHandler(back_to_topic, pattern="^back_topic$"),
                CallbackQueryHandler(start, pattern="^back_main$")
            ],
            CONFIRMING_PLAN: [
                CallbackQueryHandler(start_generation, pattern="^gen_start$"),
                CallbackQueryHandler(handle_count, pattern="^cnt_"),
                CallbackQueryHandler(back_to_count, pattern="^back_count$"),
                CallbackQueryHandler(start, pattern="^back_main$")
            ],
            GENERATING: [
                CallbackQueryHandler(handle_slide_action, pattern="^(accept|retry|edit|skip|stop)_"),
                CallbackQueryHandler(cancel_feedback, pattern="^cancel_feedback$")
            ],
            AWAITING_FEEDBACK: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_feedback),
                CallbackQueryHandler(cancel_feedback, pattern="^cancel_feedback$")
            ]
        },
        fallbacks=[
            CallbackQueryHandler(start, pattern="^back_main$"),
            CommandHandler("start", start)
        ],
        conversation_timeout=1200
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(mode_cleaner, pattern="^mode_cleaner$"))
    app.add_handler(CallbackQueryHandler(start, pattern="^back_main$"))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo))

    print("✅ Bot v9.0")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
