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

# Проверка
try:
    from google_services import GoogleBrain
except ImportError:
    sys.exit(1)

# Логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Состояния: 4 ШАГА
CHOOSING_TOPIC, ENTERING_CUSTOM_TOPIC, CHOOSING_COUNT, CONFIRMING_PLAN = range(4)

try:
    brain = GoogleBrain()
except Exception:
    sys.exit(1)

# --- УТИЛИТА ДЛЯ ОТПРАВКИ ---
async def safe_reply(update: Update, text: str, markup=None, use_md=False):
    """Безопасная отправка без крашей из-за Markdown"""
    parse_mode = "Markdown" if use_md else None
    try:
        if update.callback_query:
            try:
                await update.callback_query.edit_message_text(text, reply_markup=markup, parse_mode=parse_mode)
            except:
                # Если не получилось отредактировать (например, текст не изменился) - шлем новое
                await update.callback_query.message.reply_text(text, reply_markup=markup, parse_mode=parse_mode)
        else:
            await update.message.reply_text(text, reply_markup=markup, parse_mode=parse_mode)
    except Exception as e:
        logger.error(f"Reply Error: {e}")

# --- МЕНЮ ---

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear() # Сброс состояния
    
    text = "💎 **Wealth AI Creator v7.0 (Russian Patch)**\n\nВыбери режим:"
    keyboard = [
        [InlineKeyboardButton("📊 Создать Контент (Коллажи/Факты)", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')]
    ]
    # Используем Markdown только в меню, где мы контролируем текст
    await safe_reply(update, text, InlineKeyboardMarkup(keyboard), use_md=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_main_menu(update, context)
    return ConversationHandler.END

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query: await update.callback_query.answer()
    await send_main_menu(update, context)
    return ConversationHandler.END

# --- ОЧИСТКА ---

async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "📷 **Очистка**\nПришли фото - я уберу текст снизу.", 
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]),
        parse_mode="Markdown"
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo: return
    msg = await update.message.reply_text("⏳ Обработка...")
    try:
        f = await update.message.photo[-1].get_file()
        b = await f.download_as_bytearray()
        res = await asyncio.to_thread(brain.remove_text_from_image, bytes(b))
        if res:
            await msg.delete()
            await update.message.reply_photo(res, caption="✅ Чисто.")
        else:
            await msg.edit_text("❌ Ошибка.")
    except Exception:
        await msg.edit_text("⚠️ Сбой.")
    await send_main_menu(update, context)

# --- ЛОГИКА КАРУСЕЛЕЙ (НОВАЯ) ---

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("🧠 Анализирую тренды...")
    
    topics = await asyncio.to_thread(brain.generate_topics)
    
    kb = []
    for t in topics:
        kb.append([InlineKeyboardButton(t, callback_data=f"topic_select_{t[:30]}")])
        
    kb.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
    kb.append([InlineKeyboardButton("🔄 Новые темы", callback_data="mode_carousel")])
    kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")])
    
    # БЕЗ MARKDOWN, чтобы не падало из-за спецсимволов в темах
    await query.edit_message_text("🔥 Выбери тему (Хук):", reply_markup=InlineKeyboardMarkup(kb))
    return CHOOSING_TOPIC

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "topic_custom":
        await query.edit_message_text("✍️ Напиши тему:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]]))
        return ENTERING_CUSTOM_TOPIC
    
    topic = "Тема"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == query.data: topic = btn.text
    
    return await ask_slide_count(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await ask_slide_count(update, context, update.message.text)

async def ask_slide_count(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    """Выбор количества слайдов"""
    context.user_data['current_topic'] = topic
    
    text = f"📌 Тема: {topic}\n\nСколько слайдов генерируем?"
    keyboard = [
        [InlineKeyboardButton("🖼 1 (Только обложка)", callback_data="count_1")],
        [InlineKeyboardButton("⚡ 3 (Быстро)", callback_data="count_3")],
        [InlineKeyboardButton("📚 6 (Стандарт)", callback_data="count_6")],
        [InlineKeyboardButton("📖 10 (Лонгрид)", callback_data="count_10")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="mode_carousel")]
    ]
    
    await safe_reply(update, text, InlineKeyboardMarkup(keyboard), use_md=False)
    return CHOOSING_COUNT

async def handle_count_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    count = int(query.data.split('_')[1])
    context.user_data['slide_count'] = count
    
    topic = context.user_data.get('current_topic', 'Тема')
    return await generate_plan_step(update, context, topic, count)

async def generate_plan_step(update: Update, context: ContextTypes.DEFAULT_TYPE, topic, count):
    if context.user_data.get('is_processing'): return CONFIRMING_PLAN
    context.user_data['is_processing'] = True
    
    await update.callback_query.edit_message_text(f"📝 Пишу сценарий ({count} слайдов): {topic}...")
    
    try:
        plan = await asyncio.to_thread(brain.generate_carousel_plan, topic, count)
        context.user_data['plan'] = plan
        
        preview = f"📊 План: {topic}\n\n"
        if not plan: preview += "⚠️ Ошибка AI. Нажми 'Переписать'."
        for s in plan:
            preview += f"🔹 {s.get('ru_caption', '...')}\n"
        
        kb = [
            [InlineKeyboardButton("🚀 Генерировать", callback_data="go")],
            [InlineKeyboardButton("🔄 Переписать текст", callback_data="regen_plan")],
            [InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]
        ]
        await update.callback_query.edit_message_text(preview, reply_markup=InlineKeyboardMarkup(kb))
    finally:
        context.user_data['is_processing'] = False
        
    return CONFIRMING_PLAN

async def regenerate_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Обновляю...")
    
    topic = context.user_data.get('current_topic')
    count = context.user_data.get('slide_count', 3)
    
    if not topic:
        await query.message.reply_text("⚠️ Данные устарели.")
        return ConversationHandler.END
        
    return await generate_plan_step(update, context, topic, count)

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    plan = context.user_data.get('plan')
    if not plan:
        await query.message.reply_text("⚠️ Ошибка данных.")
        return ConversationHandler.END

    if context.user_data.get('is_gen_running'):
        await query.message.reply_text("⏳ Жди...")
        return
    context.user_data['is_gen_running'] = True
    
    try:
        await query.edit_message_text(f"🎨 Рисую {len(plan)} слайдов (Коллажи, 3:4)...")
        
        for i, slide in enumerate(plan):
            prompt = slide.get('image_prompt')
            caption = slide.get('ru_caption')
            
            status = await context.bot.send_message(update.effective_chat.id, f"Слайд {i+1}...")
            img = await asyncio.to_thread(brain.generate_image, prompt)
            
            if img:
                await status.delete()
                # Здесь Markdown можно оставить, так как caption обычно безопасен
                # но для 100% защиты от крашей caption можно экранировать
                try:
                    await context.bot.send_photo(
                        update.effective_chat.id, 
                        img, 
                        caption=f"**{caption}**\n\n#{i+1}", 
                        parse_mode="Markdown"
                    )
                except:
                    # Если маркдаун сломался - шлем без него
                     await context.bot.send_photo(update.effective_chat.id, img, caption=f"{caption}\n\n#{i+1}")
            else:
                await status.edit_text(f"⚠️ Слайд {i+1} пропущен.")
            
            if i < len(plan) - 1: await asyncio.sleep(8)
                
        await context.bot.send_message(update.effective_chat.id, "✅ Карусель готова!")
    finally:
        context.user_data['is_gen_running'] = False
        context.user_data['plan'] = None
        await send_main_menu(update, context)
        
    return ConversationHandler.END

# --- ЗАПУСК ---

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip().replace('"', '').replace("'", "")
    if not token: sys.exit(1)

    # Максимальные тайм-ауты
    request = HTTPXRequest(connection_pool_size=10, read_timeout=120.0, write_timeout=120.0, connect_timeout=60.0)
    app = Application.builder().token(token).request(request).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_TOPIC: [
                CallbackQueryHandler(handle_topic_selection, pattern='^topic_select_'),
                CallbackQueryHandler(handle_topic_selection, pattern='^topic_custom$'),
                CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')
            ],
            ENTERING_CUSTOM_TOPIC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_topic_input),
                CallbackQueryHandler(back_to_main, pattern='^back_to_main$')
            ],
            CHOOSING_COUNT: [
                CallbackQueryHandler(handle_count_selection, pattern='^count_'),
                CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')
            ],
            CONFIRMING_PLAN: [
                CallbackQueryHandler(run_final_generation, pattern='^go$'),
                CallbackQueryHandler(regenerate_plan, pattern='^regen_plan$')
            ]
        },
        fallbacks=[
            CallbackQueryHandler(back_to_main, pattern='^back_to_main$'),
            CommandHandler('start', start)
        ],
        conversation_timeout=1200
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(mode_cleaner_start, pattern='^mode_cleaner$'))
    app.add_handler(CallbackQueryHandler(back_to_main, pattern='^back_to_main$'))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    print("✅ Bot Started (Final Patch: Russian + Collages + Stability)")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
