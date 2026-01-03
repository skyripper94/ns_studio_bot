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

# 1. Проверка наличия мозгов
try:
    from google_services import GoogleBrain
except ImportError:
    print("CRITICAL: google_services.py не найден!")
    sys.exit(1)

# 2. Логирование (Чистое, без мусора)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
# Заглушаем технический шум библиотек, оставляем только важное
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Состояния
CHOOSING_MODE, ENTERING_TOPIC, CONFIRMING_PLAN = range(3)

# 3. Инициализация AI
try:
    brain = GoogleBrain()
except Exception as e:
    logger.critical(f"Brain Death: {e}")
    sys.exit(1)

# --- ИНТЕРФЕЙС ---

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Сбрасываем флаг обработки, чтобы разблокировать бота
    context.user_data['is_processing'] = False
    
    text = "💎 **Wealth AI Creator v5.0 (Final Patch)**\n\nСистемы в норме. Выбери задачу:"
    keyboard = [
        [InlineKeyboardButton("📊 Создать Карусель", callback_data='mode_carousel')],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data='mode_cleaner')]
    ]
    
    if update.callback_query:
        # Безопасное редактирование (try/except на случай, если сообщение старое)
        try:
            await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        except:
            await update.callback_query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_main_menu(update, context)
    return ConversationHandler.END

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query: await update.callback_query.answer()
    await send_main_menu(update, context)
    return ConversationHandler.END

# --- ОЧИСТКА ФОТО ---

async def mode_cleaner_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "📷 **Режим очистки**\nПришли фото - я уберу текст снизу.", 
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]])
    )

async def process_photo_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo: return
    
    if context.user_data.get('is_processing'):
        await update.message.reply_text("⏳ Я занят, подожди...")
        return
    context.user_data['is_processing'] = True

    msg = await update.message.reply_text("⏳ Обработка...")
    try:
        f = await update.message.photo[-1].get_file()
        b = await f.download_as_bytearray()
        
        # Heavy lifting in thread
        res = await asyncio.to_thread(brain.remove_text_from_image, bytes(b))
        
        if res:
            await msg.delete()
            await update.message.reply_photo(res, caption="✅ Готово.")
        else:
            await msg.edit_text("❌ Ошибка обработки.")
    except Exception as e:
        logger.error(f"Photo Error: {e}")
        await msg.edit_text("⚠️ Сбой сервера.")
    finally:
        context.user_data['is_processing'] = False
        await send_main_menu(update, context)

# --- КАРУСЕЛИ ---

async def mode_carousel_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("🧠 Gemini 2.0 генерирует идеи...")
    
    topics = await asyncio.to_thread(brain.generate_topics)
    
    kb = [[InlineKeyboardButton(t, callback_data=f"ts_{t[:30]}")] for t in topics]
    kb.append([InlineKeyboardButton("✍️ Своя тема", callback_data="topic_custom")])
    kb.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
    
    await query.edit_message_text("🔥 Темы:", reply_markup=InlineKeyboardMarkup(kb))
    return CHOOSING_MODE

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "topic_custom":
        await query.edit_message_text(
            "✍️ Напиши тему:", 
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="back_to_main")]])
        )
        return ENTERING_TOPIC
    
    topic = "Тема"
    for row in query.message.reply_markup.inline_keyboard:
        for btn in row:
            if btn.callback_data == query.data: topic = btn.text
    return await generate_plan_step(update, context, topic)

async def handle_custom_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await generate_plan_step(update, context, update.message.text)

async def generate_plan_step(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    # Блокировка повторных нажатий
    if context.user_data.get('is_processing'): return CONFIRMING_PLAN
    context.user_data['is_processing'] = True
    
    # Определяем куда отвечать
    if update.callback_query:
        msg = await update.callback_query.message.reply_text(f"📝 План: **{topic}**...", parse_mode="Markdown")
    else:
        msg = await update.message.reply_text(f"📝 План: **{topic}**...", parse_mode="Markdown")
    
    context.user_data['current_topic'] = topic

    try:
        plan = await asyncio.to_thread(brain.generate_carousel_plan, topic)
        context.user_data['plan'] = plan
        
        preview = f"📊 **План:** {topic}\n\n"
        if not plan: preview += "⚠️ Пусто (Нажми Переписать)"
        for s in plan:
            preview += f"🔹 {s.get('ru_caption', '...')}\n"
        
        kb = [
            [InlineKeyboardButton("🚀 Генерировать", callback_data="go")],
            [InlineKeyboardButton("🔄 Переписать", callback_data="regen_plan")],
            [InlineKeyboardButton("⬅️ Меню", callback_data="back_to_main")]
        ]
        await msg.edit_text(preview, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    finally:
        context.user_data['is_processing'] = False
        
    return CONFIRMING_PLAN

async def regenerate_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Обновляю...")
    topic = context.user_data.get('current_topic')
    if not topic:
        await query.message.reply_text("⚠️ Данные устарели.")
        return ConversationHandler.END
    return await generate_plan_step(update, context, topic)

async def run_final_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    plan = context.user_data.get('plan')
    if not plan:
        await query.message.reply_text("⚠️ Ошибка данных.")
        return ConversationHandler.END

    if context.user_data.get('is_gen_running'):
        await query.message.reply_text("⏳ Уже работаю...")
        return
    context.user_data['is_gen_running'] = True
    
    try:
        await query.edit_message_text(f"🎨 Рисую {len(plan)} слайдов...")
        
        for i, slide in enumerate(plan):
            prompt = slide.get('image_prompt')
            caption = slide.get('ru_caption')
            
            status = await context.bot.send_message(update.effective_chat.id, f"Слайд {i+1}...")
            img = await asyncio.to_thread(brain.generate_image, prompt)
            
            if img:
                await status.delete()
                await context.bot.send_photo(update.effective_chat.id, img, caption=f"**{caption}**\n\n#{i+1}", parse_mode="Markdown")
            else:
                await status.edit_text(f"⚠️ Слайд {i+1} пропущен.")
            
            if i < len(plan) - 1: await asyncio.sleep(8)
                
        await context.bot.send_message(update.effective_chat.id, "✅ Готово!")
    finally:
        context.user_data['is_gen_running'] = False
        context.user_data['plan'] = None
        await send_main_menu(update, context)
        
    return ConversationHandler.END

# --- ЗАПУСК ---

def main():
    token = os.getenv("TELEGRAM_TOKEN", "").strip().replace('"', '').replace("'", "")
    if not token: sys.exit(1)

    # ==========================================
    # 🛑 ФИКС СЕТИ: УВЕЛИЧЕНЫ ТАЙМ-АУТЫ ДО 120s
    # ==========================================
    request = HTTPXRequest(
        connection_pool_size=10, # Больше соединений
        read_timeout=120.0,      # Ждем ответ от Телеграма до 2 минут
        write_timeout=120.0,     # Отправляем данные до 2 минут
        connect_timeout=60.0,    # Соединяемся до 1 минуты
        pool_timeout=60.0        # Ждем свободного слота
    )

    app = Application.builder().token(token).request(request).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(mode_carousel_start, pattern='^mode_carousel$')],
        states={
            CHOOSING_MODE: [
                CallbackQueryHandler(handle_topic_selection, pattern='^ts_'),
                CallbackQueryHandler(handle_topic_selection, pattern='^topic_custom$')
            ],
            ENTERING_TOPIC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_topic_input),
                CallbackQueryHandler(back_to_main, pattern='^back_to_main$')
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
        conversation_timeout=1200 # 20 минут сессия
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(mode_cleaner_start, pattern='^mode_cleaner$'))
    app.add_handler(CallbackQueryHandler(back_to_main, pattern='^back_to_main$'))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo_cleanup))

    print("✅ Bot Started (Network Fix Applied)")
    # drop_pending_updates=True удалит старые зависшие сообщения, которые могли крашить бота при старте
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
