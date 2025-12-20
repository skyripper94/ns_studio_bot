# telegram_bot.py

"""
Telegram бот с 2 основными режимами:
1. УДАЛИТЬ - только удаление текста (существующий функционал)
2. FULL - полный workflow с 3 подрежимами + двухэтапный контроль
"""

import os
import logging
from io import BytesIO
import pickle

import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from dotenv import load_dotenv

from lama_integration import (
    flux_kontext_inpaint, 
    google_vision_ocr,
    flux_inpaint,
    openai_translate,
    create_gradient_layer,
    render_mode1_logo,
    render_mode2_text,
    render_mode3_content,
    MASK_BOTTOM_PERCENT,
    OCR_BOTTOM_PERCENT
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TEMP_DIR = '/tmp/bot_images'
os.makedirs(TEMP_DIR, exist_ok=True)

user_states = {}


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок."""
    logger.error("❌ Ошибка в обработчике", exc_info=context.error)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start - показывает меню выбора режима."""
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        user_states[user_id] = {}
    
    user_states[user_id].update({'mode': None, 'submode': None, 'step': None})
    
    keyboard = [
        [
            InlineKeyboardButton("🗑️ УДАЛИТЬ ТЕКСТ", callback_data="mode_remove"),
            InlineKeyboardButton("🔄 ПОЛНЫЙ ЦИКЛ", callback_data="mode_full")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "👋 **Бот для работы с изображениями**\n\n"
        "**🗑️ УДАЛИТЬ ТЕКСТ:**\n"
        "Только удаление текста и градиента (LaMa)\n\n"
        "**🔄 ПОЛНЫЙ ЦИКЛ:**\n"
        "OCR → Контроль → Удаление → Перевод → Контроль → Рендер\n"
        "3 режима: Лого / Текст / Контент\n\n"
        "Выберите режим:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора режима через inline кнопки."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        user_states[user_id] = {}
    
    if query.data == "mode_remove":
        user_states[user_id]['mode'] = 'remove'
        await query.edit_message_text(
            "✅ **Режим: УДАЛИТЬ ТЕКСТ**\n\n"
            "Просто отправьте изображение.\n"
            f"Бот удалит текст и градиент ({MASK_BOTTOM_PERCENT}% снизу).",
            parse_mode='Markdown'
        )
    
    elif query.data == "mode_full":
        user_states[user_id]['mode'] = 'full'
        
        keyboard = [
            [
                InlineKeyboardButton("1️⃣ ЛОГО", callback_data="submode_1"),
                InlineKeyboardButton("2️⃣ ТЕКСТ", callback_data="submode_2"),
                InlineKeyboardButton("3️⃣ КОНТЕНТ", callback_data="submode_3")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "✅ **Режим: ПОЛНЫЙ ЦИКЛ**\n\n"
            "Выберите подрежим:\n\n"
            "**1️⃣ ЛОГО** - Лого + линии + заголовок\n"
            "**2️⃣ ТЕКСТ** - Только заголовок\n"
            "**3️⃣ КОНТЕНТ** - Заголовок + подзаголовок\n\n"
            "После выбора отправьте изображение.",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    elif query.data.startswith("submode_"):
        submode = int(query.data.split("_")[1])
        user_states[user_id]['submode'] = submode
        
        mode_names = {
            1: "ЛОГО (лого + заголовок)",
            2: "ТЕКСТ (только заголовок)",
            3: "КОНТЕНТ (заголовок + подзаголовок)"
        }
        
        # Подсказка для режима 3
        mode3_hint = ""
        if submode == 3:
            mode3_hint = (
                "\n\n"
                "📝 **Как писать для режима КОНТЕНТ:**\n"
                "Все строки КРОМЕ последней = ЗАГОЛОВОК (бирюзовый)\n"
                "Последняя строка = ПОДЗАГОЛОВОК (белый)\n\n"
                "Пример:\n"
                "`Самые дорогие`\n"
                "`творения человечества`\n"
                "`Стоимость $100 млрд.`"
            )
        
        await query.edit_message_text(
            f"✅ **Выбран режим {submode}: {mode_names[submode]}**\n\n"
            f"Теперь отправьте изображение для обработки.\n\n"
            f"Бот выполнит:\n"
            f"1. OCR → контроль\n"
            f"2. Удаление текста\n"
            f"3. Перевод → контроль\n"
            f"4. Нанесение текста"
            f"{mode3_hint}",
            parse_mode='Markdown'
        )
    
    elif query.data == "next_ocr":
        await handle_ocr_next(update, context)
    
    elif query.data == "edit_ocr":
        await handle_ocr_edit(update, context)
    
    elif query.data == "next_llm":
        await handle_llm_next(update, context)
    
    elif query.data == "edit_llm":
        await handle_llm_edit(update, context)


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка полученного изображения."""
    user_id = update.effective_user.id
    
    if user_id not in user_states or user_states[user_id].get('mode') is None:
        await update.message.reply_text("⚠️ Сначала выберите режим командой /start")
        return
    
    mode = user_states[user_id]['mode']
    submode = user_states[user_id].get('submode')
    
    if mode == 'full' and submode is None:
        await update.message.reply_text("⚠️ Сначала выберите подрежим (1/2/3)")
        return
    
    try:
        photo = await update.message.photo[-1].get_file()
        image_bytes = await photo.download_as_bytearray()
        
        logger.info(f"✅ Изображение от пользователя {user_id}, режим: {mode}, подрежим: {submode}")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mode == 'remove':
            await process_remove_mode(update, image)
        
        elif mode == 'full':
            await process_full_mode_step1(update, image, submode, user_id)
    
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


async def process_remove_mode(update: Update, image: np.ndarray):
    """РЕЖИМ УДАЛЕНИЯ: только убираем текст."""
    status_msg = await update.message.reply_text("⏳ Удаление текста...")
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_start = int(height * (1 - MASK_BOTTOM_PERCENT / 100))
    mask[mask_start:, :] = 255
    
    result = flux_kontext_inpaint(image, mask)
    
    success, buffer = cv2.imencode('.png', result)
    if success:
        await update.message.reply_photo(
            photo=BytesIO(buffer.tobytes()),
            caption="✅ **Текст удалён!**\n🎨 LaMa",
            parse_mode='Markdown'
        )
        await status_msg.delete()


async def process_full_mode_step1(update: Update, image: np.ndarray, submode: int, user_id: int):
    """ШАГ 1: OCR → показать → ждать решения."""
    status_msg = await update.message.reply_text("⏳ **Шаг 1/4:** OCR...", parse_mode='Markdown')
    
    ocr = google_vision_ocr(image, crop_bottom_percent=OCR_BOTTOM_PERCENT)
    
    if not ocr["text"]:
        await update.message.reply_text("⚠️ Текст не обнаружен")
        await status_msg.delete()
        return
    
    ocr_text = ocr["text"]
    ocr_preview = ocr_text[:300] + "..." if len(ocr_text) > 300 else ocr_text
    
    image_path = f"{TEMP_DIR}/{user_id}_image.pkl"
    with open(image_path, 'wb') as f:
        pickle.dump(image, f)
    
    user_states[user_id].update({
        'step': 'waiting_ocr_decision',
        'ocr_text': ocr_text,
        'image_path': image_path,
        'submode': submode
    })
    
    keyboard = [
        [
            InlineKeyboardButton("✏️ Править", callback_data="edit_ocr"),
            InlineKeyboardButton("➡️ Далее", callback_data="next_ocr")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"📝 **OCR распознал:**\n\n{ocr_preview}\n\n"
        f"Выберите действие:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    await status_msg.delete()


async def handle_ocr_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь нажал ✏️ Править для OCR."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    user_states[user_id]['step'] = 'editing_ocr'
    
    await query.edit_message_text(
        "✏️ **Отправьте исправленный текст**\n\n"
        "Пришлите текст который должен быть переведён.",
        parse_mode='Markdown'
    )


async def handle_ocr_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь нажал ➡️ Далее для OCR."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    state = user_states[user_id]
    ocr_text = state['ocr_text']
    
    await query.edit_message_text(
        f"✅ **OCR текст принят**\n\n{ocr_text[:200]}...",
        parse_mode='Markdown'
    )
    
    await process_full_mode_step2(query, user_id, ocr_text)


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстового ввода (для редактирования OCR/LLM)."""
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        return
    
    step = user_states[user_id].get('step')
    
    if step == 'editing_ocr':
        custom_text = update.message.text.strip()
        user_states[user_id]['ocr_text'] = custom_text
        
        await update.message.reply_text(
            f"✅ **Текст обновлён**\n\n{custom_text[:200]}...",
            parse_mode='Markdown'
        )
        
        await process_full_mode_step2(update, user_id, custom_text)
    
    elif step == 'editing_llm':
        custom_translation = update.message.text.strip()
        
        state = user_states[user_id]
        submode = state['submode']
        
        if submode == 3:
            lines = custom_translation.split('\n')
            if len(lines) >= 2:
                user_states[user_id]['llm_title'] = '\n'.join(lines[:-1])
                user_states[user_id]['llm_subtitle'] = lines[-1]
            else:
                user_states[user_id]['llm_title'] = custom_translation
                user_states[user_id]['llm_subtitle'] = ""
        else:
            user_states[user_id]['llm_title'] = custom_translation
        
        await update.message.reply_text(
            f"✅ **Перевод обновлён**\n\n{custom_translation[:200]}...",
            parse_mode='Markdown'
        )
        
        await process_full_mode_step3(update, user_id)


async def process_full_mode_step2(update, user_id: int, ocr_text: str):
    """ШАГ 2: Inpaint + LLM → показать → ждать решения."""
    
    if hasattr(update, 'message'):
        msg_target = update.message
    else:
        msg_target = update
    
    status_msg = await msg_target.reply_text(
        "⏳ **Шаг 2/4:** Удаление текста...",
        parse_mode='Markdown'
    )
    
    state = user_states[user_id]
    image_path = state['image_path']
    submode = state['submode']
    
    with open(image_path, 'rb') as f:
        image = pickle.load(f)
    
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_start = int(h * (1 - MASK_BOTTOM_PERCENT / 100))
    mask[mask_start:, :] = 255
    
    clean_image = flux_inpaint(image, mask)
    
    clean_path = f"{TEMP_DIR}/{user_id}_clean.pkl"
    with open(clean_path, 'wb') as f:
        pickle.dump(clean_image, f)
    user_states[user_id]['clean_path'] = clean_path
    
    await status_msg.edit_text("⏳ **Шаг 3/4:** Перевод (LLM)...", parse_mode='Markdown')
    
    if submode == 3:
        lines = ocr_text.split('\n')
        if len(lines) >= 2:
            title = " ".join(lines[:-1])
            subtitle = lines[-1]
        else:
            title, subtitle = ocr_text, ""
        
        title_translated = openai_translate(title)
        subtitle_translated = openai_translate(subtitle) if subtitle else ""
        
        user_states[user_id]['llm_title'] = title_translated
        user_states[user_id]['llm_subtitle'] = subtitle_translated
        
        llm_preview = f"{title_translated}\n{subtitle_translated}" if subtitle_translated else title_translated
    else:
        title_translated = openai_translate(ocr_text)
        user_states[user_id]['llm_title'] = title_translated
        user_states[user_id]['llm_subtitle'] = ""
        llm_preview = title_translated
    
    user_states[user_id]['step'] = 'waiting_llm_decision'
    
    keyboard = [
        [
            InlineKeyboardButton("✏️ Править", callback_data="edit_llm"),
            InlineKeyboardButton("➡️ Далее", callback_data="next_llm")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await msg_target.reply_text(
        f"🌐 **LLM перевёл:**\n\n{llm_preview}\n\n"
        f"Выберите действие:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    # НЕ УДАЛЯЕМ сообщение "Шаг 3/4: Перевод (LLM)..." - оставляем как с OCR


async def handle_llm_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    user_states[user_id]['step'] = 'editing_llm'
    
    submode = user_states[user_id]['submode']
    
    if submode == 3:
        hint = (
            "**Как писать:**\n"
            "Все строки КРОМЕ последней → ЗАГОЛОВОК (бирюзовый)\n"
            "Последняя строка → ПОДЗАГОЛОВОК (белый)\n\n"
            "Пример:\n"
            "`Портфель Ambani`\n"
            "`Недвижимость на $50 млрд.`"
        )
    else:
        hint = "Пришлите текст для заголовка"
    
    # ИЗМЕНЕНО: было query.edit_message_text
    await query.message.reply_text(
        f"✏️ **Отправьте исправленный перевод**\n\n{hint}",
        parse_mode='Markdown'
    )


async def handle_llm_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    state = user_states[user_id]
    llm_title = state['llm_title']
    llm_subtitle = state.get('llm_subtitle', '')
    
    preview = f"{llm_title}\n{llm_subtitle}" if llm_subtitle else llm_title
    
    # ИЗМЕНЕНО: было query.edit_message_text
    await query.message.reply_text(
        f"✅ **Перевод принят**\n\n{preview[:200]}...",
        parse_mode='Markdown'
    )
    
    await process_full_mode_step3(query, user_id)
    

async def process_full_mode_step3(update, user_id: int):
    """ШАГ 3: Градиент + Рендер → готово."""
    
    if hasattr(update, 'message'):
        msg_target = update.message
    else:
        msg_target = update
    
    status_msg = await msg_target.reply_text(
        "⏳ **Шаг 4/4:** Рендер...",
        parse_mode='Markdown'
    )
    
    state = user_states[user_id]
    clean_path = state['clean_path']
    submode = state['submode']
    llm_title = state['llm_title']
    llm_subtitle = state.get('llm_subtitle', '')
    
    with open(clean_path, 'rb') as f:
        clean_image = pickle.load(f)
    
    from PIL import Image as PILImage
    clean_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(clean_rgb).convert("RGBA")
    
    if submode == 3:
        grad = create_gradient_layer(pil.size[0], pil.size[1], cover_percent=65, solid_raise_px=0)
    else:
        grad = create_gradient_layer(pil.size[0], pil.size[1], cover_percent=65, solid_raise_px=80)
    
    pil = PILImage.alpha_composite(pil, grad)
    
    if submode == 1:
        pil = render_mode1_logo(pil, llm_title)
    elif submode == 2:
        pil = render_mode2_text(pil, llm_title)
    elif submode == 3:
        pil = render_mode3_content(pil, llm_title, llm_subtitle)
    
    out_rgb = np.array(pil.convert("RGB"))
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    
    success, buffer = cv2.imencode('.png', out_bgr)
    if success:
        mode_names = {1: "ЛОГО", 2: "ТЕКСТ", 3: "КОНТЕНТ"}
        
        await msg_target.reply_photo(
            photo=BytesIO(buffer.tobytes()),
            caption=(
                f"✅ **Готово! (Режим {submode}: {mode_names[submode]})**\n\n"
                f"🎨 LaMa → Градиент → Рендер"
            ),
            parse_mode='Markdown'
        )
        await status_msg.delete()
    
    try:
        os.remove(state['image_path'])
        os.remove(clean_path)
    except:
        pass
    
    user_states[user_id]['step'] = None


def main():
    """Запуск бота."""
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN не установлен!")
        return
    
    logger.info("🚀 Запуск бота...")
    
    request = HTTPXRequest(connect_timeout=10.0, read_timeout=40.0, write_timeout=40.0, pool_timeout=40.0)
    application = Application.builder().token(TELEGRAM_TOKEN).request(request).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(mode_callback))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))
    application.add_error_handler(on_error)
    
    logger.info("✅ Бот запущен!")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, poll_interval=1.0, timeout=30)


if __name__ == '__main__':
    main()
