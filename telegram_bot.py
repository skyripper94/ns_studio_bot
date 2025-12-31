# telegram_bot.py - IMPROVED VERSION WITH RETRY

"""
Telegram бот с 2 основными режимами:
1. УДАЛИТЬ - только удаление текста
2. FULL - полный workflow с 3 подрежимами + двухэтапный контроль
"""

import os
import logging
from io import BytesIO
import pickle
import time
import re
import asyncio

import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError
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
    enhance_image, 
    MASK_BOTTOM_MODE1,
    MASK_BOTTOM_MODE2,
    MASK_BOTTOM_MODE3,
    OCR_BOTTOM_PERCENT,
    GRADIENT_HEIGHT_MODE12,
    GRADIENT_HEIGHT_MODE3
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

class RedactTelegramTokenFilter(logging.Filter):
    _re = re.compile(r"(https://api\.telegram\.org/bot)(\d+:[A-Za-z0-9_-]+)")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self._re.sub(r"\1***", str(record.msg))
            if record.args:
                record.args = tuple(self._re.sub(r"\1***", str(a)) for a in record.args)
        except Exception:
            pass
        return True

logging.getLogger().addFilter(RedactTelegramTokenFilter())

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TEMP_DIR = '/tmp/bot_images'
os.makedirs(TEMP_DIR, exist_ok=True)

user_states = {}

RETRY_ATTEMPTS = 3
RETRY_DELAY = 3


async def send_with_retry(coro_func, retries=RETRY_ATTEMPTS, delay=RETRY_DELAY):
    last_error = None
    for attempt in range(retries):
        try:
            return await coro_func()
        except (TimedOut, NetworkError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < retries - 1:
                logger.warning(f"⚠️ Сеть: попытка {attempt+1}/{retries}, ошибка: {type(e).__name__}, жду {delay}с...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"❌ Все {retries} попыток неудачны: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка: {e}")
            raise
    raise last_error


def escape_md(text: str) -> str:
    for ch in ('_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'):
        text = text.replace(ch, '\\' + ch)
    return text


def cleanup_temp_files(temp_dir: str, max_age_hours: int = 12) -> int:
    now = time.time()
    max_age_sec = max_age_hours * 3600

    removed = 0
    try:
        for name in os.listdir(temp_dir):
            path = os.path.join(temp_dir, name)
            if not os.path.isfile(path):
                continue

            if not (name.endswith(".pkl") or name.endswith(".png")):
                continue
            if ("_image" not in name) and ("_clean" not in name) and ("_final" not in name):
                continue

            age_sec = now - os.path.getmtime(path)
            if age_sec >= max_age_sec:
                os.remove(path)
                removed += 1

    except Exception as e:
        logger.warning(f"cleanup_temp_files: ошибка: {e}")

    return removed


def _pick_msg_target(obj):
    if hasattr(obj, "message") and obj.message:
        return obj.message

    if hasattr(obj, "callback_query") and obj.callback_query:
        if getattr(obj.callback_query, "message", None):
            return obj.callback_query.message

    if hasattr(obj, "message") and obj.message:
        return obj.message

    if hasattr(obj, "reply_text"):
        return obj

    if hasattr(obj, "effective_message") and obj.effective_message:
        return obj.effective_message

    return None


def _cleanup_user_files(user_id: int):
    state = user_states.get(user_id, {})
    for k in ("image_path", "clean_path"):
        p = state.get(k)
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except:
                pass


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("❌ Ошибка в обработчике", exc_info=context.error)
    
    if update and hasattr(update, 'effective_user') and update.effective_user:
        _cleanup_user_files(update.effective_user.id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    removed = cleanup_temp_files(TEMP_DIR, max_age_hours=6)
    if removed:
        logger.info(f"🧹 TEMP cleanup: удалено {removed} старых файлов из {TEMP_DIR}")

    _cleanup_user_files(user_id)

    user_states[user_id] = {'mode': None, 'submode': None, 'step': None}

    keyboard = [
        [
            InlineKeyboardButton("🗑️ УДАЛИТЬ ТЕКСТ", callback_data="mode_remove"),
            InlineKeyboardButton("🔄 ПОЛНЫЙ ЦИКЛ", callback_data="mode_full")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await send_with_retry(lambda: update.message.reply_text(
        "👋 **Бот для работы с изображениями**\n\n"
        "**🗑️ УДАЛИТЬ ТЕКСТ:**\n"
        "Только удаление текста и градиента (LaMa)\n\n"
        "**🔄 ПОЛНЫЙ ЦИКЛ:**\n"
        "OCR → Контроль → Удаление → Перевод → Контроль → Рендер\n"
        "3 режима: Лого / Текст / Контент\n\n"
        "Выберите режим:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    ))


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        user_states[user_id] = {}
    
    if query.data == "back_to_start":
        _cleanup_user_files(user_id)
        user_states[user_id] = {'mode': None, 'submode': None, 'step': None}
        keyboard = [
            [
                InlineKeyboardButton("🗑️ УДАЛИТЬ ТЕКСТ", callback_data="mode_remove"),
                InlineKeyboardButton("🔄 ПОЛНЫЙ ЦИКЛ", callback_data="mode_full")
            ]
        ]
        await send_with_retry(lambda: query.edit_message_text(
            "👋 Выберите режим:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        ))
    
    elif query.data == "mode_remove":
        user_states[user_id]['mode'] = 'remove'
        keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="back_to_start")]]
        await send_with_retry(lambda: query.edit_message_text(
            "✅ **Режим: УДАЛИТЬ ТЕКСТ**\n\n"
            "Просто отправьте изображение.\n"
            f"Бот удалит текст и градиент ({MASK_BOTTOM_MODE2}% снизу).",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        ))

    elif query.data.startswith("render_mode_"):
        submode = int(query.data.split("_")[-1])
        user_states[user_id]['submode'] = submode
        user_states[user_id]['step'] = 'editing_llm'
        user_states[user_id]['llm_title'] = ''
        user_states[user_id]['llm_subtitle'] = ''
        
        if submode == 3:
            hint = (
                "Все строки КРОМЕ последней → ЗАГОЛОВОК (зеленый)\n"
                "Последняя строка → ПОДЗАГОЛОВОК (белый)\n\n"
                "Можно `|` для переноса."
            )
        else:
            hint = "Можно `|` для принудительного переноса."
        
        await send_with_retry(lambda: query.message.reply_text(
            f"✏️ **Введите текст для рендера:**\n\n{hint}",
            parse_mode='Markdown'
        ))
    
    elif query.data == "mode_full":
        user_states[user_id]['mode'] = 'full'
        
        keyboard = [
            [
                InlineKeyboardButton("1️⃣ ЛОГО", callback_data="submode_1"),
                InlineKeyboardButton("2️⃣ ТЕКСТ", callback_data="submode_2"),
                InlineKeyboardButton("3️⃣ КОНТЕНТ", callback_data="submode_3")
            ],
            [InlineKeyboardButton("◀️ Назад", callback_data="back_to_start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await send_with_retry(lambda: query.edit_message_text(
            "✅ **Режим: ПОЛНЫЙ ЦИКЛ**\n\n"
            "Выберите подрежим:\n\n"
            "**1️⃣ ЛОГО** - Лого + линии + заголовок\n"
            "**2️⃣ ТЕКСТ** - Только заголовок\n"
            "**3️⃣ КОНТЕНТ** - Заголовок + подзаголовок\n\n"
            "После выбора отправьте изображение.",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        ))
    
    elif query.data.startswith("submode_"):
        submode = int(query.data.split("_")[1])
        user_states[user_id]['submode'] = submode
        
        mode_names = {
            1: "ЛОГО (лого + заголовок)",
            2: "ТЕКСТ (только заголовок)",
            3: "КОНТЕНТ (заголовок + подзаголовок)"
        }
        
        mode3_hint = ""
        if submode == 3:
            mode3_hint = (
                "\n\n"
                "📝 **Как писать для режима КОНТЕНТ:**\n"
                "Все строки КРОМЕ последней = ЗАГОЛОВОК (зеленый)\n"
                "Последняя строка = ПОДЗАГОЛОВОК (белый)\n\n"
                "Пример:\n"
                "`Самые дорогие`\n"
                "`творения человечества`\n"
                "`Стоимость $100 млрд.`"
            )
        
        keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="mode_full")]]
        
        await send_with_retry(lambda: query.edit_message_text(
            f"✅ **Выбран режим {submode}: {mode_names[submode]}**\n\n"
            f"Теперь отправьте изображение для обработки.\n\n"
            f"Бот выполнит:\n"
            f"1. OCR → контроль\n"
            f"2. Удаление текста\n"
            f"3. Перевод → контроль\n"
            f"4. Нанесение текста"
            f"{mode3_hint}",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        ))
    
    elif query.data == "next_ocr":
        await handle_ocr_next(update, context)
    
    elif query.data == "edit_ocr":
        await handle_ocr_edit(update, context)
    
    elif query.data == "next_llm":
        await handle_llm_next(update, context)
    
    elif query.data == "edit_llm":
        await handle_llm_edit(update, context)
        
    elif query.data == "rerender_text":
        await handle_rerender_text(update, context)
    
    elif query.data == "change_mode_keep_image":
        keyboard = [
            [
                InlineKeyboardButton("1️⃣ ЛОГО", callback_data="rerender_mode_1"),
                InlineKeyboardButton("2️⃣ ТЕКСТ", callback_data="rerender_mode_2"),
                InlineKeyboardButton("3️⃣ КОНТЕНТ", callback_data="rerender_mode_3")
            ]
        ]
        await send_with_retry(lambda: query.message.reply_text(
            "Выберите новый режим (изображение сохранено):",
            reply_markup=InlineKeyboardMarkup(keyboard)
        ))
    
    elif query.data.startswith("rerender_mode_"):
        submode = int(query.data.split("_")[-1])
        user_states[user_id]['submode'] = submode
        user_states[user_id]['step'] = 'editing_llm'
        user_states[user_id]['llm_title'] = ''
        user_states[user_id]['llm_subtitle'] = ''
        
        if submode == 3:
            hint = (
                "Все строки КРОМЕ последней → ЗАГОЛОВОК (зеленый)\n"
                "Последняя строка → ПОДЗАГОЛОВОК (белый)\n\n"
                "Можно `|` для переноса."
            )
        else:
            hint = "Можно `|` для принудительного переноса."
        
        await send_with_retry(lambda: query.message.reply_text(
            f"✏️ **Введите текст для рендера (режим {submode}):**\n\n{hint}",
            parse_mode='Markdown'
        ))
        
    elif query.data == "finish_render":
        await handle_finish_render(update, context)

    elif query.data == "add_text_after_remove":
        await handle_add_text_after_remove(update, context)


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_states or user_states[user_id].get('mode') is None:
        await send_with_retry(lambda: update.message.reply_text("⚠️ Сначала выберите режим командой /start"))
        return
    
    mode = user_states[user_id]['mode']
    submode = user_states[user_id].get('submode')
    
    if mode == 'full' and submode is None:
        await send_with_retry(lambda: update.message.reply_text("⚠️ Сначала выберите подрежим (1/2/3)"))
        return
    
    image_bytes = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            photo = await update.message.photo[-1].get_file()
            image_bytes = await photo.download_as_bytearray()
            break
        except (TimedOut, NetworkError) as e:
            logger.warning(f"⚠️ Попытка {attempt+1}/{RETRY_ATTEMPTS} скачать фото: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            await send_with_retry(lambda: update.message.reply_text("⚠️ Ошибка сети при загрузке фото. Попробуйте ещё раз."))
            return
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки фото: {e}")
            await send_with_retry(lambda: update.message.reply_text(f"❌ Ошибка: {str(e)}"))
            return
    
    if image_bytes is None:
        return
    
    try:
        logger.info(f"✅ Изображение от пользователя {user_id}, режим: {mode}, подрежим: {submode}")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mode == 'remove':
            await process_remove_mode(update, image)
        
        elif mode == 'full':
            await process_full_mode_step1(update, image, submode, user_id)
    
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        await send_with_retry(lambda: update.message.reply_text(f"❌ Ошибка: {str(e)}"))


async def process_remove_mode(update: Update, image: np.ndarray):
    user_id = update.effective_user.id
    status_msg = await send_with_retry(lambda: update.message.reply_text("⏳ Удаление текста (~20-40 сек)..."))
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_start = int(height * (1 - MASK_BOTTOM_MODE2 / 100))
    mask[mask_start:, :] = 255
    
    result = flux_kontext_inpaint(image, mask)
    
    clean_path = f"{TEMP_DIR}/{user_id}_clean.pkl"
    with open(clean_path, 'wb') as f:
        pickle.dump(result, f)
    
    user_states[user_id]['clean_path'] = clean_path
    user_states[user_id]['step'] = 'post_remove'
    
    result_enhanced = enhance_image(result)
    success, buffer = cv2.imencode('.png', result_enhanced)
    if success:
        await send_with_retry(lambda: update.message.reply_photo(
            photo=BytesIO(buffer.tobytes()),
            caption="✅ **Текст удалён\\!**",
            parse_mode='MarkdownV2'
        ))
    
    try:
        await status_msg.delete()
    except:
        pass
    
    keyboard = [
        [
            InlineKeyboardButton("➕ Добавить текст", callback_data="add_text_after_remove"),
            InlineKeyboardButton("✅ Готово", callback_data="finish_render")
        ]
    ]
    await send_with_retry(lambda: update.message.reply_text(
        "Что дальше?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    ))


async def process_full_mode_step1(update: Update, image: np.ndarray, submode: int, user_id: int):
    status_msg = await send_with_retry(lambda: update.message.reply_text("⏳ **Шаг 1/4:** OCR...", parse_mode='Markdown'))
    
    ocr = google_vision_ocr(image, crop_bottom_percent=OCR_BOTTOM_PERCENT)
    
    if not ocr["text"]:
        await send_with_retry(lambda: update.message.reply_text("⚠️ Текст не обнаружен"))
        try:
            await status_msg.delete()
        except:
            pass
        return
    
    ocr_text = ocr["text"]
    ocr_preview = escape_md(ocr_text[:300] + "..." if len(ocr_text) > 300 else ocr_text)
    
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
    
    await send_with_retry(lambda: update.message.reply_text(
        f"📝 **OCR распознал:**\n\n{ocr_preview}\n\n"
        f"Выберите действие:",
        reply_markup=reply_markup,
        parse_mode='MarkdownV2'
    ))
    
    try:
        await status_msg.delete()
    except:
        pass


async def handle_ocr_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    user_states[user_id]['step'] = 'editing_ocr'
    
    await send_with_retry(lambda: query.edit_message_text(
        "✏️ **Отправьте исправленный текст**\n\n"
        "Пришлите текст который должен быть переведён.",
        parse_mode='Markdown'
    ))


async def handle_ocr_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    state = user_states[user_id]
    ocr_text = state['ocr_text']
    
    preview = escape_md(ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text)
    
    await send_with_retry(lambda: query.edit_message_text(
        f"✅ **OCR текст принят**\n\n{preview}",
        parse_mode='MarkdownV2'
    ))
    
    await process_full_mode_step2(query, user_id, ocr_text)


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        return
    
    step = user_states[user_id].get('step')
    
    if step == 'editing_ocr':
        custom_text = update.message.text.strip()
        user_states[user_id]['ocr_text'] = custom_text
        
        await send_with_retry(lambda: update.message.reply_text(
            f"✅ **Текст обновлён**\n\n{custom_text[:200]}...",
            parse_mode='Markdown'
        ))
        
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
        
        await send_with_retry(lambda: update.message.reply_text(
            f"✅ **Текст принят**\n\n{custom_translation[:200]}...",
            parse_mode='Markdown'
        ))
        
        await process_full_mode_step3(update, user_id)


async def process_full_mode_step2(update, user_id: int, ocr_text: str):
    msg_target = _pick_msg_target(update)
    if msg_target is None:
        logger.error("❌ step2: msg_target is None")
        return

    status_msg = await send_with_retry(lambda: msg_target.reply_text(
        "⏳ **Шаг 2/4:** Удаление текста... 🔄",
        parse_mode='Markdown'
    ))

    state = user_states[user_id]
    image_path = state['image_path']
    submode = state['submode']

    with open(image_path, 'rb') as f:
        image = pickle.load(f)

    h, w = image.shape[:2]

    if submode == 1:
        mask_percent = MASK_BOTTOM_MODE1
    elif submode == 2:
        mask_percent = MASK_BOTTOM_MODE2
    else:
        mask_percent = MASK_BOTTOM_MODE3

    mask = np.zeros((h, w), dtype=np.uint8)
    mask_start = int(h * (1 - mask_percent / 100))
    mask[mask_start:, :] = 255

    clean_image = flux_inpaint(image, mask)

    clean_path = f"{TEMP_DIR}/{user_id}_clean.pkl"
    with open(clean_path, 'wb') as f:
        pickle.dump(clean_image, f)
    user_states[user_id]['clean_path'] = clean_path

    preview_bgr = enhance_image(clean_image)
    success, buf = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if success:
        await send_with_retry(lambda: msg_target.reply_photo(
            photo=BytesIO(buf.tobytes()),
            caption="🧹 Текст удалён"
        ))

    try:
        await status_msg.edit_text("⏳ **Шаг 3/4:** Перевод... 🌐", parse_mode='Markdown')
    except:
        pass

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

    llm_preview_escaped = escape_md(llm_preview)

    user_states[user_id]['step'] = 'waiting_llm_decision'

    keyboard = [
        [
            InlineKeyboardButton("✏️ Править", callback_data="edit_llm"),
            InlineKeyboardButton("➡️ Далее", callback_data="next_llm")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await send_with_retry(lambda: msg_target.reply_text(
        f"🌐 **LLM перевёл:**\n\n{llm_preview_escaped}\n\n"
        f"Выберите действие:",
        reply_markup=reply_markup,
        parse_mode='MarkdownV2'
    ))


async def handle_llm_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    user_states[user_id]['step'] = 'editing_llm'
    submode = user_states[user_id]['submode']

    if submode == 3:
        hint = (
            "**Как писать:**\n"
            "Все строки КРОМЕ последней → ЗАГОЛОВОК (зеленый)\n"
            "Последняя строка → ПОДЗАГОЛОВОК (белый)\n\n"
            "Пример:\n"
            "`Портфель Ambani`\n"
            "`Недвижимость на $50 млрд.`"
        )
    else:
        hint = "Пришлите текст для заголовка"

    msg_target = _pick_msg_target(update)
    if msg_target is None:
        logger.error("❌ handle_llm_edit: msg_target is None")
        return

    await send_with_retry(lambda: msg_target.reply_text(
        f"✏️ **Отправьте исправленный перевод**\n\n{hint}",
        parse_mode='Markdown'
    ))


async def handle_llm_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    state = user_states[user_id]
    llm_title = state['llm_title']
    llm_subtitle = state.get('llm_subtitle', '')

    preview = f"{llm_title}\n{llm_subtitle}" if llm_subtitle else llm_title

    msg_target = _pick_msg_target(update)
    if msg_target is None:
        logger.error("❌ handle_llm_next: msg_target is None")
        return

    await send_with_retry(lambda: msg_target.reply_text(
        f"✅ **Перевод принят**\n\n{preview[:200]}...",
        parse_mode='Markdown'
    ))

    await process_full_mode_step3(update, user_id)
    

async def process_full_mode_step3(update, user_id: int):
    msg_target = _pick_msg_target(update)
    if msg_target is None:
        logger.error("❌ step3: msg_target is None")
        return

    status_msg = await send_with_retry(lambda: msg_target.reply_text(
        "⏳ **Шаг 4/4:** Рендер... 🎨",
        parse_mode='Markdown'
    ))

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
        grad = create_gradient_layer(pil.size[0], pil.size[1], gradient_height_percent=GRADIENT_HEIGHT_MODE3)
    else:
        grad = create_gradient_layer(pil.size[0], pil.size[1], gradient_height_percent=GRADIENT_HEIGHT_MODE12)

    pil = PILImage.alpha_composite(pil, grad)

    if submode == 1:
        pil = render_mode1_logo(pil, llm_title)
    elif submode == 2:
        pil = render_mode2_text(pil, llm_title)
    elif submode == 3:
        pil = render_mode3_content(pil, llm_title, llm_subtitle)

    out_rgb = np.array(pil.convert("RGB"))
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = enhance_image(out_bgr)

    success, buffer = cv2.imencode('.png', out_bgr)
    if success:
        mode_names = {1: "ЛОГО", 2: "ТЕКСТ", 3: "КОНТЕНТ"}

        await send_with_retry(lambda: msg_target.reply_photo(
            photo=BytesIO(buffer.tobytes()),
            caption=(
                f"✅ **Готово! (Режим {submode}: {mode_names[submode]})**\n\n"
                f"🎨 LaMa → Градиент → Рендер"
            ),
            parse_mode='Markdown'
        ))

    try:
        await status_msg.delete()
    except:
        pass

    user_states[user_id]['step'] = 'post_render'

    keyboard = [
        [
            InlineKeyboardButton("🔁 Перерендерить", callback_data="rerender_text"),
            InlineKeyboardButton("🔄 Другой режим", callback_data="change_mode_keep_image"),
        ],
        [InlineKeyboardButton("✅ Завершить", callback_data="finish_render")]
    ]
    await send_with_retry(lambda: msg_target.reply_text(
        "Что дальше?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    ))


async def handle_rerender_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    if user_id not in user_states:
        return

    user_states[user_id]['step'] = 'editing_llm'

    submode = user_states[user_id].get('submode')

    if submode == 3:
        hint = (
            "**Как писать:**\n"
            "Все строки КРОМЕ последней → ЗАГОЛОВОК (зеленый)\n"
            "Последняя строка → ПОДЗАГОЛОВОК (белый)\n\n"
            "Можно использовать `|` для принудительного переноса.\n"
            "Пример: `ПРОИСХОДИТ|В МИРЕ`"
        )
    else:
        hint = "Пришлите новый текст заголовка (можно `|` для переноса)."

    await send_with_retry(lambda: query.message.reply_text(
        f"✏️ **Перерендер текста**\n\n{hint}",
        parse_mode='Markdown'
    ))


async def handle_finish_render(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    if user_id not in user_states:
        return

    _cleanup_user_files(user_id)

    user_states[user_id]['step'] = None

    await send_with_retry(lambda: query.message.reply_text(
        "✅ **Готово. Сессия закрыта, временные файлы очищены.**", 
        parse_mode='Markdown'
    ))


async def handle_add_text_after_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    keyboard = [
        [
            InlineKeyboardButton("1️⃣ ЛОГО", callback_data="render_mode_1"),
            InlineKeyboardButton("2️⃣ ТЕКСТ", callback_data="render_mode_2"),
            InlineKeyboardButton("3️⃣ КОНТЕНТ", callback_data="render_mode_3")
        ]
    ]
    await send_with_retry(lambda: query.message.reply_text(
        "Выберите режим рендера:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    ))


def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN не установлен!")
        return
    
    logger.info("🚀 Запуск бота...")
    
    request = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=120.0,
        write_timeout=120.0,
        pool_timeout=120.0,
        connection_pool_size=8
    )
    application = Application.builder().token(TELEGRAM_TOKEN).request(request).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(mode_callback))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))
    application.add_error_handler(on_error)
    
    logger.info("✅ Бот запущен!")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, poll_interval=1.0, timeout=30)


def run_with_retry(max_retries=10, base_delay=5):
    for attempt in range(max_retries):
        try:
            main()
            break
        except Exception as e:
            err_str = str(e)
            if "Timed out" in err_str or "ConnectTimeout" in err_str or "NetworkError" in err_str:
                delay = base_delay * (2 ** min(attempt, 5))
                logger.warning(f"⚠️ Ошибка подключения (попытка {attempt+1}/{max_retries}): {e}")
                logger.info(f"⏳ Повтор через {delay} сек...")
                time.sleep(delay)
            else:
                logger.error(f"❌ Критическая ошибка: {e}")
                raise


if __name__ == '__main__':
    run_with_retry()
