#!/usr/bin/env python3
"""
Telegram бот для удаления текста/лого с изображений через Replicate API
"""

import os
import logging
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Импортируем функцию из lama_integration
from lama_integration import replicate_inpaint

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============= КОНФИГУРАЦИЯ =============

# Telegram токен (получите у @BotFather)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')

# Папки для работы
TEMP_DIR = Path('/tmp/bot_images')
TEMP_DIR.mkdir(exist_ok=True)

# Параметры детекции
BOTTOM_PERCENT = 45              # Обрабатываем нижние 45%
DILATE_KERNEL_SIZE = 12          # Расширение маски

# Цвета для детекции
YELLOW_LOWER = np.array([15, 100, 100], dtype=np.uint8)   # HSV
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)
WHITE_LOWER = np.array([0, 0, 200], dtype=np.uint8)
WHITE_UPPER = np.array([180, 30, 255], dtype=np.uint8)

# Настройки нового текста
DEFAULT_MAIN_TEXT = "YOUR TEXT HERE"
DEFAULT_SECONDARY_TEXT = "$100 BILLION"
MAIN_TEXT_COLOR = (0, 150, 255)      # Синий RGB
SECONDARY_TEXT_COLOR = (255, 255, 255)  # Белый RGB

# ========================================


def create_mask(image_path: Path) -> np.ndarray:
    """
    Создает маску для удаления текста/лого в нижней части
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Определяем область для обработки (нижняя часть)
    roi_start = int(height * (100 - BOTTOM_PERCENT) / 100)
    roi = img[roi_start:, :]
    
    # Конвертируем в HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Создаем маски для желтого и белого
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    
    # Объединяем маски
    text_mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    # Находим контуры и заполняем
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(text_mask, contours, -1, 255, -1)
    
    # Расширяем маску
    kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    text_mask = cv2.GaussianBlur(text_mask, (5, 5), 0)
    
    # Вставляем в полную маску
    mask[roi_start:, :] = text_mask
    
    # Детектим светлые объекты (лого)
    logo_region = img[roi_start:roi_start + int(height * 0.15), :]
    hsv_logo = cv2.cvtColor(logo_region, cv2.COLOR_BGR2HSV)
    bright_mask = cv2.inRange(hsv_logo, np.array([0, 0, 150]), np.array([180, 255, 255]))
    
    kernel_small = np.ones((5, 5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_small)
    bright_mask = cv2.dilate(bright_mask, kernel_small, iterations=2)
    
    mask[roi_start:roi_start + int(height * 0.15), :] = cv2.bitwise_or(
        mask[roi_start:roi_start + int(height * 0.15), :],
        bright_mask
    )
    
    return mask


def process_image(image_path: Path, mask: np.ndarray) -> Path:
    """
    Удаляет области по маске используя Replicate API или OpenCV fallback
    """
    img = cv2.imread(str(image_path))
    
    # Используем replicate_inpaint (с автоматическим fallback на OpenCV)
    result = replicate_inpaint(img, mask)
    
    # Сохраняем результат
    output_path = TEMP_DIR / f"cleaned_{image_path.name}"
    cv2.imwrite(str(output_path), result)
    
    return output_path


def add_text(image_path: Path, main_text: str, secondary_text: str = "") -> Path:
    """
    Добавляет новый текст на изображение
    """
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Загружаем шрифт
    try:
        main_font_size = int(height * 0.07)
        main_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", main_font_size)
        secondary_font_size = int(height * 0.035)
        secondary_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", secondary_font_size)
    except:
        main_font = ImageFont.load_default()
        secondary_font = ImageFont.load_default()
    
    # Рисуем основной текст
    bbox = draw.textbbox((0, 0), main_text, font=main_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    main_x = (width - text_width) // 2
    main_y = height - int(height * 0.20)
    
    # Обводка
    for adj_x in range(-3, 4):
        for adj_y in range(-3, 4):
            draw.text((main_x + adj_x, main_y + adj_y), main_text, font=main_font, fill=(0, 0, 0))
    
    draw.text((main_x, main_y), main_text, font=main_font, fill=MAIN_TEXT_COLOR)
    
    # Дополнительный текст
    if secondary_text:
        bbox2 = draw.textbbox((0, 0), secondary_text, font=secondary_font)
        text_width2 = bbox2[2] - bbox2[0]
        secondary_x = (width - text_width2) // 2
        secondary_y = main_y + text_height + 10
        
        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                draw.text((secondary_x + adj_x, secondary_y + adj_y), secondary_text, 
                         font=secondary_font, fill=(0, 0, 0))
        
        draw.text((secondary_x, secondary_y), secondary_text, 
                 font=secondary_font, fill=SECONDARY_TEXT_COLOR)
    
    # Сохраняем
    output_path = TEMP_DIR / f"with_text_{image_path.name}"
    img = img.convert('RGB')
    img.save(output_path, quality=95)
    
    return output_path


# ============= КОМАНДЫ БОТА =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🎨 **Image Eraser Bot**

Я удаляю текст и лого с изображений!

**Как пользоваться:**
1️⃣ Отправьте изображение
2️⃣ Я автоматически найду и удалю текст/лого
3️⃣ Получите чистое изображение

Просто отправьте изображение чтобы начать! 📸
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📖 **Инструкция:**

1. Отправьте изображение боту
2. Бот автоматически удалит текст/лого
3. Получите чистое изображение

**Команда:**
`/addtext "Ваш текст" "Второй текст"`

**Пример:**
`/addtext "WE SHARE INSIGHTS" "$9 BILLION"`
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик полученных изображений"""
    status_msg = await update.message.reply_text("⏳ Обрабатываю изображение...")
    
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        image_path = TEMP_DIR / f"{photo.file_id}.jpg"
        await file.download_to_drive(image_path)
        
        logger.info(f"Получено изображение: {image_path}")
        
        # Создаем маску
        await status_msg.edit_text("🔍 Ищу текст и лого...")
        mask = create_mask(image_path)
        
        mask_path = TEMP_DIR / f"mask_{photo.file_id}.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Обрабатываем (Replicate или OpenCV fallback)
        await status_msg.edit_text("🎨 Удаляю текст...")
        cleaned_path = process_image(image_path, mask)
        
        await status_msg.edit_text("✅ Готово!")
        
        # Короткий ID для кнопок
        import uuid
        short_id = str(uuid.uuid4())[:8]
        
        keyboard = [
            [
                InlineKeyboardButton("➕ Добавить текст", callback_data=f"addtext_{short_id}"),
                InlineKeyboardButton("👁️ Показать маску", callback_data=f"showmask_{short_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        with open(cleaned_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f,
                caption="✨ Текст удален!\n\nХотите добавить свой текст?",
                reply_markup=reply_markup
            )
        
        await status_msg.delete()
        
        context.user_data[short_id] = {
            'original': image_path,
            'cleaned': cleaned_path,
            'mask': mask_path,
            'file_id': photo.file_id
        }
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("showmask_"):
        short_id = data.replace("showmask_", "")
        if short_id in context.user_data:
            mask_path = context.user_data[short_id]['mask']
            with open(mask_path, 'rb') as f:
                await query.message.reply_photo(
                    photo=f,
                    caption="👁️ Маска удаления (белое = удалено)"
                )
        else:
            await query.message.reply_text("❌ Данные изображения не найдены.")
    
    elif data.startswith("addtext_"):
        short_id = data.replace("addtext_", "")
        context.user_data['current_image_id'] = short_id
        
        await query.message.reply_text(
            "✍️ Отправьте текст в формате:\n\n"
            "`/addtext \"Основной текст\" \"Дополнительный\"`\n\n"
            "Например:\n"
            "`/addtext \"WE SHARE INSIGHTS\" \"$9 BILLION\"`",
            parse_mode='Markdown'
        )


async def addtext_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для добавления текста"""
    if not context.args:
        await update.message.reply_text(
            "Используйте формат:\n"
            "`/addtext \"Основной текст\" \"Дополнительный\"`",
            parse_mode='Markdown'
        )
        return
    
    text = ' '.join(context.args)
    import re
    texts = re.findall(r'"([^"]*)"', text)
    
    if not texts:
        await update.message.reply_text("❌ Не найден текст в кавычках!")
        return
    
    main_text = texts[0] if len(texts) > 0 else DEFAULT_MAIN_TEXT
    secondary_text = texts[1] if len(texts) > 1 else ""
    
    current_image_id = context.user_data.get('current_image_id')
    
    if not current_image_id or current_image_id not in context.user_data:
        await update.message.reply_text("❌ Сначала отправьте изображение!")
        return
    
    cleaned_path = context.user_data[current_image_id]['cleaned']
    status_msg = await update.message.reply_text("✍️ Добавляю текст...")
    
    try:
        result_path = add_text(cleaned_path, main_text, secondary_text)
        
        with open(result_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f,
                caption=f"✨ Текст добавлен!\n\n📝 {main_text}\n💰 {secondary_text}"
            )
        
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Ошибка добавления текста: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")


def main():
    """Запуск бота"""
    if TELEGRAM_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.error("❌ Установите TELEGRAM_BOT_TOKEN!")
        return
    
    logger.info("🚀 Запуск бота...")
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("addtext", addtext_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("✅ Бот запущен!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
