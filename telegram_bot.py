#!/usr/bin/env python3
"""
Telegram бот для удаления текста/лого с изображений через LaMa
"""

import os
import logging
from io import BytesIO
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
from PIL import Image
import cv2
import numpy as np

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


class ImageProcessor:
    """Обработчик изображений"""
    
    def __init__(self):
        self.lama_loaded = False
        self.lama_inpainter = None
        
    def load_lama(self):
        """Загружает LaMa модель"""
        if self.lama_loaded:
            return True
        
        try:
            from lama_integration import get_inpainter
            self.lama_inpainter = get_inpainter()
            self.lama_loaded = self.lama_inpainter.load_model()
            return self.lama_loaded
        except Exception as e:
            logger.error(f"Ошибка загрузки LaMa: {e}")
            return False
    
    def create_mask(self, image_path: Path) -> np.ndarray:
        """
        Создает маску для удаления текста/лого в нижней части
        
        Args:
            image_path: путь к изображению
            
        Returns:
            mask: numpy array с маской (255 = удалить, 0 = оставить)
        """
        # Читаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        height, width = img.shape[:2]
        
        # Создаем пустую маску
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
        
        # Расширяем маску для захвата краев
        kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        
        # Размываем для плавности
        text_mask = cv2.GaussianBlur(text_mask, (5, 5), 0)
        
        # Вставляем в полную маску
        mask[roi_start:, :] = text_mask
        
        # Детектим светлые объекты в верхней части нижней зоны (лого с полосками)
        logo_region = img[roi_start:roi_start + int(height * 0.15), :]  # 15% от общей высоты
        hsv_logo = cv2.cvtColor(logo_region, cv2.COLOR_BGR2HSV)
        
        # Ищем светлые области (лого обычно светлые на темном фоне)
        bright_mask = cv2.inRange(hsv_logo, np.array([0, 0, 150]), np.array([180, 255, 255]))
        
        # Морфологические операции
        kernel_small = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_small)
        bright_mask = cv2.dilate(bright_mask, kernel_small, iterations=2)
        
        # Добавляем в маску
        mask[roi_start:roi_start + int(height * 0.15), :] = cv2.bitwise_or(
            mask[roi_start:roi_start + int(height * 0.15), :],
            bright_mask
        )
        
        return mask
    
    def process_with_lama(self, image_path: Path, mask: np.ndarray) -> Path:
        """
        Удаляет области по маске используя LaMa
        
        Args:
            image_path: путь к оригинальному изображению
            mask: маска для удаления
            
        Returns:
            output_path: путь к обработанному изображению
        """
        # Читаем изображение
        img = cv2.imread(str(image_path))
        
        # Используем LaMa inpainter если загружен
        if self.lama_inpainter:
            result = self.lama_inpainter.inpaint(img, mask)
        else:
            # Fallback на OpenCV
            result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # Сохраняем результат
        output_path = TEMP_DIR / f"cleaned_{image_path.name}"
        cv2.imwrite(str(output_path), result)
        
        return output_path
    
    def add_text(self, image_path: Path, main_text: str, secondary_text: str = "") -> Path:
        """
        Добавляет новый текст на изображение
        
        Args:
            image_path: путь к изображению
            main_text: основной текст
            secondary_text: дополнительный текст (опционально)
            
        Returns:
            output_path: путь к изображению с текстом
        """
        from PIL import ImageDraw, ImageFont
        
        # Открываем изображение
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
        
        # Рисуем основной текст с обводкой
        bbox = draw.textbbox((0, 0), main_text, font=main_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        main_x = (width - text_width) // 2
        main_y = height - int(height * 0.20)
        
        # Обводка
        for adj_x in range(-3, 4):
            for adj_y in range(-3, 4):
                draw.text((main_x + adj_x, main_y + adj_y), main_text, font=main_font, fill=(0, 0, 0))
        
        # Основной текст
        draw.text((main_x, main_y), main_text, font=main_font, fill=MAIN_TEXT_COLOR)
        
        # Дополнительный текст если есть
        if secondary_text:
            bbox2 = draw.textbbox((0, 0), secondary_text, font=secondary_font)
            text_width2 = bbox2[2] - bbox2[0]
            
            secondary_x = (width - text_width2) // 2
            secondary_y = main_y + text_height + 10
            
            # Обводка
            for adj_x in range(-2, 3):
                for adj_y in range(-2, 3):
                    draw.text((secondary_x + adj_x, secondary_y + adj_y), secondary_text, 
                             font=secondary_font, fill=(0, 0, 0))
            
            # Текст
            draw.text((secondary_x, secondary_y), secondary_text, 
                     font=secondary_font, fill=SECONDARY_TEXT_COLOR)
        
        # Сохраняем
        output_path = TEMP_DIR / f"with_text_{image_path.name}"
        img = img.convert('RGB')
        img.save(output_path, quality=95)
        
        return output_path


# Глобальный процессор
processor = ImageProcessor()


# ============= КОМАНДЫ БОТА =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🎨 **Image Eraser Bot**

Я удаляю текст и лого с изображений!

**Как пользоваться:**
1️⃣ Отправьте изображение
2️⃣ Я автоматически найду и удалю:
   • Лого сверху
   • Желтые полоски
   • Текст внизу

**Команды:**
/clean - Очистить изображение
/addtext - Добавить свой текст
/help - Помощь

Просто отправьте изображение чтобы начать! 📸
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📖 **Инструкция:**

**Базовое использование:**
1. Отправьте изображение боту
2. Бот автоматически удалит текст/лого
3. Получите чистое изображение

**Дополнительные команды:**

`/addtext "Ваш текст" "Второй текст"`
Добавляет новый текст на чистое изображение

**Пример:**
`/addtext "WE SHARE INSIGHTS" "$9 BILLION"`

**Настройки:**
Бот обрабатывает нижние 45% изображения
Автоматически находит:
• Желтый/белый текст
• Светлые лого
• Декоративные полоски

**Batch обработка:**
Можете отправить несколько изображений подряд!
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик полученных изображений"""
    
    # Отправляем статус
    status_msg = await update.message.reply_text("⏳ Обрабатываю изображение...")
    
    try:
        # Получаем файл
        photo = update.message.photo[-1]  # Берем самое большое разрешение
        file = await context.bot.get_file(photo.file_id)
        
        # Скачиваем
        image_path = TEMP_DIR / f"{photo.file_id}.jpg"
        await file.download_to_drive(image_path)
        
        logger.info(f"Получено изображение: {image_path}")
        
        # Загружаем LaMa если еще не загружена
        if not processor.lama_loaded:
            await status_msg.edit_text("⏳ Загружаю модель LaMa...")
            processor.load_lama()
        
        # Создаем маску
        await status_msg.edit_text("🔍 Ищу текст и лого...")
        mask = processor.create_mask(image_path)
        
        # Сохраняем маску для preview (опционально)
        mask_path = TEMP_DIR / f"mask_{photo.file_id}.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Обрабатываем через LaMa
        await status_msg.edit_text("🎨 Удаляю текст...")
        cleaned_path = processor.process_with_lama(image_path, mask)
        
        # Отправляем результат
        await status_msg.edit_text("✅ Готово!")
        
        # Создаем короткий ID для callback_data (Telegram лимит 64 байта)
        import uuid
        short_id = str(uuid.uuid4())[:8]  # Используем первые 8 символов UUID
        
        # Создаем кнопки
        keyboard = [
            [
                InlineKeyboardButton("➕ Добавить текст", callback_data=f"addtext_{short_id}"),
                InlineKeyboardButton("👁️ Показать маску", callback_data=f"showmask_{short_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Отправляем чистое изображение
        with open(cleaned_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f,
                caption="✨ Текст удален!\n\nХотите добавить свой текст?",
                reply_markup=reply_markup
            )
        
        # Удаляем статус
        await status_msg.delete()
        
        # Сохраняем пути в контексте с коротким ID
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
        # Показываем маску
        short_id = data.replace("showmask_", "")
        if short_id in context.user_data:
            mask_path = context.user_data[short_id]['mask']
            with open(mask_path, 'rb') as f:
                await query.message.reply_photo(
                    photo=f,
                    caption="👁️ Маска удаления (белое = удалено)"
                )
        else:
            await query.message.reply_text("❌ Данные изображения не найдены. Отправьте изображение снова.")
    
    elif data.startswith("addtext_"):
        # Запрашиваем текст
        short_id = data.replace("addtext_", "")
        # Сохраняем short_id для команды addtext
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
    
    # Парсим аргументы
    text = ' '.join(context.args)
    import re
    texts = re.findall(r'"([^"]*)"', text)
    
    if not texts:
        await update.message.reply_text("❌ Не найден текст в кавычках!")
        return
    
    main_text = texts[0] if len(texts) > 0 else DEFAULT_MAIN_TEXT
    secondary_text = texts[1] if len(texts) > 1 else ""
    
    # Берем current_image_id из контекста
    current_image_id = context.user_data.get('current_image_id')
    
    if not current_image_id or current_image_id not in context.user_data:
        await update.message.reply_text("❌ Сначала отправьте изображение для очистки!")
        return
    
    cleaned_path = context.user_data[current_image_id]['cleaned']
    
    status_msg = await update.message.reply_text("✍️ Добавляю текст...")
    
    try:
        # Добавляем текст
        result_path = processor.add_text(cleaned_path, main_text, secondary_text)
        
        # Отправляем результат
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
    
    # Проверяем токен
    if TELEGRAM_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.error("❌ Установите TELEGRAM_BOT_TOKEN!")
        return
    
    logger.info("🚀 Запуск бота...")
    
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("addtext", addtext_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Запускаем
    logger.info("✅ Бот запущен!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()