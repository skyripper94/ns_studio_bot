# lama_integration.py

"""
==============================================
НАСТРОЙКИ ДЛЯ БЫСТРОЙ ПРАВКИ
==============================================
"""

# ============== API КЛЮЧИ ==============
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')
GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# ============== ЦВЕТА ==============
COLOR_TURQUOISE = (0, 206, 209)      # Бирюзовый для заголовков
COLOR_WHITE = (255, 255, 255)        # Белый для подзаголовков/лого
COLOR_OUTLINE = (60, 60, 60)         # Обводка текста (#3C3C3C)

# ============== РАЗМЕРЫ ШРИФТОВ ==============
FONT_SIZE_MODE1 = 48          # Заголовок в режиме "Лого"
FONT_SIZE_MODE2 = 46          # Заголовок в режиме "Текст"
FONT_SIZE_MODE3_TITLE = 44    # Заголовок в режиме "Контент"
FONT_SIZE_MODE3_SUBTITLE = 40 # Подзаголовок в режиме "Контент"
FONT_SIZE_LOGO = 20           # Размер @neurostep.media
FONT_SIZE_MIN = 36            # Минимальный размер при автоподборе

# ============== ОТСТУПЫ И РАССТОЯНИЯ ==============
SPACING_BOTTOM = 140              # Отступ снизу до текста
SPACING_LOGO_TO_TITLE = 4         # Между логотипом и заголовком
SPACING_TITLE_TO_SUBTITLE = 10    # Между заголовком и подзаголовком
LINE_SPACING = 32                 # Между строками текста
LOGO_LINE_LENGTH = 300            # Длина линий возле лого

# ============== МАСКА И ГРАДИЕНТ ==============
MASK_BOTTOM_PERCENT = 35          # Сколько % снизу удаляет FLUX (35% = нижняя треть)
GRADIENT_START_PERCENT = 55       # Откуда начинается градиент (55% = чуть выше середины)
GRADIENT_INTENSITY_CURVE = 1.2    # Кривая интенсивности (больше = резче переход, меньше = плавнее)

# ============== РАСТЯЖЕНИЕ ТЕКСТА ==============
TEXT_STRETCH_HEIGHT = 1.25        # Растяжение текста по высоте (1.25 = +25%)
TEXT_STRETCH_WIDTH = 1.10         # Растяжение текста по ширине (1.10 = +10%)

# ============== ТЕНИ И ОБВОДКИ ==============
TEXT_SHADOW_OFFSET = 2            # Смещение тени (больше = дальше тень)
TEXT_OUTLINE_THICKNESS = 1        # Толщина обводки (увеличить для жирнее)

# ============== РАЗМЕРЫ И КАЧЕСТВО ==============
TEXT_WIDTH_PERCENT = 0.9          # Ширина текстового блока от ширины картинки (0.9 = 90%)
OCR_BOTTOM_PERCENT = 35           # Область OCR снизу (должна совпадать с MASK_BOTTOM_PERCENT)

# ============== FLUX ПАРАМЕТРЫ ==============
FLUX_NUM_STEPS = 50               # Количество шагов FLUX (больше = качественнее, но медленнее)
FLUX_GO_FAST = False              # Быстрый режим FLUX (True = быстрее но хуже качество)

# ============== ПУТЬ К ШРИФТУ ==============
FONT_PATH = '/app/fonts/WaffleSoft.otf'

"""
==============================================
"""

import os
import logging
import numpy as np
import cv2
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import openai

logger = logging.getLogger(__name__)

REPLICATE_MODEL = 'black-forest-labs/flux-kontext-pro'
openai.api_key = OPENAI_API_KEY


def google_vision_ocr(image: np.ndarray, crop_bottom_percent: int = OCR_BOTTOM_PERCENT) -> dict:
    """
    OCR через Google Vision API на нижней части изображения
    Возвращает: {'text': полный текст, 'lines': список строк}
    """
    if not GOOGLE_VISION_API_KEY:
        logger.warning("⚠️ GOOGLE_VISION_API_KEY не установлен")
        return {'text': '', 'lines': []}
    
    try:
        # Обрезаем нижнюю часть для OCR
        height, width = image.shape[:2]
        crop_start = int(height * (1 - crop_bottom_percent / 100))
        cropped = image[crop_start:, :]
        
        logger.info(f"🔍 OCR на {crop_bottom_percent}% снизу (строки {crop_start}-{height})")
        
        # Конвертируем в RGB и кодируем в base64
        image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Запрос к Google Vision API
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        payload = {
            "requests": [{
                "image": {"content": image_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }
        
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        
        if 'responses' not in result or not result['responses']:
            logger.warning("⚠️ Нет результатов OCR")
            return {'text': '', 'lines': []}
        
        response_data = result['responses'][0]
        
        if 'textAnnotations' not in response_data:
            logger.warning("⚠️ Текст не обнаружен")
            return {'text': '', 'lines': []}
        
        annotations = response_data['textAnnotations']
        full_text = annotations[0]['description']
        logger.info(f"📝 Распознан текст: {full_text}")
        
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        return {'text': full_text, 'lines': lines}
        
    except Exception as e:
        logger.error(f"❌ Ошибка Google Vision OCR: {e}")
        return {'text': '', 'lines': []}


def openai_translate(text: str, context: str = "") -> str:
    """
    Перевод и адаптация текста через OpenAI GPT-4
    Адаптирует для русскоязычной аудитории (не дословный перевод!)
    """
    if not OPENAI_API_KEY or not text:
        logger.warning("⚠️ OPENAI_API_KEY не установлен или нет текста")
        return text
    
    try:
        logger.info(f"🌐 Перевод: {text}")
        
        system_prompt = """Ты профессиональный переводчик для русскоязычной (СНГ) аудитории.

Правила перевода:
1. Названия брендов оставляй на английском (SpaceX, Tesla, Apple и т.д.)
2. Адаптируй под естественный русский язык, не переводи дословно
3. Используй короткие синонимы вместо длинных слов
4. Сокращай валюту: "billion" → "млрд.", "million" → "млн."
5. Делай текст живым и понятным для СНГ
6. Возвращай ТОЛЬКО переведённый текст, без пояснений

Примеры:
"The Most Expensive Things Humans Have Ever Created" → "Самые дорогие творения человечества"
"SpaceX Starlink Satellite Constellation" → "Спутниковая сеть SpaceX Starlink"
"$10 billion" → "$10 млрд."
"We Share Insights That Expand Your View" → "Делимся знаниями, расширяющими кругозор"
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Переведи и адаптируй: {text}"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        translated = response.choices[0].message.content.strip()
        logger.info(f"✅ Переведено: {translated}")
        
        return translated
        
    except Exception as e:
        logger.error(f"❌ Ошибка OpenAI перевода: {e}")
        return text


def opencv_fallback(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Запасной вариант через OpenCV (если FLUX не работает)
    Использует 2 алгоритма: NS и TELEA
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    logger.info("✅ OpenCV fallback inpainting")
    return result


def flux_kontext_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    FLUX Kontext Pro - удаление содержимого ТОЛЬКО в области маски
    ВАЖНО: FLUX работает ТОЛЬКО внутри маски, не трогает области вне маски!
    
    Параметры:
    - num_inference_steps: увеличить для лучшего качества (FLUX_NUM_STEPS)
    - go_fast: True = быстрее но хуже качество (FLUX_GO_FAST)
    """
    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN не установлен, используем OpenCV")
        return opencv_fallback(image, mask)
    
    try:
        import replicate
        
        logger.info("🚀 FLUX - удаление в области маски")
        
        # Конвертируем полное изображение в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Подготовка маски
        pil_mask = Image.fromarray(mask)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # СТРОГИЙ промпт: работать ТОЛЬКО в маске
        prompt = "Seamlessly remove all text, decorative lines and logos ONLY in the masked region. Restore natural background without blur. Keep everything outside the mask completely unchanged and untouched."
        
        logger.info("📤 Отправка в FLUX...")
        
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "input_image": img_buffer,
                "mask": mask_buffer,
                "output_format": "png",
                "go_fast": FLUX_GO_FAST,
                "num_inference_steps": FLUX_NUM_STEPS
            }
        )
        
        # Получение результата
        if hasattr(output, 'read'):
            result_bytes = output.read()
        elif isinstance(output, str):
            response = requests.get(output, timeout=60)
            result_bytes = response.content
        elif isinstance(output, list) and len(output) > 0:
            response = requests.get(output[0], timeout=60)
            result_bytes = response.content
        else:
            logger.error(f"❌ Неизвестный формат вывода: {type(output)}")
            return opencv_fallback(image, mask)
        
        result_pil = Image.open(BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert('RGB'))
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # Проверка и коррекция размера (если FLUX изменил размер)
        if result_bgr.shape[:2] != image.shape[:2]:
            logger.warning(f"⚠️ FLUX изменил размер, возвращаем обратно")
            result_bgr = cv2.resize(result_bgr, (image.shape[1], image.shape[0]), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        logger.info("✅ FLUX завершил работу!")
        return result_bgr
        
    except Exception as e:
        logger.error(f"❌ Ошибка FLUX: {e}")
        return opencv_fallback(image, mask)


def create_gradient_layer(width: int, height: int, start_percent: int = GRADIENT_START_PERCENT) -> Image.Image:
    """
    Создание градиентного слоя (RGBA)
    
    Градиент: 
    - Низ (100%): непрозрачный черный (alpha=255)
    - Середина (~50%): начинает интенсивно светлеть
    - Верх (0%): полностью прозрачный (alpha=0)
    
    Настройка интенсивности: GRADIENT_INTENSITY_CURVE (больше = резче переход)
    """
    gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    start_row = int(height * (1 - start_percent / 100))
    gradient_height = height - start_row
    
    for y in range(height):
        if y >= start_row:
            # Инвертированный прогресс: 1.0 снизу → 0.0 сверху
            progress = 1.0 - (y - start_row) / gradient_height
            
            # Применяем кривую для интенсивности (больше GRADIENT_INTENSITY_CURVE = резче)
            alpha = int(255 * (progress ** GRADIENT_INTENSITY_CURVE))
            
            for x in range(width):
                gradient.putpixel((x, y), (0, 0, 0, alpha))
    
    logger.info(f"✨ Градиент создан от строки {start_row} ({start_percent}%)")
    return gradient


def calculate_adaptive_font_size(text: str, font_path: str, max_width: int, 
                                  initial_size: int, min_size: int = FONT_SIZE_MIN) -> tuple:
    """
    Автоподбор размера шрифта под ширину
    Возвращает: (размер_шрифта, объект_шрифта, список_строк)
    
    Уменьшение min_size даст меньший минимальный шрифт
    """
    font_size = initial_size
    
    while font_size >= min_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            # Разбиваем текст на строки с учетом ширины
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = font.getbbox(test_line)
                width = bbox[2] - bbox[0]
                
                if width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Проверяем что все строки влезают
            fits = all(
                font.getbbox(line)[2] - font.getbbox(line)[0] <= max_width
                for line in lines
            )
            
            if fits:
                return font_size, font, lines
            
        except Exception as e:
            logger.error(f"Ошибка шрифта размера {font_size}: {e}")
        
        font_size -= 2
    
    # Крайний случай
    font = ImageFont.truetype(font_path, min_size)
    return min_size, font, [text]


def draw_text_with_stretch(draw: ImageDraw.Draw, x: int, y: int, 
                           text: str, font: ImageFont.FreeTypeFont,
                           fill_color: tuple, outline_color: tuple,
                           stretch_width: float = TEXT_STRETCH_WIDTH,
                           stretch_height: float = TEXT_STRETCH_HEIGHT,
                           shadow_offset: int = TEXT_SHADOW_OFFSET) -> int:
    """
    Отрисовка текста с растяжением, тенью и обводкой
    
    Растяжение:
    - stretch_width: коэффициент ширины (1.10 = +10%)
    - stretch_height: коэффициент высоты (1.25 = +25%)
    
    Возвращает: высоту нарисованного текста
    """
    # Получаем размеры оригинального текста
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Создаем временное изображение для текста
    temp_width = int(text_width * 1.5)
    temp_height = int(text_height * 2)
    temp_image = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_image)
    
    # Рисуем текст в центре временного изображения
    temp_x = (temp_width - text_width) // 2
    temp_y = (temp_height - text_height) // 2
    
    # Тень
    temp_draw.text((temp_x + shadow_offset, temp_y + shadow_offset), 
                   text, font=font, fill=(0, 0, 0, 128))
    
    # Обводка (8 направлений, толщина контролируется TEXT_OUTLINE_THICKNESS)
    for thickness in range(TEXT_OUTLINE_THICKNESS):
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            temp_draw.text((temp_x + dx*(thickness+1), temp_y + dy*(thickness+1)), 
                          text, font=font, fill=outline_color)
    
    # Основной текст
    temp_draw.text((temp_x, temp_y), text, font=font, fill=fill_color)
    
    # Вырезаем только область с текстом
    text_bbox = temp_image.getbbox()
    if text_bbox:
        text_crop = temp_image.crop(text_bbox)
        
        # Применяем растяжение
        stretched_width = int(text_crop.width * stretch_width)
        stretched_height = int(text_crop.height * stretch_height)
        stretched_text = text_crop.resize((stretched_width, stretched_height), 
                                         Image.LANCZOS)
        
        # Вставляем растянутый текст
        final_x = x - (stretched_width - text_width) // 2
        final_y = y - (stretched_height - text_height) // 2
        
        # Получаем базовое изображение из draw
        base_image = draw._image
        base_image.paste(stretched_text, (final_x, final_y), stretched_text)
        
        return stretched_height
    
    return text_height


def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Режим 1: Лого + 2 линии + Заголовок (UPPERCASE)
    
    Элементы:
    - Логотип @neurostep.media по центру
    - Две горизонтальные линии слева и справа от лого
    - Заголовок снизу (бирюзовый, с растяжением)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    title_translated = title_translated.upper()
    
    # Подбор размера заголовка
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE1
    )
    
    # Расчет высот заголовка
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        title_heights.append(int((bbox[3] - bbox[1]) * TEXT_STRETCH_HEIGHT))
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    
    # Лого
    logo_font = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOGO)
    logo_text = "@neurostep.media"
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_height = logo_bbox[3] - logo_bbox[1]
    
    # Общая высота композиции
    total_height = logo_height + SPACING_LOGO_TO_TITLE + total_title_height
    start_y = height - SPACING_BOTTOM - total_height
    
    # Отрисовка лого
    logo_x = (width - logo_width) // 2
    logo_y = start_y
    
    # Линии возле лого (горизонтальные, на уровне центра лого)
    line_y = logo_y + logo_height // 2
    line_left_start = logo_x - LOGO_LINE_LENGTH - 10
    line_right_start = logo_x + logo_width + 10
    
    draw.line([(line_left_start, line_y), (line_left_start + LOGO_LINE_LENGTH, line_y)],
              fill=COLOR_TURQUOISE, width=1)
    draw.line([(line_right_start, line_y), (line_right_start + LOGO_LINE_LENGTH, line_y)],
              fill=COLOR_TURQUOISE, width=1)
    
    # Текст лого (белый, без растяжения)
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)
    
    # Отрисовка заголовка (с растяжением)
    title_y = start_y + logo_height + SPACING_LOGO_TO_TITLE
    
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - int(line_width * TEXT_STRETCH_WIDTH)) // 2
        
        drawn_height = draw_text_with_stretch(
            draw, line_x, title_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE
        )
        
        title_y += drawn_height + LINE_SPACING
    
    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Режим 2: Только заголовок (UPPERCASE)
    
    Элементы:
    - Заголовок по центру (бирюзовый, с растяжением)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    title_translated = title_translated.upper()
    
    # Подбор размера
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE2
    )
    
    # Расчет высот
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        title_heights.append(int((bbox[3] - bbox[1]) * TEXT_STRETCH_HEIGHT))
    
    total_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    start_y = height - SPACING_BOTTOM - total_height
    
    # Отрисовка заголовка
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - int(line_width * TEXT_STRETCH_WIDTH)) // 2
        
        drawn_height = draw_text_with_stretch(
            draw, line_x, current_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE
        )
        
        current_y += drawn_height + LINE_SPACING
    
    return image


def render_mode3_content(image: Image.Image, title_translated: str, 
                         subtitle_translated: str) -> Image.Image:
    """
    Режим 3: Заголовок + Подзаголовок (ОБА UPPERCASE)
    
    Элементы:
    - Заголовок (бирюзовый, с растяжением)
    - Подзаголовок (белый, с растяжением, меньше размером)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    title_translated = title_translated.upper()
    subtitle_translated = subtitle_translated.upper()
    
    # Подбор размера заголовка
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE3_TITLE
    )
    
    # Подбор размера подзаголовка (пропорционально меньше)
    subtitle_initial_size = int(title_size * 0.8)
    subtitle_size, subtitle_font, subtitle_lines = calculate_adaptive_font_size(
        subtitle_translated, FONT_PATH, max_text_width, subtitle_initial_size
    )
    
    # Расчет высот
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        title_heights.append(int((bbox[3] - bbox[1]) * TEXT_STRETCH_HEIGHT))
    
    subtitle_heights = []
    for line in subtitle_lines:
        bbox = subtitle_font.getbbox(line)
        subtitle_heights.append(int((bbox[3] - bbox[1]) * TEXT_STRETCH_HEIGHT))
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    total_subtitle_height = sum(subtitle_heights) + (len(subtitle_lines) - 1) * LINE_SPACING
    
    total_height = total_title_height + SPACING_TITLE_TO_SUBTITLE + total_subtitle_height
    start_y = height - SPACING_BOTTOM - total_height
    
    # Отрисовка заголовка (бирюзовый)
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - int(line_width * TEXT_STRETCH_WIDTH)) // 2
        
        drawn_height = draw_text_with_stretch(
            draw, line_x, current_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE
        )
        
        current_y += drawn_height + LINE_SPACING
    
    # Отрисовка подзаголовка (белый)
    current_y += SPACING_TITLE_TO_SUBTITLE
    
    for i, line in enumerate(subtitle_lines):
        line_bbox = subtitle_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - int(line_width * TEXT_STRETCH_WIDTH)) // 2
        
        drawn_height = draw_text_with_stretch(
            draw, line_x, current_y, line, subtitle_font,
            COLOR_WHITE, COLOR_OUTLINE
        )
        
        current_y += drawn_height + LINE_SPACING
    
    return image


def process_full_workflow(image: np.ndarray, mode: int) -> tuple:
    """
    Полный workflow для режимов 1, 2, 3
    
    ЛОГИКА:
    1. OCR (Google Vision) → извлекаем текст для перевода
    2. МАСКА = нижние 35% (ВСЕГДА) → FLUX удаляет ВСЁ (текст, линии, лого)
    3. Перевод текста (OpenAI GPT-4)
    4. Наложение градиентного слоя поверх чистого изображения
    5. Отрисовка текста поверх градиента
    
    Режимы:
    - mode=1: Лого + заголовок
    - mode=2: Только заголовок
    - mode=3: Заголовок + подзаголовок
    
    Возвращает: (результат_изображение, ocr_данные)
    """
    logger.info("=" * 60)
    logger.info(f"🚀 ПОЛНЫЙ WORKFLOW - РЕЖИМ {mode}")
    logger.info("=" * 60)
    
    height, width = image.shape[:2]
    
    # ========================================
    # ШАГ 1: OCR (для извлечения текста)
    # ========================================
    logger.info("📋 ШАГ 1: OCR (Google Vision)")
    ocr_data = google_vision_ocr(image, crop_bottom_percent=OCR_BOTTOM_PERCENT)
    
    if not ocr_data['text']:
        logger.warning("⚠️ Текст не обнаружен")
        return image, ocr_data
    
    # ========================================
    # ШАГ 2: Создание маски = нижние 35%
    # Удаляет ВСЁ: текст, линии, лого, градиент
    # ========================================
    logger.info("📋 ШАГ 2: Создание маски (нижние 35%)")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_start = int(height * (1 - MASK_BOTTOM_PERCENT / 100))
    mask[mask_start:, :] = 255
    
    logger.info(f"📐 Маска: строки {mask_start}-{height} (нижние {MASK_BOTTOM_PERCENT}%)")
    
    # ========================================
    # ШАГ 3: FLUX удаляет всё в маске
    # ========================================
    logger.info("📋 ШАГ 3: Удаление содержимого (FLUX Kontext Pro)")
    clean_image = flux_kontext_inpaint(image, mask)
    
    # ========================================
    # ШАГ 4: Перевод текста
    # ========================================
    logger.info("📋 ШАГ 4: Перевод (OpenAI GPT-4)")
    
    if mode == 3:
        # Режим 3: заголовок + подзаголовок
        lines = ocr_data['lines']
        if len(lines) >= 2:
            title = ' '.join(lines[:-1])  # Все строки кроме последней
            subtitle = lines[-1]          # Последняя строка
        else:
            title = ocr_data['text']
            subtitle = ""
        
        title_translated = openai_translate(title)
        subtitle_translated = openai_translate(subtitle) if subtitle else ""
    else:
        # Режимы 1 и 2: только заголовок
        title_translated = openai_translate(ocr_data['text'])
        subtitle_translated = ""
    
    # ========================================
    # ШАГ 5: Конвертация в PIL и наложение градиента
    # ========================================
    logger.info("📋 ШАГ 5: Наложение градиентного слоя")
    
    clean_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(clean_rgb).convert('RGBA')
    
    actual_width, actual_height = pil_image.size
    logger.info(f"📐 Размер изображения: {actual_width}x{actual_height}")
    
    # Создание градиента как отдельного слоя
    gradient_layer = create_gradient_layer(actual_width, actual_height, 
                                          start_percent=GRADIENT_START_PERCENT)
    
    # Наложение градиента ПОВЕРХ изображения
    pil_image = Image.alpha_composite(pil_image, gradient_layer)
    
    logger.info("✅ Градиент наложен")
    
    # ========================================
    # ШАГ 6: Отрисовка текста ПОВЕРХ градиента
    # ========================================
    logger.info(f"📋 ШАГ 6: Отрисовка текста (Режим {mode})")
    
    if mode == 1:
        pil_image = render_mode1_logo(pil_image, title_translated)
    elif mode == 2:
        pil_image = render_mode2_text(pil_image, title_translated)
    elif mode == 3:
        pil_image = render_mode3_content(pil_image, title_translated, subtitle_translated)
    
    # Конвертация обратно в BGR для OpenCV
    result_rgb = np.array(pil_image.convert('RGB'))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    logger.info("=" * 60)
    logger.info("✅ WORKFLOW ЗАВЕРШЁН!")
    logger.info("=" * 60)
    
    return result_bgr, ocr_data


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Функция совместимости (алиас для flux_kontext_inpaint)"""
    return flux_kontext_inpaint(image, mask)
