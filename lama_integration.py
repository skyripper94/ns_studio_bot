# lama_integration.py

# ============== ИМПОРТЫ ==============
import os
import logging
import base64
from io import BytesIO

import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont

import openai
import re

logger = logging.getLogger(__name__)

"""
==============================================
НАСТРОЙКИ ДЛЯ БЫСТРОЙ РУЧНОЙ ПРАВКИ
==============================================
"""

# ============== API КЛЮЧИ ==============
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ============== REPLICATE / FLUX (INPAINT) ==============
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "black-forest-labs/flux-fill-pro").strip()
FLUX_STEPS = int(os.getenv("FLUX_STEPS", "50"))
FLUX_GUIDANCE = float(os.getenv("FLUX_GUIDANCE", "25"))
FLUX_OUTPUT_FORMAT = os.getenv("FLUX_OUTPUT_FORMAT", "png")
FLUX_PROMPT_UPSAMPLING = False
REPLICATE_HTTP_TIMEOUT = int(os.getenv("REPLICATE_HTTP_TIMEOUT", "120"))

FORCE_PRESERVE_OUTSIDE_MASK = True

# ============== ЦВЕТА ==============
COLOR_TURQUOISE = (0, 206, 209)
COLOR_WHITE = (255, 255, 255)
COLOR_OUTLINE = (60, 60, 60)

# ============== РАЗМЕРЫ ШРИФТОВ ==============
FONT_SIZE_MODE1 = 56
FONT_SIZE_MODE2 = 52
FONT_SIZE_MODE3_TITLE = 54
FONT_SIZE_MODE3_SUBTITLE = 52
FONT_SIZE_LOGO = 24
FONT_SIZE_MIN = 44

# ============== ОТСТУПЫ И РАССТОЯНИЯ ==============
SPACING_BOTTOM = 40
SPACING_LOGO_TO_TITLE = 6
SPACING_TITLE_TO_SUBTITLE = 10
LINE_SPACING = -16
LOGO_LINE_LENGTH = 310
LOGO_LINE_THICKNESS_PX = 3

# ============== МАСКА / OCR ==============
MASK_BOTTOM_PERCENT = 32
OCR_BOTTOM_PERCENT = 32

# ============== ГРАДИЕНТ ==============
GRADIENT_COVER_PERCENT = 50
GRADIENT_SOLID_FRACTION = 0.35
GRADIENT_SOLID_RAISE_PX = int(os.getenv("GRADIENT_SOLID_RAISE_PX", "125"))
GRADIENT_INTENSITY_CURVE = 2.6

# ============== РАСТЯЖЕНИЕ ТЕКСТА ==============
TEXT_STRETCH_HEIGHT = 1.5
TEXT_STRETCH_WIDTH = 1.15

# ============== ТЕНИ / ОБВОДКИ ==============
TEXT_SHADOW_OFFSET = 1
TEXT_OUTLINE_THICKNESS = 1

# ============== БЛОК ТЕКСТА ==============
TEXT_WIDTH_PERCENT = 0.50

# ============== OPENCV FALLBACK ==============
OPENCV_BLUR_SIGMA = 5
OPENCV_INPAINT_RADIUS = 3

# ============== ПУТЬ К ШРИФТУ ==============
FONT_PATH = os.getenv("FONT_PATH", "/app/fonts/WaffleSoft.otf").strip()

"""
==============================================
КОНЕЦ НАСТРОЕК
==============================================
"""

openai.api_key = OPENAI_API_KEY


# ---------------------------------------------------------------------
# OCR (Google Vision)
# ---------------------------------------------------------------------
def google_vision_ocr(image_bgr: np.ndarray, crop_bottom_percent: int = OCR_BOTTOM_PERCENT) -> dict:
    """OCR через Google Vision API по нижней части изображения."""
    if not GOOGLE_VISION_API_KEY:
        logger.warning("⚠️ GOOGLE_VISION_API_KEY не установлен")
        return {"text": "", "lines": []}

    try:
        h, w = image_bgr.shape[:2]
        crop_start = int(h * (1 - crop_bottom_percent / 100))
        cropped = image_bgr[crop_start:, :]

        logger.info(f"🔍 OCR на {crop_bottom_percent}% снизу (строки {crop_start}-{h})")

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        payload = {
            "requests": [{
                "image": {"content": image_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }

        resp = requests.post(url, json=payload, timeout=30)
        data = resp.json()

        if not data.get("responses"):
            logger.warning("⚠️ Нет результатов OCR")
            return {"text": "", "lines": []}

        r0 = data["responses"][0]
        ann = r0.get("textAnnotations")
        if not ann:
            logger.warning("⚠️ Текст не обнаружен")
            return {"text": "", "lines": []}

        full_text = ann[0].get("description", "")
        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]

        if lines and lines[0].strip().lower() in {"wealth", "@neurostep.media"}:
            lines = lines[1:]
            full_text = "\n".join(lines)

        logger.info(f"📝 OCR строки: {len(lines)}")
        return {"text": full_text.strip(), "lines": lines}

    except Exception as e:
        logger.error(f"❌ Ошибка Google Vision OCR: {e}")
        return {"text": "", "lines": []}


# ---------------------------------------------------------------------
# Чистка перевода (перед OpenAI)
# ---------------------------------------------------------------------
def _preclean_ocr_for_cover(text: str) -> str:
    """Лёгкая нормализация OCR-текста под обложку."""
    if not text:
        return text

    t = str(text)

    t = re.sub(r"@\S+", "", t)
    t = re.sub(r"(https?://\S+|www\.\S+)", "", t)
    t = re.sub(r"\b\d{1,2}:\d{2}\b", "", t)
    t = re.sub(r"[""«»\"']", "", t)
    t = re.sub(r"[|•·]+", " ", t)
    t = re.sub(r"\s*[-–—]{2,}\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    t = re.sub(r"(?i)\bwill\s+leave\s+you\s+speechless\b", "БЕЗ СЛОВ", t)
    t = re.sub(r"(?i)\bspeechless\b", "БЕЗ СЛОВ", t)

    t = re.sub(r"(?i)\bmulti[-\s]?billion\b", "МУЛЬТИ-МЛРД.", t)
    t = re.sub(r"(?i)\bmulti[-\s]?million\b", "МУЛЬТИ-МЛН.", t)

    t = re.sub(r"(?i)\bbillion\b", "млрд.", t)
    t = re.sub(r"(?i)\bmillion\b", "млн.", t)

    t = re.sub(r"(?i)\$\s*(\d+(?:\.\d+)?)\s*billion\b", r"$\1 млрд.", t)
    t = re.sub(r"(?i)\$\s*(\d+(?:\.\d+)?)\s*million\b", r"$\1 млн.", t)

    t = re.sub(r"(?i)\b([A-Z]{2,})S\b", r"\1", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------
# Перевод (OpenAI)
# ---------------------------------------------------------------------
def openai_translate(text: str) -> str:
    """Перевод и адаптация под СНГ."""
    if not OPENAI_API_KEY or not text:
        logger.warning("⚠️ OPENAI_API_KEY не установлен или нет текста")
        return text

    try:
        logger.info(f"🌐 Перевод: {text}")
        clean_text = _preclean_ocr_for_cover(text)
        logger.info(f"🧹 После чистки: {clean_text}")

        system_prompt = """Ты профессиональный редактор заголовков для русскоязычной (СНГ) аудитории (обложки/превью) в стиле Wealth: спокойно, уверенно, без кликбейта.
Задача: НЕ дословный перевод, а краткая адаптация смысла под обложку. Не выдумывай фактов.

Правила:
1) Бренды и имена оставляй на английском (SpaceX, Tesla, Apple, Ambani).
2) Адаптируй под естественный русский, без дословщины.
3) Используй короткие синонимы вместо длинных слов.
4) Сокращай валюту/числа: "billion" → "млрд.", "million" → "млн.", "$10 billion" → "$10 млрд."
5) Формат вывода: 1–3 строки. Разделитель строк — символ "\\n".
   - Если фраза короткая — 1 строка.
   - Если длинная — 2–3 строки (предпочтительно 2).
6) Никаких обращений к зрителю ("ВАС/ТЕБЯ") и никаких обещаний/кликбейта.
7) Верни ТОЛЬКО итоговый текст, без кавычек и пояснений, без точки в конце.

Плохие примеры (так нельзя):
- "АМБАНИ ВЛАДЕЕТ ВСЕМ"
- "ИМПЕРИЯ БЕЗ ГРАНИЦ"
- "ЭТО ТОЛЬКО НАЧАЛО"
- "СПОКОЙНАЯ СИЛА"
- "ЗАСТАВИТ ВАС ОТКРЫТЬ РОТ"
- "ВЫ НЕ ПОВЕРИТЕ", "ШОК", "СЕНСАЦИЯ"

Хорошие примеры:
"The Most Expensive Things Humans Have Ever Created" → "Самые дорогие творения человечества"
"SpaceX Starlink Satellite Constellation" → "Спутниковая сеть SpaceX Starlink"
"$10 billion" → "$10 млрд."
"We Share Insights That Expand Your View" → "Делимся знаниями, расширяющими кругозор"
"Aircraft" → "Истребитель"
"Northrop B-2 Spirit" → "Стелс-бомбардировщик B-2 Northrop Spirit"
"""

        resp = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Сделай заголовок для обложки: {clean_text}"},
            ],
            temperature=0.4,
            max_tokens=120,
        )

        translated = resp.choices[0].message.content.strip()

        translated = translated.strip().strip('"').strip("'").strip()
        translated = translated.rstrip(".")
        lines = [ln.strip() for ln in translated.splitlines() if ln.strip()]
        if len(lines) > 3:
            lines = lines[:3]
        translated = "\n".join(lines)

        logger.info(f"✅ Переведено: {translated}")
        return translated

    except Exception as e:
        logger.error(f"❌ Ошибка OpenAI перевода: {e}")
        return text


# ---------------------------------------------------------------------
# OpenCV fallback
# ---------------------------------------------------------------------
def opencv_fallback(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Запасной вариант без Replicate."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    result = image_bgr.copy()

    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=OPENCV_BLUR_SIGMA, sigmaY=OPENCV_BLUR_SIGMA)
    result[mask_u8 == 255] = blurred[mask_u8 == 255]

    try:
        result = cv2.inpaint(result, mask_u8, inpaintRadius=OPENCV_INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
    except Exception:
        pass

    logger.info("✅ OpenCV fallback (blur + light inpaint)")
    return result


# ---------------------------------------------------------------------
# Replicate FLUX Fill
# ---------------------------------------------------------------------
def flux_inpaint(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Inpaint через Replicate на модели FLUX Fill."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN не установлен → fallback OpenCV")
        return opencv_fallback(image_bgr, mask_u8)

    try:
        import replicate

        client = replicate.Client(api_token=REPLICATE_API_TOKEN)

        logger.info(f"🚀 Replicate inpaint: {REPLICATE_MODEL}")

        prompt = (
            "Remove all text, decorative lines and logos in the masked region. "
            "Reconstruct the original background naturally with clean, sharp detail. "
            "Match lighting, texture, and perspective. No blur, no smears, no artifacts, no repeating patterns. "
            "Do not change anything outside the mask. "
        )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_buf = BytesIO()
        pil_img.save(img_buf, format="PNG", compress_level=0)
        img_buf.seek(0)

        pil_mask = Image.fromarray(mask_u8, mode="L")
        mask_buf = BytesIO()
        pil_mask.save(mask_buf, format="PNG", compress_level=0)
        mask_buf.seek(0)

        output = client.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "image": img_buf,
                "mask": mask_buf,
                "steps": int(np.clip(FLUX_STEPS, 15, 50)),
                "guidance": float(np.clip(FLUX_GUIDANCE, 1.5, 100)),
                "prompt_upsampling": bool(FLUX_PROMPT_UPSAMPLING),
                "output_format": FLUX_OUTPUT_FORMAT,
            },
        )

        if isinstance(output, str):
            r = requests.get(output, timeout=REPLICATE_HTTP_TIMEOUT)
            r.raise_for_status()
            result_bytes = r.content
        elif isinstance(output, list) and output:
            r = requests.get(output[0], timeout=REPLICATE_HTTP_TIMEOUT)
            r.raise_for_status()
            result_bytes = r.content
        elif hasattr(output, "read"):
            result_bytes = output.read()
        else:
            logger.error(f"❌ Неизвестный формат output от Replicate: {type(output)}")
            return opencv_fallback(image_bgr, mask_u8)

        out_pil = Image.open(BytesIO(result_bytes)).convert("RGB")
        out_rgb = np.array(out_pil)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        if out_bgr.shape[:2] != image_bgr.shape[:2]:
            logger.warning("⚠️ Replicate изменил размер → ресайз обратно (LANCZOS)")
            out_bgr = cv2.resize(out_bgr, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        if FORCE_PRESERVE_OUTSIDE_MASK:
            out_bgr = _composite_by_mask(image_bgr, out_bgr, mask_u8)

        logger.info("✅ Replicate inpaint OK")
        return out_bgr

    except Exception as e:
        logger.error(f"❌ Ошибка Replicate inpaint: {e}")
        return opencv_fallback(image_bgr, mask_u8)


def _composite_by_mask(original_bgr: np.ndarray, edited_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Смешивание по маске."""
    m = (mask_u8.astype(np.float32) / 255.0)[:, :, None]
    out = (original_bgr.astype(np.float32) * (1.0 - m) + edited_bgr.astype(np.float32) * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def flux_kontext_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ALIAS для совместимости."""
    return flux_inpaint(image, mask)


# ---------------------------------------------------------------------
# Градиент
# ---------------------------------------------------------------------
def create_gradient_layer(width: int, height: int,
                          cover_percent: int = GRADIENT_COVER_PERCENT) -> Image.Image:
    """Создаёт RGBA-слой градиента для нижних cover_percent%."""
    cover_percent = int(np.clip(cover_percent, 1, 100))
    start_row = int(height * (1 - cover_percent / 100))
    grad_h = max(1, height - start_row)

    y = np.arange(height, dtype=np.float32)
    t = (y - start_row) / float(grad_h)
    t = np.clip(t, 0.0, 1.0)

    base_solid_from = 1.0 - float(np.clip(GRADIENT_SOLID_FRACTION, 0.0, 1.0))
    raise_t = float(np.clip(GRADIENT_SOLID_RAISE_PX, 0, height)) / float(grad_h)
    solid_from = float(np.clip(base_solid_from - raise_t, 0.0, 1.0))

    top_part = np.clip(t / max(solid_from, 1e-6), 0.0, 1.0)
    alpha = np.where(
        t >= solid_from,
        255.0,
        255.0 * (top_part ** float(GRADIENT_INTENSITY_CURVE)),
    ).astype(np.uint8)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha[:, None]

    logger.info(f"✨ Градиент: cover={cover_percent}%, start_row={start_row}, solid_from={solid_from:.3f}, raise_px={GRADIENT_SOLID_RAISE_PX}")
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------
# Текст: подбор размера и отрисовка со "stretch"
# ---------------------------------------------------------------------
def calculate_adaptive_font_size(text: str, font_path: str, max_width: int,
                                 initial_size: int, min_size: int = FONT_SIZE_MIN,
                                 stretch_width: float = TEXT_STRETCH_WIDTH) -> tuple:
    """Автоподбор размера шрифта. Greedy перенос."""
    text = (text or "").strip()
    if not text:
        font = ImageFont.truetype(font_path, int(min_size))
        return int(min_size), font, [""]

    words = text.split()
    if not words:
        font = ImageFont.truetype(font_path, int(min_size))
        return int(min_size), font, [text]

    size = int(initial_size)
    while size >= int(min_size):
        try:
            font = ImageFont.truetype(font_path, int(size))
            lines = _wrap_greedy(words, font, max_width, stretch_width)
            if lines:
                return int(size), font, lines
        except Exception as e:
            logger.error(f"Ошибка шрифта {size}: {e}")
        size -= 2

    font = ImageFont.truetype(font_path, int(min_size))
    return int(min_size), font, [text]


def _wrap_greedy(words: list, font: ImageFont.FreeTypeFont, max_width: int, stretch: float) -> list:
    """Greedy перенос: добавляем слова пока влезают."""
    if not words:
        return []
    
    space_w = max(1, _text_width_px(font, " "))
    lines = []
    current = []
    current_w = 0
    
    for w in words:
        w_width = _text_width_px(font, w)
        test_w = current_w + (space_w if current else 0) + w_width
        
        if current and int(test_w * stretch) > max_width:
            lines.append(" ".join(current))
            current = [w]
            current_w = w_width
        else:
            current.append(w)
            current_w = test_w
    
    if current:
        lines.append(" ".join(current))
    
    return lines if lines else []


def _text_width_px(font: ImageFont.FreeTypeFont, text: str) -> int:
    """Ширина текста в пикселях."""
    bb = font.getbbox(text)
    return int(bb[2] - bb[0])


def draw_text_with_stretch(base_image: Image.Image,
                           x: int, y: int,
                           text: str,
                           font: ImageFont.FreeTypeFont,
                           fill_color: tuple,
                           outline_color: tuple,
                           stretch_width: float = TEXT_STRETCH_WIDTH,
                           stretch_height: float = TEXT_STRETCH_HEIGHT,
                           shadow_offset: int = TEXT_SHADOW_OFFSET) -> int:
    """Рисует текст с тенью+обводкой, затем растягивает."""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = max(6, shadow_offset + TEXT_OUTLINE_THICKNESS * 2)
    temp_w = int(tw * (stretch_width + 1.0)) + pad * 2
    temp_h = int(th * (stretch_height + 1.0)) + pad * 2

    temp = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(temp)

    tx, ty = pad, pad

    # Тень
    d.text((tx + shadow_offset, ty + shadow_offset), text, font=font, fill=(0, 0, 0, 128))

    # Обводка
    for t in range(int(TEXT_OUTLINE_THICKNESS)):
        r = t + 1
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            d.text((tx + dx * r, ty + dy * r), text, font=font, fill=outline_color)

    # Основной
    d.text((tx, ty), text, font=font, fill=fill_color)

    bb = temp.getbbox()
    if not bb:
        return th

    crop = temp.crop(bb)
    sw = max(1, int(crop.width * stretch_width))
    sh = max(1, int(crop.height * stretch_height))
    crop = crop.resize((sw, sh), Image.Resampling.LANCZOS)

    base_image.paste(crop, (x, y), crop)
    return sh


def _estimate_fixed_line_height(font: ImageFont.FreeTypeFont) -> int:
    """Фиксированная высота строки."""
    try:
        ascent, descent = font.getmetrics()
        base = int((ascent + descent) * TEXT_STRETCH_HEIGHT)
    except Exception:
        base = int(font.size * TEXT_STRETCH_HEIGHT)
    pad = max(6, TEXT_SHADOW_OFFSET + int(TEXT_OUTLINE_THICKNESS) * 2)
    return base + pad


# ---------------------------------------------------------------------
# РЕНДЕРЫ РЕЖИМОВ
# ---------------------------------------------------------------------
def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    """Режим 1: Лого + линии + заголовок (UPPERCASE)."""
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()
    _, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE1, stretch_width=TEXT_STRETCH_WIDTH
    )

    line_h = _estimate_fixed_line_height(title_font)
    total_title_h = line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)

    # Лого
    logo_font = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOGO)
    logo_text = "@neurostep.media"
    bb = logo_font.getbbox(logo_text)
    logo_w = bb[2] - bb[0]
    logo_h = bb[3] - bb[1]

    total_h = logo_h + SPACING_LOGO_TO_TITLE + total_title_h
    start_y = height - SPACING_BOTTOM - total_h

    # Лого позиция
    logo_x = (width - logo_w) // 2
    logo_y = start_y

    # Линии
    line_y = logo_y + logo_h // 2
    line_left_start = logo_x - LOGO_LINE_LENGTH - 10
    line_right_start = logo_x + logo_w + 10

    draw.line([(line_left_start, line_y), (line_left_start + LOGO_LINE_LENGTH, line_y)], fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX)
    draw.line([(line_right_start, line_y), (line_right_start + LOGO_LINE_LENGTH, line_y)], fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX)
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)

    # Заголовок
    cur_y = start_y + logo_h + SPACING_LOGO_TO_TITLE
    block_left = (width - max_text_width) // 2
    for i, ln in enumerate(title_lines):
        draw_text_with_stretch(image, block_left, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h
        if i < len(title_lines) - 1:
            cur_y += LINE_SPACING

    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """Режим 2: только заголовок (UPPERCASE)."""
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()
    _, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE2, stretch_width=TEXT_STRETCH_WIDTH
    )

    line_h = _estimate_fixed_line_height(title_font)
    total_h = line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)

    start_y = height - SPACING_BOTTOM - total_h
    cur_y = start_y
    block_left = (width - max_text_width) // 2

    for i, ln in enumerate(title_lines):
        draw_text_with_stretch(image, block_left, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h
        if i < len(title_lines) - 1:
            cur_y += LINE_SPACING

    return image


def render_mode3_content(image: Image.Image, title_translated: str, subtitle_translated: str) -> Image.Image:
    """Режим 3: заголовок + подзаголовок (оба UPPERCASE)."""
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()
    subtitle = (subtitle_translated or "").upper()

    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE3_TITLE, stretch_width=TEXT_STRETCH_WIDTH
    )

    subtitle_initial = int(title_size * 0.80)
    _, subtitle_font, subtitle_lines = calculate_adaptive_font_size(
        subtitle, FONT_PATH, max_text_width, subtitle_initial, stretch_width=TEXT_STRETCH_WIDTH
    )

    title_line_h = _estimate_fixed_line_height(title_font)
    sub_line_h = _estimate_fixed_line_height(subtitle_font)

    total_title_h = title_line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)
    total_sub_h = sub_line_h * len(subtitle_lines) + max(0, (len(subtitle_lines) - 1) * LINE_SPACING)

    total_h = total_title_h + SPACING_TITLE_TO_SUBTITLE + total_sub_h
    start_y = height - SPACING_BOTTOM - total_h

    cur_y = start_y
    block_left = (width - max_text_width) // 2

    for i, ln in enumerate(title_lines):
        draw_text_with_stretch(image, block_left, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += title_line_h
        if i < len(title_lines) - 1:
            cur_y += LINE_SPACING

    cur_y += SPACING_TITLE_TO_SUBTITLE

    for i, ln in enumerate(subtitle_lines):
        draw_text_with_stretch(image, block_left, cur_y, ln, subtitle_font, COLOR_WHITE, COLOR_OUTLINE)
        cur_y += sub_line_h
        if i < len(subtitle_lines) - 1:
            cur_y += LINE_SPACING

    return image


# ---------------------------------------------------------------------
# ОСНОВНОЙ WORKFLOW
# ---------------------------------------------------------------------
def process_full_workflow(image_bgr: np.ndarray, mode: int) -> tuple:
    """Полный workflow для режимов 1,2,3."""
    logger.info("=" * 60)
    logger.info(f"🚀 ПОЛНЫЙ WORKFLOW - РЕЖИМ {mode}")
    logger.info("=" * 60)

    h, w = image_bgr.shape[:2]

    # ШАГ 1: OCR
    logger.info("📋 ШАГ 1: OCR (Google Vision)")
    ocr = google_vision_ocr(image_bgr, crop_bottom_percent=OCR_BOTTOM_PERCENT)
    if not ocr["text"]:
        logger.warning("⚠️ Текст не обнаружен")
        return image_bgr, ocr

    # ШАГ 2: Маска нижних N%
    logger.info("📋 ШАГ 2: Маска (нижние %)")
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_start = int(h * (1 - MASK_BOTTOM_PERCENT / 100))
    mask[mask_start:, :] = 255
    logger.info(f"📐 Маска: строки {mask_start}-{h} (нижние {MASK_BOTTOM_PERCENT}%)")

    # ШАГ 3: Inpaint (Replicate → FLUX Fill)
    logger.info("📋 ШАГ 3: Inpaint (Replicate FLUX Fill)")
    clean_bgr = flux_inpaint(image_bgr, mask)

    # ШАГ 4: Перевод
    logger.info("📋 ШАГ 4: Перевод (OpenAI)")
    title_translated, subtitle_translated = "", ""

    if mode == 3:
        lines = ocr["lines"]
        if len(lines) >= 2:
            title = " ".join(lines[:-1])
            subtitle = lines[-1]
        else:
            title, subtitle = ocr["text"], ""

        title_translated = openai_translate(title)
        subtitle_translated = openai_translate(subtitle) if subtitle else ""
    else:
        title_translated = openai_translate(ocr["text"])

    # ШАГ 5: Градиент (точно на нижние N%)
    logger.info("📋 ШАГ 5: Градиент")
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(clean_rgb).convert("RGBA")

    grad = create_gradient_layer(pil.size[0], pil.size[1], cover_percent=GRADIENT_COVER_PERCENT)
    pil = Image.alpha_composite(pil, grad)
    logger.info("✅ Градиент наложен")

    # ШАГ 6: Текст/лого по режимам
    logger.info("📋 ШАГ 6: Рендер текста")
    if mode == 1:
        pil = render_mode1_logo(pil, title_translated)
    elif mode == 2:
        pil = render_mode2_text(pil, title_translated)
    elif mode == 3:
        pil = render_mode3_content(pil, title_translated, subtitle_translated)
    else:
        logger.warning(f"⚠️ Неизвестный режим {mode} → пропускаю рендер")

    out_rgb = np.array(pil.convert("RGB"))
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    logger.info("=" * 60)
    logger.info("✅ WORKFLOW ЗАВЕРШЁН!")
    logger.info("=" * 60)
    return out_bgr, ocr


# Совместимость
def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Алиас для inpaint."""
    return flux_inpaint(image, mask)
