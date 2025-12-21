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
FLUX_GUIDANCE = float(os.getenv("FLUX_GUIDANCE", "3.5"))
FLUX_OUTPUT_FORMAT = os.getenv("FLUX_OUTPUT_FORMAT", "png")
FLUX_PROMPT_UPSAMPLING = False
REPLICATE_HTTP_TIMEOUT = int(os.getenv("REPLICATE_HTTP_TIMEOUT", "120"))

FORCE_PRESERVE_OUTSIDE_MASK = True

# ============== ЦВЕТА ==============
COLOR_TURQUOISE = (0, 206, 209)
COLOR_WHITE = (255, 255, 255)
COLOR_OUTLINE = (60, 60, 60)

# ============== РАЗМЕРЫ ШРИФТОВ ==============
FONT_SIZE_MODE1 = 48
FONT_SIZE_MODE2 = 48
FONT_SIZE_MODE3_TITLE = 48
FONT_SIZE_MODE3_SUBTITLE = 46
FONT_SIZE_LOGO = 24
FONT_SIZE_MIN = 44

# ============== ОТСТУПЫ И РАССТОЯНИЯ ==============
SPACING_BOTTOM = -80
SPACING_BOTTOM_MODE3 = 40
SPACING_LOGO_TO_TITLE = 8
SPACING_TITLE_TO_SUBTITLE = -30
LINE_SPACING = -45
LOGO_LINE_LENGTH = 310
LOGO_LINE_THICKNESS_PX = 3

# ============== МАСКА / OCR ==============
MASK_BOTTOM_PERCENT = 32
OCR_BOTTOM_PERCENT = 32

# ============== ГРАДИЕНТ (Instagram-стиль) ==============
GRADIENT_HEIGHT_MODE12 = 55
GRADIENT_HEIGHT_MODE3 = 45
GRADIENT_SOLID_FRACTION = 0.5
GRADIENT_TRANSITION_CURVE = 2.2
GRADIENT_BLUR_SIGMA = 120

# ============== УЛУЧШЕНИЯ ГРАДИЕНТА ==============
GRADIENT_NOISE_INTENSITY = 8

# ============== ПОСТОБРАБОТКА ИЗОБРАЖЕНИЯ ==============
ENHANCE_BRIGHTNESS = 1.05
ENHANCE_CONTRAST = 1.0
ENHANCE_SATURATION = 1.25
ENHANCE_SHARPNESS = 1.3

# ============== КЕРНИНГ ТЕКСТА ==============
LETTER_SPACING_PX = 3

# ============== УЛУЧШЕНИЯ ТЕКСТА ==============
TEXT_GRAIN_INTENSITY = 0.15
TEXT_INNER_SHADOW_SIZE = 1
TEXT_SHARPEN_AMOUNT = 0.3

# ============== РАСТЯЖЕНИЕ ТЕКСТА ==============
TEXT_STRETCH_HEIGHT = 2.1
TEXT_STRETCH_WIDTH = 1.05

# ============== ТЕНИ / ОБВОДКИ ==============
TEXT_SHADOW_OFFSET = 2
TEXT_OUTLINE_THICKNESS = 1

# ============== БЛОК ТЕКСТА ==============
TEXT_WIDTH_PERCENT = 0.90

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


def _preclean_ocr_for_cover(text: str) -> str:
    if not text:
        return text
    t = str(text)
    
    t = re.sub(r"@\S+", "", t)
    t = re.sub(r"(https?://\S+|www\.\S+)", "", t)
    t = re.sub(r"\b\d{1,2}:\d{2}\b", "", t)
    t = re.sub(r"[""«»\"']", "", t)
    t = re.sub(r"[|•·]+", " ", t)
    t = re.sub(r"\s*[-–—]{2,}\s*", " ", t)
    
    t = re.sub(r"(?i)\$\s*(\d+(?:\.\d+)?)\s*billion", r"$\1 млрд.", t)
    t = re.sub(r"(?i)\$\s*(\d+(?:\.\d+)?)\s*million", r"$\1 млн.", t)
    t = re.sub(r"(?i)\bmulti[-\s]?billion", "мульти-млрд.", t)
    t = re.sub(r"(?i)\bmulti[-\s]?million", "мульти-млн.", t)
    t = re.sub(r"(?i)\bbillion", "млрд.", t)
    t = re.sub(r"(?i)\bmillion", "млн.", t)
    
    t = re.sub(r"\b([A-Z]{2,})S\b", r"\1", t)
    
    t = re.sub(r"\s+", " ", t).strip()
    return t


def openai_translate(text: str) -> str:
    if not OPENAI_API_KEY or not text:
        logger.warning("⚠️ OPENAI_API_KEY не установлен или нет текста")
        return text

    try:
        logger.info(f"🌐 Перевод: {text}")
        clean_text = _preclean_ocr_for_cover(text)
        logger.info(f"🧹 После чистки: {clean_text}")

        system_prompt = """Ты — переводчик и редактор коротких титров для видео/карусели. 
Задача: перевести с английского на русский и адаптировать для экрана. 

Правила:
1) Сохраняй названия объектов/брендов/мест (Antilia, Sea Wind, Mandarin Oriental, Tribeca и т.п.) в оригинале латиницей, но города/регионы переводи на русский: Mumbai → Мумбаи, New York → Нью-Йорк, Buckinghamshire → Бакингемшир.
2) Формат вывода: каждая строка отдельно, без списков и без комментариев. 
3) Денежные суммы конвертируй в русское оформление: 
   - million → «млн $», billion → «млрд $»
   - десятичная запятая: 4.6 → 4,6
   - знак валюты ставь после числа: «79 млн $», «4,6 млрд $».
4) Если строка — слоган/фраза, переведи естественно, без кальки, в нейтрально-деловом тоне.
5) Никаких добавлений фактов. Не выдумывай детали. 
6) Если в тексте есть многоточие/обрыв — сохрани его.

Вход: набор строк на английском (каждая с новой строки).
Выход: тот же набор строк на русском, в том же порядке.

✅ ХОРОШО:

"Aircraft" → "Истребитель"
"Northrop B-2 Spirit" → "Стелс-бомбардировщик B-2 Northrop Spirit"
"""

        resp = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Сделай заголовок для обложки: {clean_text}"},
            ],
            temperature=0.2,
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


def opencv_fallback(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
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


def flux_inpaint(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN не установлен → fallback OpenCV")
        return opencv_fallback(image_bgr, mask_u8)

    try:
        import replicate
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)

        logger.info(f"🚀 Replicate inpaint: LaMa")

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
            "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72",
            input={
                "image": img_buf,
                "mask": mask_buf
            }
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
            logger.error(f"❌ Неизвестный формат output")
            return opencv_fallback(image_bgr, mask_u8)

        out_pil = Image.open(BytesIO(result_bytes)).convert("RGB")
        out_rgb = np.array(out_pil)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        if out_bgr.shape[:2] != image_bgr.shape[:2]:
            logger.warning("⚠️ Replicate изменил размер → ресайз обратно")
            out_bgr = cv2.resize(out_bgr, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        if FORCE_PRESERVE_OUTSIDE_MASK:
            out_bgr = _composite_by_mask(image_bgr, out_bgr, mask_u8)

        logger.info("✅ LaMa inpaint OK")
        return out_bgr

    except Exception as e:
        logger.error(f"❌ Ошибка Replicate inpaint: {e}")
        return opencv_fallback(image_bgr, mask_u8)

def _composite_by_mask(original_bgr: np.ndarray, edited_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0)[:, :, None]
    out = (original_bgr.astype(np.float32) * (1.0 - m) + edited_bgr.astype(np.float32) * m)
    return np.clip(out, 0, 255).astype(np.uint8)

def flux_kontext_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return flux_inpaint(image, mask)


def create_gradient_layer(width: int, height: int,
                          gradient_height_percent: int) -> Image.Image:
    
    grad_h = int(height * gradient_height_percent / 100)
    start_row = height - grad_h
    
    alpha = np.zeros(height, dtype=np.float32)
    
    for i in range(height):
        if i < start_row:
            alpha[i] = 0.0
        else:
            t = (height - 1 - i) / float(grad_h)
            
            if t <= GRADIENT_SOLID_FRACTION:
                alpha[i] = 1.0
            else:
                t_norm = (t - GRADIENT_SOLID_FRACTION) / (1.0 - GRADIENT_SOLID_FRACTION)
                alpha[i] = 1.0 - (t_norm ** GRADIENT_TRANSITION_CURVE)
    
    alpha_u8 = (alpha * 255).astype(np.uint8)
    alpha_2d = np.tile(alpha_u8[:, None], (1, width))
    
    if GRADIENT_NOISE_INTENSITY > 0:
        noise = np.random.normal(0, GRADIENT_NOISE_INTENSITY, (height, width)).astype(np.float32)
        alpha_2d_float = alpha_2d.astype(np.float32) + noise
        alpha_2d = np.clip(alpha_2d_float, 0, 255).astype(np.uint8)
    
    ksize_y = int(GRADIENT_BLUR_SIGMA * 6) | 1
    alpha_blurred = cv2.GaussianBlur(alpha_2d, (1, ksize_y), sigmaX=0, sigmaY=GRADIENT_BLUR_SIGMA)
    
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha_blurred
    
    logger.info(f"✨ Градиент: {gradient_height_percent}%, solid={GRADIENT_SOLID_FRACTION*100}%, blur={GRADIENT_BLUR_SIGMA}, noise={GRADIENT_NOISE_INTENSITY}")
    return Image.fromarray(rgba, mode="RGBA")


def enhance_image(image_bgr: np.ndarray) -> np.ndarray:
    
    enhanced = cv2.convertScaleAbs(image_bgr, alpha=ENHANCE_CONTRAST, beta=(ENHANCE_BRIGHTNESS - 1.0) * 30)
    
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * ENHANCE_SATURATION, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    if ENHANCE_SHARPNESS > 1.0:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, ENHANCE_SHARPNESS, blurred, -(ENHANCE_SHARPNESS - 1.0), 0)
    
    logger.info(f"📸 Улучшение: яркость={ENHANCE_BRIGHTNESS:.2f}, контраст={ENHANCE_CONTRAST:.2f}, насыщенность={ENHANCE_SATURATION:.2f}, резкость={ENHANCE_SHARPNESS:.2f}")
    return enhanced


def calculate_adaptive_font_size(text: str, font_path: str, max_width: int,
                                 initial_size: int, min_size: int = FONT_SIZE_MIN,
                                 stretch_width: float = TEXT_STRETCH_WIDTH) -> tuple:
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
    if not words:
        return []
    
    space_w = max(1, _text_width_px(font, " ", spacing=LETTER_SPACING_PX))
    lines = []
    current = []
    current_w = 0
    
    for w in words:
        w_width = _text_width_px(font, w, spacing=LETTER_SPACING_PX)
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


def _text_width_px(font: ImageFont.FreeTypeFont, text: str, spacing: int = 0) -> int:
    bb = font.getbbox(text)
    base_width = int(bb[2] - bb[0])
    
    if spacing > 0 and len(text) > 1:
        return base_width + (len(text) - 1) * spacing
    
    return base_width


def _draw_text_with_letter_spacing(draw: ImageDraw.ImageDraw, pos: tuple, text: str, 
                                   font: ImageFont.FreeTypeFont, fill: tuple, spacing: int = 0) -> int:
    if spacing <= 0:
        draw.text(pos, text, font=font, fill=fill)
        bb = font.getbbox(text)
        return int(bb[2] - bb[0])
    
    x, y = pos
    total_width = 0
    
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        bb = font.getbbox(char)
        char_width = int(bb[2] - bb[0])
        x += char_width + spacing
        total_width += char_width + spacing
    
    return total_width - spacing if total_width > 0 else 0


def draw_text_with_stretch(base_image: Image.Image,
                           x: int, y: int,
                           text: str,
                           font: ImageFont.FreeTypeFont,
                           fill_color: tuple,
                           outline_color: tuple,
                           stretch_width: float = TEXT_STRETCH_WIDTH,
                           stretch_height: float = TEXT_STRETCH_HEIGHT,
                           shadow_offset: int = TEXT_SHADOW_OFFSET,
                           apply_enhancements: bool = True) -> int:
    bbox = font.getbbox(text)
    tw = _text_width_px(font, text, spacing=LETTER_SPACING_PX)
    th = bbox[3] - bbox[1]

    pad = max(6, shadow_offset + TEXT_OUTLINE_THICKNESS * 2)
    temp_w = int(tw * (stretch_width + 1.0)) + pad * 2
    temp_h = int(th * (stretch_height + 1.0)) + pad * 2

    temp = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(temp)

    tx, ty = pad, pad

    _draw_text_with_letter_spacing(d, (tx + shadow_offset, ty + shadow_offset), text, font, (0, 0, 0, 128), spacing=LETTER_SPACING_PX)

    for t in range(int(TEXT_OUTLINE_THICKNESS)):
        r = t + 1
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            _draw_text_with_letter_spacing(d, (tx + dx * r, ty + dy * r), text, font, outline_color, spacing=LETTER_SPACING_PX)

    _draw_text_with_letter_spacing(d, (tx, ty), text, font, fill_color, spacing=LETTER_SPACING_PX)
    
    if apply_enhancements:
        if TEXT_INNER_SHADOW_SIZE > 0:
            temp_arr = np.array(temp)
            alpha = temp_arr[:, :, 3]
            
            kernel = np.ones((TEXT_INNER_SHADOW_SIZE * 2 + 1, TEXT_INNER_SHADOW_SIZE * 2 + 1), np.uint8)
            eroded = cv2.erode(alpha, kernel, iterations=1)
            inner_shadow_mask = (alpha > 0) & (eroded == 0)
            
            temp_arr[inner_shadow_mask, :3] = temp_arr[inner_shadow_mask, :3] * 0.7
            temp = Image.fromarray(temp_arr)
        
        if TEXT_GRAIN_INTENSITY > 0:
            temp_arr = np.array(temp).astype(np.float32)
            alpha = temp_arr[:, :, 3]
            
            noise = np.random.normal(0, TEXT_GRAIN_INTENSITY * 25, (temp_h, temp_w, 3))
            text_mask = alpha > 0
            
            temp_arr[:, :, :3][text_mask] += noise[text_mask]
            temp_arr = np.clip(temp_arr, 0, 255).astype(np.uint8)
            temp = Image.fromarray(temp_arr)

    bb = temp.getbbox()
    if not bb:
        return th

    crop = temp.crop(bb)
    sw = max(1, int(crop.width * stretch_width))
    sh = max(1, int(crop.height * stretch_height))
    crop = crop.resize((sw, sh), Image.Resampling.LANCZOS)
    
    if apply_enhancements and TEXT_SHARPEN_AMOUNT > 0:
        crop_arr = np.array(crop).astype(np.float32)
        rgb = crop_arr[:, :, :3]
        alpha = crop_arr[:, :, 3:4]
        
        blurred = cv2.GaussianBlur(rgb, (0, 0), 1.0)
        sharpened = rgb + TEXT_SHARPEN_AMOUNT * (rgb - blurred)
        sharpened = np.clip(sharpened, 0, 255)
        
        crop_arr[:, :, :3] = sharpened
        crop = Image.fromarray(crop_arr.astype(np.uint8))

    base_image.paste(crop, (x, y), crop)
    return sh


def _estimate_fixed_line_height(font: ImageFont.FreeTypeFont) -> int:
    try:
        ascent, descent = font.getmetrics()
        base = int((ascent + descent) * TEXT_STRETCH_HEIGHT)
    except Exception:
        base = int(font.size * TEXT_STRETCH_HEIGHT)
    pad = max(6, TEXT_SHADOW_OFFSET + int(TEXT_OUTLINE_THICKNESS) * 2)
    return base + pad


def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
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

    logo_font = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOGO)
    logo_text = "@neurostep.media"
    bb = logo_font.getbbox(logo_text)
    logo_w = bb[2] - bb[0]
    logo_h = bb[3] - bb[1]

    total_h = logo_h + SPACING_LOGO_TO_TITLE + total_title_h
    start_y = height - SPACING_BOTTOM - total_h

    logo_x = (width - logo_w) // 2
    logo_y = start_y

    line_y = logo_y + logo_h // 2
    line_left_start = logo_x - LOGO_LINE_LENGTH - 10
    line_right_start = logo_x + logo_w + 10

    draw.line([(line_left_start, line_y), (line_left_start + LOGO_LINE_LENGTH, line_y)], fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX)
    draw.line([(line_right_start, line_y), (line_right_start + LOGO_LINE_LENGTH, line_y)], fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX)
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)

    cur_y = start_y + logo_h + SPACING_LOGO_TO_TITLE
    block_left = (width - max_text_width) // 2
    
    for i, ln in enumerate(title_lines):
        line_w = int(_text_width_px(title_font, ln, spacing=LETTER_SPACING_PX) * TEXT_STRETCH_WIDTH)
        line_x = block_left + (max_text_width - line_w) // 2
        draw_text_with_stretch(image, line_x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h
        if i < len(title_lines) - 1:
            cur_y += LINE_SPACING

    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
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
        line_w = int(_text_width_px(title_font, ln, spacing=LETTER_SPACING_PX) * TEXT_STRETCH_WIDTH)
        line_x = block_left + (max_text_width - line_w) // 2
        draw_text_with_stretch(image, line_x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h
        if i < len(title_lines) - 1:
            cur_y += LINE_SPACING

    return image


def render_mode3_content(image: Image.Image, title_translated: str, subtitle_translated: str) -> Image.Image:
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
    start_y = height - SPACING_BOTTOM_MODE3 - total_h

    cur_y = start_y
    block_left = (width - max_text_width) // 2

    for i, ln in enumerate(title_lines):
        line_w = int(_text_width_px(title_font, ln, spacing=LETTER_SPACING_PX) * TEXT_STRETCH_WIDTH)
        line_x = block_left + (max_text_width - line_w) // 2
        draw_text_with_stretch(image, line_x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += title_line_h
        if i < len(title_lines) - 1:
            cur_y += LINE_SPACING

    cur_y += SPACING_TITLE_TO_SUBTITLE

    for i, ln in enumerate(subtitle_lines):
        line_w = int(_text_width_px(subtitle_font, ln, spacing=LETTER_SPACING_PX) * TEXT_STRETCH_WIDTH)
        line_x = block_left + (max_text_width - line_w) // 2
        draw_text_with_stretch(image, line_x, cur_y, ln, subtitle_font, COLOR_WHITE, COLOR_OUTLINE)
        cur_y += sub_line_h
        if i < len(subtitle_lines) - 1:
            cur_y += LINE_SPACING

    return image


def process_full_workflow(image_bgr: np.ndarray, mode: int) -> tuple:
    logger.info("=" * 60)
    logger.info(f"🚀 ПОЛНЫЙ WORKFLOW - РЕЖИМ {mode}")
    logger.info("=" * 60)

    h, w = image_bgr.shape[:2]

    logger.info("📋 ШАГ 1: OCR (Google Vision)")
    ocr = google_vision_ocr(image_bgr, crop_bottom_percent=OCR_BOTTOM_PERCENT)
    if not ocr["text"]:
        logger.warning("⚠️ Текст не обнаружен")
        return image_bgr, ocr

    logger.info("📋 ШАГ 2: Маска (нижние %)")
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_start = int(h * (1 - MASK_BOTTOM_PERCENT / 100))
    mask[mask_start:, :] = 255
    logger.info(f"📐 Маска: строки {mask_start}-{h} (нижние {MASK_BOTTOM_PERCENT}%)")

    logger.info("📋 ШАГ 3: Inpaint (Replicate FLUX Fill)")
    clean_bgr = flux_inpaint(image_bgr, mask)

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

    logger.info("📋 ШАГ 5: Градиент")
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(clean_rgb).convert("RGBA")

    if mode == 3:
        grad = create_gradient_layer(pil.size[0], pil.size[1], gradient_height_percent=GRADIENT_HEIGHT_MODE3)
    else:
        grad = create_gradient_layer(pil.size[0], pil.size[1], gradient_height_percent=GRADIENT_HEIGHT_MODE12)
    pil = Image.alpha_composite(pil, grad)
    logger.info("✅ Градиент наложен")

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


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return flux_inpaint(image, mask)
