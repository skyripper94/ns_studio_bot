# lama_integration.py - FIXED VERSION

import os
import logging
import base64
from io import BytesIO

import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import openai
import re

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "black-forest-labs/flux-fill-pro").strip()
FLUX_STEPS = int(os.getenv("FLUX_STEPS", "50"))
FLUX_GUIDANCE = float(os.getenv("FLUX_GUIDANCE", "3.5"))
FLUX_OUTPUT_FORMAT = os.getenv("FLUX_OUTPUT_FORMAT", "png")
FLUX_PROMPT_UPSAMPLING = False
REPLICATE_HTTP_TIMEOUT = int(os.getenv("REPLICATE_HTTP_TIMEOUT", "120"))

FORCE_PRESERVE_OUTSIDE_MASK = True

COLOR_TURQUOISE = (0, 206, 209)
COLOR_WHITE = (255, 255, 255)
COLOR_OUTLINE = (60, 60, 60)

FONT_SIZE_MODE1 = 54
FONT_SIZE_MODE2 = 54
FONT_SIZE_MODE3_TITLE = 56
FONT_SIZE_MODE3_SUBTITLE = 50
FONT_SIZE_LOGO = 24
FONT_SIZE_MIN = 44

SPACING_BOTTOM_MODE1 = 20
SPACING_BOTTOM_MODE2 = 45
SPACING_BOTTOM_MODE3 = 75
SPACING_LOGO_TO_TITLE = -15
SPACING_TITLE_TO_SUBTITLE = -50
LINE_SPACING = -60
LINE_SPACING_SUBTITLE = -50
LOGO_LINE_LENGTH = 310
LOGO_LINE_THICKNESS_PX = 3

MASK_BOTTOM_MODE1 = 38
MASK_BOTTOM_MODE2 = 33
MASK_BOTTOM_MODE3 = 33
OCR_BOTTOM_PERCENT = 32

GRADIENT_HEIGHT_MODE12 = 42
GRADIENT_HEIGHT_MODE3 = 38
GRADIENT_SOLID_FRACTION = 0.5
GRADIENT_TRANSITION_CURVE = 2.2
GRADIENT_BLUR_SIGMA = 120
GRADIENT_NOISE_INTENSITY = 10

ENHANCE_BRIGHTNESS = 1.05
ENHANCE_CONTRAST = 1.0
ENHANCE_SATURATION = 1.25
ENHANCE_SHARPNESS = 1.3

LETTER_SPACING_PX = 4
TEXT_GRAIN_INTENSITY = 0.25
TEXT_INNER_SHADOW_SIZE = 1
TEXT_SHARPEN_AMOUNT = 0.3

TEXT_STRETCH_HEIGHT = 1.8
TEXT_STRETCH_WIDTH = 1.05

TEXT_SHADOW_OFFSET = 2
TEXT_OUTLINE_THICKNESS = 2

TEXT_WIDTH_PERCENT = 0.90

OPENCV_BLUR_SIGMA = 5
OPENCV_INPAINT_RADIUS = 3

FONT_PATH = os.getenv("FONT_PATH", "/app/fonts/WaffleSoft.otf").strip()

openai.api_key = OPENAI_API_KEY


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

        system_prompt = """Ты редактор коротких титров и описаний для Instagram/TikTok. 
Задача: перевести и адаптировать текст с английского на русский под визуальный формат.

Правила:
1) Сохраняй названия объектов/брендов/мест (Antilia, Sea Wind, Mandarin Oriental, Tribeca и т.п.) в оригинале латиницей, но города/регионы переводи на русский: Mumbai → Мумбаи, New York → Нью-Йорк, Buckinghamshire → Бакингемшир.
2) Формат вывода: каждая строка отдельно, без списков и без комментариев. 
3) Денежные суммы конвертируй в русское оформление: 
   - million → «миллион $», billion → «миллиард $»
   - десятичная запятая: 4.6 → 4,6
   - знак валюты ставь после числа: «79 миллион $», «4,6 миллиард $».
4) Если строка — слоган/фраза, переведи естественно, без кальки, в нейтрально-деловом тоне.
5) Никаких добавлений фактов. Не выдумывай детали. 
6) Если в тексте есть многоточие/обрыв — сохрани его.
7) Букву Ё заменяй на Е

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


def create_gradient_layer(width: int, height: int, gradient_height_percent: int) -> Image.Image:
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

# --- PATCH: auto "no orphan prepositions" + manual hard breaks ---

ORPHANS_RU = {
    # 1-буквенные (главные)
    "В", "К", "С", "У", "О", "А", "И", "Я", "ВО", "НА", "НО", "НЕ", "ПО", "ЗА", "ОТ", "ДО"

    # если хочешь — включи 2-буквенные (иногда тоже полезно)
    # "ВО", "НА", "НО", "НЕ", "ПО", "ЗА", "ОТ", "ДО"
}

_ORPHAN_STRIP = ".,:;!?…—-()[]{}\"'«»"

def _norm_orphan(word: str) -> str:
    # "В," -> "В"
    return (word or "").strip().strip(_ORPHAN_STRIP).upper()

def _wrap_text_preserve_breaks(text: str, font, max_width: int, stretch_width: float) -> list:
    """
    Вариант 2 (ручной):
    - '|' = жесткий перенос строки
    - '\n' = жесткий перенос строки
    Каждый кусок переносится отдельно и не "перетекает" через разрыв.
    """
    text = (text or "").replace("|", "\n").strip()
    if not text:
        return [""]

    out = []
    for part in text.splitlines():
        part = part.strip()
        if not part:
            continue
        words = part.split()
        out.extend(_wrap_greedy(words, font, max_width, stretch_width))
    return out if out else [""]


def calculate_adaptive_font_size(text: str, font_path: str, max_width: int,
                                 initial_size: int, min_size: int = FONT_SIZE_MIN,
                                 stretch_width: float = TEXT_STRETCH_WIDTH) -> tuple:
    text = (text or "").strip()
    if not text:
        font = ImageFont.truetype(font_path, int(min_size))
        return int(min_size), font, [""]

    size = int(initial_size)
    while size >= int(min_size):
        try:
            font = ImageFont.truetype(font_path, int(size))
            # ВАЖНО: теперь учитываем ручные переносы ('|' и '\n')
            lines = _wrap_text_preserve_breaks(text, font, max_width, stretch_width)
            if lines:
                return int(size), font, lines
        except Exception as e:
            logger.error(f"Ошибка шрифта {size}: {e}")
        size -= 2

    font = ImageFont.truetype(font_path, int(min_size))
    return int(min_size), font, [text]


def _wrap_greedy(words: list, font: ImageFont.FreeTypeFont, max_width: int, stretch: float) -> list:
    """
    Вариант 1 (авто):
    - перенос по ширине
    - запрет "висячих" коротких слов (В/К/С/...) в конце строки:
      если строка заканчивается на такой токен — переносим его вниз.
    """
    if not words:
        return []

    space_w = max(1, _text_width_px(font, " ", spacing=LETTER_SPACING_PX))

    def line_w(ws: list) -> int:
        if not ws:
            return 0
        w = 0
        for j, ww in enumerate(ws):
            ww_w = _text_width_px(font, ww, spacing=LETTER_SPACING_PX)
            w += (space_w if j > 0 else 0) + ww_w
        return int(w * stretch)

    lines = []
    cur = []

    i = 0
    n = len(words)

    while i < n:
        w = words[i]
        if not cur:
            cur = [w]
            i += 1
            continue

        trial = cur + [w]
        if line_w(trial) <= max_width:
            cur = trial
            i += 1
            continue

        # строка переполнена -> фиксируем "висячий" предлог/союз на конце
        if len(cur) >= 2 and _norm_orphan(cur[-1]) in ORPHANS_RU:
            orphan = cur.pop()              # снимаем "В"
            lines.append(" ".join(cur))     # строка без него
            cur = [orphan]                  # новая строка начинается с "В"
            # слово w пока НЕ берём — оно будет добавлено на следующей итерации
        else:
            lines.append(" ".join(cur))
            cur = []

        # не забываем: текущий w ещё не обработан, поэтому не увеличиваем i

    if cur:
        # финальный добивочный фикс: если последняя строка оканчивается на orphan, сдвинем его вниз (если есть куда)
        if len(cur) >= 2 and _norm_orphan(cur[-1]) in ORPHANS_RU and lines:
            orphan = cur.pop()
            lines.append(" ".join(cur))
            lines.append(orphan)
        else:
            lines.append(" ".join(cur))

    return [ln for ln in lines if ln.strip()]


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


def get_fixed_line_metrics(font: ImageFont.FreeTypeFont) -> dict:
    reference_chars = "АбБгДйруфЦЩAgjpqy|"
    ascent, descent = font.getmetrics()
    ref_bbox = font.getbbox(reference_chars)
    real_top = ref_bbox[1]
    real_bottom = ref_bbox[3]
    
    return {
        'ascent': ascent,
        'descent': descent,
        'font_height': ascent + descent,
        'real_top': real_top,
        'real_bottom': real_bottom,
        'total_height': real_bottom - real_top
    }


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
    if not text.strip():
        return 0
    
    metrics = get_fixed_line_metrics(font)
    ascent = metrics['ascent']
    descent = metrics['descent']
    font_h = metrics['font_height']
    
    pad = max(15, shadow_offset + TEXT_OUTLINE_THICKNESS * 2 + 10)
    tw = _text_width_px(font, text, spacing=LETTER_SPACING_PX)
    
    temp_w = tw + pad * 2
    temp_h = font_h + pad * 2
    
    temp = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(temp)
    
    tx = pad
    ty = pad
    
    _draw_text_with_letter_spacing(d, (tx + shadow_offset, ty + shadow_offset), 
                                   text, font, (0, 0, 0, 128), spacing=LETTER_SPACING_PX)
    
    for t in range(int(TEXT_OUTLINE_THICKNESS)):
        r = t + 1
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            _draw_text_with_letter_spacing(d, (tx + dx * r, ty + dy * r), 
                                          text, font, outline_color, spacing=LETTER_SPACING_PX)
    
    _draw_text_with_letter_spacing(d, (tx, ty), text, font, fill_color, spacing=LETTER_SPACING_PX)
    
    if apply_enhancements:
        temp = _apply_text_enhancements(temp)
    
    bb = temp.getbbox()
    if not bb:
        return 0
    
    margin = 3
    crop_left = max(0, bb[0] - margin)
    crop_top = max(0, bb[1] - margin)
    crop_right = min(temp_w, bb[2] + margin)
    crop_bottom = min(temp_h, bb[3] + margin)
    
    crop = temp.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_w, crop_h = crop.size
    
    target_w = max(1, int(crop_w * stretch_width))
    target_h = max(1, int(crop_h * stretch_height))
    
    stretched = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    if apply_enhancements and TEXT_SHARPEN_AMOUNT > 0:
        stretched = _sharpen_image(stretched)
    
    base_image.paste(stretched, (x, y), stretched)
    
    return target_h


def build_stretched_line_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    outline_color: tuple,
    stretch_width: float = TEXT_STRETCH_WIDTH,
    stretch_height: float = TEXT_STRETCH_HEIGHT,
    shadow_offset: int = TEXT_SHADOW_OFFSET,
    apply_enhancements: bool = True,
):
    """
    Returns:
      (line_img_rgba_stretched, baseline_offset_stretched_px)
    baseline_offset = distance from top of the returned image to baseline.
    """
    text = (text or "")
    if not text.strip():
        return None, 0

    ascent, descent = font.getmetrics()
    font_h = ascent + descent

    pad = max(15, shadow_offset + TEXT_OUTLINE_THICKNESS * 2 + 10)
    tw = _text_width_px(font, text, spacing=LETTER_SPACING_PX)
    temp_w = tw + pad * 2
    temp_h = font_h + pad * 2

    # baseline in temp coords
    baseline_y = pad + ascent
    text_top_y = baseline_y - ascent  # == pad

    temp = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(temp)

    tx = pad
    ty = text_top_y

    # shadow
    _draw_text_with_letter_spacing(
        d, (tx + shadow_offset, ty + shadow_offset),
        text, font, (0, 0, 0, 128), spacing=LETTER_SPACING_PX
    )

    # outline
    for t in range(int(TEXT_OUTLINE_THICKNESS)):
        r = t + 1
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]:
            _draw_text_with_letter_spacing(
                d, (tx + dx * r, ty + dy * r),
                text, font, outline_color, spacing=LETTER_SPACING_PX
            )

    # fill
    _draw_text_with_letter_spacing(d, (tx, ty), text, font, fill_color, spacing=LETTER_SPACING_PX)

    if apply_enhancements:
        temp = _apply_text_enhancements(temp)

    bb = temp.getbbox()
    if not bb:
        return None, 0

    # --- FIX: crop X by bbox, but keep Y fixed to a line-box
    margin_x = 3
    crop_left  = max(0, bb[0] - margin_x)
    crop_right = min(temp_w, bb[2] + margin_x)

    # фиксированное вертикальное окно вокруг "нормальной" строки
    extra_y = shadow_offset + int(TEXT_OUTLINE_THICKNESS) * 2 + 6
    crop_top    = max(0, pad - extra_y)
    crop_bottom = min(temp_h, pad + font_h + extra_y)

    crop = temp.crop((crop_left, crop_top, crop_right, crop_bottom))

    baseline_offset = baseline_y - crop_top
    baseline_offset = max(0, baseline_offset)

    # stretch
    target_w = max(1, int(crop.size[0] * stretch_width))
    target_h = max(1, int(crop.size[1] * stretch_height))
    crop = crop.resize((target_w, target_h), Image.LANCZOS)

    baseline_offset_stretched = int(baseline_offset * stretch_height)
    return crop, baseline_offset_stretched


def layout_baseline_block(
    lines: list,
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    outline_color: tuple,
    stretch_width: float = TEXT_STRETCH_WIDTH,
    stretch_height: float = TEXT_STRETCH_HEIGHT,
    shadow_offset: int = TEXT_SHADOW_OFFSET,
    apply_enhancements: bool = True,
    line_spacing: int = LINE_SPACING,
):
    """
    Prepares baseline-aligned layout for multiple lines.
    Returns:
      dict with:
        items: [{img, base_off, baseline_rel}]
        total_h: int
        shift: int  (value added to all baseline_rel to make top = 0)
        step: int   (baseline step)
    """
    ascent, descent = font.getmetrics()

    # constant baseline step (this is the key)
    extra = shadow_offset + int(TEXT_OUTLINE_THICKNESS) * 2 + 6
    line_box_h = ascent + descent + extra
    step = int(line_box_h * stretch_height) + line_spacing

    items_raw = []
    for ln in lines:
        img, base_off = build_stretched_line_image(
            ln, font,
            fill_color=fill_color,
            outline_color=outline_color,
            stretch_width=stretch_width,
            stretch_height=stretch_height,
            shadow_offset=shadow_offset,
            apply_enhancements=apply_enhancements,
        )
        if img is None:
            continue
        items_raw.append((img, base_off))

    if not items_raw:
        return {"items": [], "total_h": 0, "shift": 0, "step": step}

    # compute extents in a baseline coordinate system
    min_top = 10**9
    max_bottom = -10**9
    items = []

    for i, (img, base_off) in enumerate(items_raw):
        baseline_rel = i * step
        top = baseline_rel - base_off
        bottom = top + img.size[1]
        min_top = min(min_top, top)
        max_bottom = max(max_bottom, bottom)
        items.append({"img": img, "base_off": base_off, "baseline_rel": baseline_rel})

    shift = -min_top
    total_h = max_bottom - min_top

    return {"items": items, "total_h": total_h, "shift": shift, "step": step}


def _apply_text_enhancements(img: Image.Image) -> Image.Image:
    if TEXT_INNER_SHADOW_SIZE <= 0 and TEXT_GRAIN_INTENSITY <= 0:
        return img
    
    arr = np.array(img)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    
    if TEXT_INNER_SHADOW_SIZE > 0:
        kernel = np.ones((TEXT_INNER_SHADOW_SIZE * 2 + 1, TEXT_INNER_SHADOW_SIZE * 2 + 1), np.uint8)
        eroded = cv2.erode(alpha, kernel, iterations=1)
        inner_shadow_mask = (alpha > 0) & (eroded == 0)
        arr[inner_shadow_mask, :3] = (arr[inner_shadow_mask, :3] * 0.7).astype(np.uint8)
    
    if TEXT_GRAIN_INTENSITY > 0:
        noise = np.random.normal(0, TEXT_GRAIN_INTENSITY * 25, (h, w, 3))
        text_mask = alpha > 0
        arr_float = arr[:, :, :3].astype(np.float32)
        arr_float[text_mask] += noise[text_mask]
        arr[:, :, :3] = np.clip(arr_float, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def _sharpen_image(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    rgb = arr[:, :, :3]
    blurred = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    sharpened = rgb + TEXT_SHARPEN_AMOUNT * (rgb - blurred)
    sharpened = np.clip(sharpened, 0, 255)
    arr[:, :, :3] = sharpened
    return Image.fromarray(arr.astype(np.uint8))


def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()
    _, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE1, stretch_width=TEXT_STRETCH_WIDTH
    )

    # baseline layout for title block
    title_layout = layout_baseline_block(
        title_lines, title_font,
        fill_color=COLOR_TURQUOISE,
        outline_color=COLOR_OUTLINE,
        stretch_width=TEXT_STRETCH_WIDTH,
        stretch_height=TEXT_STRETCH_HEIGHT,
        shadow_offset=TEXT_SHADOW_OFFSET,
        apply_enhancements=True,
        line_spacing=LINE_SPACING,
    )
    total_title_h = title_layout["total_h"]

    # logo block
    draw = ImageDraw.Draw(image, "RGBA")
    logo_font = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOGO)
    logo_text = "@neurostep.media"
    bb = logo_font.getbbox(logo_text)
    logo_w = bb[2] - bb[0]
    logo_h = bb[3] - bb[1]

    total_h = logo_h + SPACING_LOGO_TO_TITLE + total_title_h
    start_y = height - SPACING_BOTTOM_MODE1 - total_h

    logo_x = (width - logo_w) // 2
    logo_y = start_y

    # lines around logo
    line_y = logo_y + logo_h // 2
    line_left_start = logo_x - LOGO_LINE_LENGTH - 10
    line_right_start = logo_x + logo_w + 10

    draw.line(
        [(line_left_start, line_y), (line_left_start + LOGO_LINE_LENGTH, line_y)],
        fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX
    )
    draw.line(
        [(line_right_start, line_y), (line_right_start + LOGO_LINE_LENGTH, line_y)],
        fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX
    )
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)

    # paste title lines baseline-aligned
    block_top = start_y + logo_h + SPACING_LOGO_TO_TITLE
    for it in title_layout["items"]:
        img = it["img"]
        baseline = block_top + title_layout["shift"] + it["baseline_rel"]
        y = baseline - it["base_off"]
        x = (width - img.size[0]) // 2
        image.alpha_composite(img, (int(x), int(y)))

    return image



def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()
    _, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE2, stretch_width=TEXT_STRETCH_WIDTH
    )

    layout = layout_baseline_block(
        title_lines, title_font,
        fill_color=COLOR_TURQUOISE,
        outline_color=COLOR_OUTLINE,
        stretch_width=TEXT_STRETCH_WIDTH,
        stretch_height=TEXT_STRETCH_HEIGHT,
        shadow_offset=TEXT_SHADOW_OFFSET,
        apply_enhancements=True,
        line_spacing=LINE_SPACING,
    )

    total_h = layout["total_h"]
    start_y = height - SPACING_BOTTOM_MODE2 - total_h

    for it in layout["items"]:
        img = it["img"]
        baseline = start_y + layout["shift"] + it["baseline_rel"]
        y = baseline - it["base_off"]
        x = (width - img.size[0]) // 2
        image.alpha_composite(img, (int(x), int(y)))

    return image


def render_mode3_content(image: Image.Image, title_translated: str, subtitle_translated: str) -> Image.Image:
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper().strip()
    subtitle = (subtitle_translated or "").upper().strip()

    # 1) Подбираем шрифт/переносы для title
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE3_TITLE, stretch_width=TEXT_STRETCH_WIDTH
    )

    # 2) Подбираем шрифт/переносы для subtitle (может стать 2+ строками — это ок)
    subtitle_initial = int(title_size * 0.80)
    _, subtitle_font, subtitle_lines = calculate_adaptive_font_size(
        subtitle, FONT_PATH, max_text_width, subtitle_initial, stretch_width=TEXT_STRETCH_WIDTH
    )

    # 3) Строим baseline-layout для обоих блоков (это и есть “фикс” межстрочника/оптики)
    title_layout = layout_baseline_block(
        title_lines, title_font,
        fill_color=COLOR_TURQUOISE,
        outline_color=COLOR_OUTLINE,
        stretch_width=TEXT_STRETCH_WIDTH,
        stretch_height=TEXT_STRETCH_HEIGHT,
        shadow_offset=TEXT_SHADOW_OFFSET,
        apply_enhancements=True,
        line_spacing=LINE_SPACING,
    )

    sub_layout = layout_baseline_block(
        subtitle_lines, subtitle_font,
        fill_color=COLOR_WHITE,
        outline_color=COLOR_OUTLINE,
        stretch_width=TEXT_STRETCH_WIDTH,
        stretch_height=TEXT_STRETCH_HEIGHT,
        shadow_offset=TEXT_SHADOW_OFFSET,
        apply_enhancements=True,
        line_spacing=LINE_SPACING_SUBTITLE,
    )

    gap = SPACING_TITLE_TO_SUBTITLE if (title_layout["items"] and sub_layout["items"]) else 0
    total_h = title_layout["total_h"] + gap + sub_layout["total_h"]
    start_y = height - SPACING_BOTTOM_MODE3 - total_h

    # 4) Рисуем title (baseline-aligned)
    block_top = start_y
    for it in title_layout["items"]:
        img = it["img"]
        baseline = block_top + title_layout["shift"] + it["baseline_rel"]
        y = baseline - it["base_off"]
        x = (width - img.size[0]) // 2
        image.alpha_composite(img, (int(x), int(y)))

    # 5) Рисуем subtitle (baseline-aligned) — ключевой момент для 2+ строк
    block_top = start_y + title_layout["total_h"] + gap
    for it in sub_layout["items"]:
        img = it["img"]
        baseline = block_top + sub_layout["shift"] + it["baseline_rel"]
        y = baseline - it["base_off"]
        x = (width - img.size[0]) // 2
        image.alpha_composite(img, (int(x), int(y)))

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
    
    if mode == 1:
        mask_percent = MASK_BOTTOM_MODE1
    elif mode == 2:
        mask_percent = MASK_BOTTOM_MODE2
    else:
        mask_percent = MASK_BOTTOM_MODE3
        
    mask_start = int(h * (1 - mask_percent / 100))
    mask[mask_start:, :] = 255
    logger.info(f"📐 Маска: строки {mask_start}-{h} (нижние {mask_percent}%)")

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
