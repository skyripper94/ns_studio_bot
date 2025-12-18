# lama_integration.py
"""
Полный workflow для обработки изображений:
1) OCR (Google Vision) по нижней части
2) Inpaint (Replicate FLUX Fill) ТОЛЬКО по маске (нижние N%)
3) Перевод (OpenAI)
4) Наложение градиента (точно на нижние N%)
5) Отрисовка текста/линий/лого по режимам

ВАЖНО:
- Ваше "мыло" в логе появилось потому что Replicate вернул 401 Invalid token → сработал OpenCV fallback (он всегда мажет на большой маске).
- Модель flux-kontext-pro НЕ поддерживает mask, поэтому могла “лезть” за пределы области. Для масочного инпейнта нужно flux-fill-pro.
"""

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

logger = logging.getLogger(__name__)

"""
==============================================
НАСТРОЙКИ ДЛЯ БЫСТРОЙ РУЧНОЙ ПРАВКИ
(все ключевые коэффициенты вынесены сюда)
==============================================
"""

# ============== API КЛЮЧИ ==============
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ============== REPLICATE / FLUX (INPAINT) ==============
# МОДЕЛЬ ДЛЯ МАСКОВОГО INPAINT:
# flux-kontext-pro — это “edit”, без маски; для маски нужно flux-fill-pro.
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "black-forest-labs/flux-fill-pro").strip()  # поменять если надо
FLUX_STEPS = int(os.getenv("FLUX_STEPS", "50"))      # 15..50 (больше = детальнее, медленнее; у модели max=50)
FLUX_GUIDANCE = float(os.getenv("FLUX_GUIDANCE", "25"))  # 1.5..100 (по умолчанию у модели 60; выше = сильнее следует промпту, но может портить качество)
FLUX_OUTPUT_FORMAT = os.getenv("FLUX_OUTPUT_FORMAT", "png")  # png = без потерь
FLUX_PROMPT_UPSAMPLING = False  # True = творчески “додумает” промпт, обычно не надо для чистки
REPLICATE_HTTP_TIMEOUT = int(os.getenv("REPLICATE_HTTP_TIMEOUT", "120"))  # таймаут скачивания результата

# Жёсткая гарантия: “всё вне маски НЕ меняем”, даже если модель попыталась
FORCE_PRESERVE_OUTSIDE_MASK = True

# ============== ЦВЕТА ==============
COLOR_TURQUOISE = (0, 206, 209)  # Бирюзовый для заголовков
COLOR_WHITE = (255, 255, 255)    # Белый для подзаголовков/лого
COLOR_OUTLINE = (60, 60, 60)     # Обводка текста (#3C3C3C)

# ============== РАЗМЕРЫ ШРИФТОВ ==============
FONT_SIZE_MODE1 = 58             # Заголовок в режиме 1 (лого)
FONT_SIZE_MODE2 = 52             # Заголовок в режиме 2 (только текст)
FONT_SIZE_MODE3_TITLE = 54       # Заголовок в режиме 3
FONT_SIZE_MODE3_SUBTITLE = 52    # Подзаголовок в режиме 3
FONT_SIZE_LOGO = 24              # Размер @neurostep.media
FONT_SIZE_MIN = 44               # Минимальный размер при автоподборе (уменьшить = мельче)

# ============== ОТСТУПЫ И РАССТОЯНИЯ ==============
SPACING_BOTTOM = 40             # Отступ снизу до композиции
SPACING_LOGO_TO_TITLE = 6        # Между логотипом и заголовком
SPACING_TITLE_TO_SUBTITLE = 10   # Между заголовком и подзаголовком
LINE_SPACING = -16                # Между строками
LOGO_LINE_LENGTH = 320           # Длина линий возле лого
LOGO_LINE_THICKNESS_PX = 3   # толщина полос возле логотипа (@neurostep.media)

# ============== МАСКА / OCR ==============
MASK_BOTTOM_PERCENT = 32         # Сколько % снизу чистим (маска)
OCR_BOTTOM_PERCENT = 32          # OCR зона снизу (держать равной маске)

# ============== ГРАДИЕНТ ==============
# Градиент покрывает ТОЛЬКО нижние MASK_BOTTOM_PERCENT, как вы описали
GRADIENT_COVER_PERCENT = 50      # если хотите отдельно — меняйте; по умолчанию = 35%
GRADIENT_SOLID_FRACTION = 0.35   # какая часть градиента снизу 100% непрозрачная (0.5 = нижняя половина)
GRADIENT_SOLID_RAISE_PX = int(os.getenv("GRADIENT_SOLID_RAISE_PX", "125"))  # ↑ границу "чёрной основы" на N px (скрыть артефакты)
GRADIENT_INTENSITY_CURVE = 2.6   # плавность в верхней половине (больше = резче переход)

# ============== РАСТЯЖЕНИЕ ТЕКСТА ==============
TEXT_STRETCH_HEIGHT = 1.3       # +25% по высоте
TEXT_STRETCH_WIDTH = 1.15        # +10% по ширине

# ============== ТЕНИ / ОБВОДКИ ==============
TEXT_SHADOW_OFFSET = 1           # Смещение тени (больше = дальше тень)
TEXT_OUTLINE_THICKNESS = 1       # Толщина обводки (увеличить = жирнее)

# ============== БЛОК ТЕКСТА ==============
TEXT_WIDTH_PERCENT = 0.90        # Ширина блока текста от ширины картинки

# ============== OPENCV FALLBACK ==============
# Если Replicate недоступен/упал, включается fallback. На большой маске идеала не будет.
OPENCV_BLUR_SIGMA = 5            # Блюр внутри маски (больше = сильнее “съест” артефакты)
OPENCV_INPAINT_RADIUS = 3        # Радиус инпейнта (больше = сильнее “мажет”)

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
    """OCR через Google Vision API по нижней части изображения (чтобы не ловить весь кадр)."""
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
        # Логи не любят \n внутри одной строки — поэтому lines отдельным списком
        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]

        # Небольшая фильтрация типовых “служебных” строк, если они попали в OCR
        if lines and lines[0].strip().lower() in {"wealth", "@neurostep.media"}:
            lines = lines[1:]
            full_text = "\n".join(lines)

        logger.info(f"📝 OCR строки: {len(lines)}")
        return {"text": full_text.strip(), "lines": lines}

    except Exception as e:
        logger.error(f"❌ Ошибка Google Vision OCR: {e}")
        return {"text": "", "lines": []}


# ---------------------------------------------------------------------
# Перевод (OpenAI)
# ---------------------------------------------------------------------
def openai_translate(text: str) -> str:
    """Перевод и адаптация под СНГ (коротко, по смыслу, без лишнего)."""
    if not OPENAI_API_KEY or not text:
        logger.warning("⚠️ OPENAI_API_KEY не установлен или нет текста")
        return text

    try:
        logger.info(f"🌐 Перевод: {text}")

        # ВАЖНО: мы не делаем дословный перевод — мы редактируем заголовок под обложку.
        # Формат нужен стабильный, чтобы верстка не "плясала".
        system_prompt = """Ты редактор заголовков для обложек (СНГ) в стиле Wealth: спокойно, уверенно, без кликбейта.
Задача: не перевод, а короткая сильная формулировка, которая хорошо смотрится крупным КАПСОМ в 3 строки.

Правила:
1) Верни РОВНО 3 строки, разделяй только символом \n.
2) Длина каждой строки: 10–18 символов (без пробелов) — если длиннее, перефразируй.
3) Избегай длинных слов (желательно до 12–13 букв). Заменяй на короткие синонимы.
4) billion→МЛРД., million→МЛН. (в капсе).
5) Никаких “ВАС/ТЕБЯ”, никаких “заставит/шок/рот/не поверите”.
6) Если вход уже на русском — улучши и сократи, не “переводи”.
7) Верни только заголовок, без кавычек и пояснений.
"""

        resp = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Заголовок для обложки: {text}"},
            ],
            temperature=0.4,
            max_tokens=200,
        )

        translated = resp.choices[0].message.content.strip()
        logger.info(f"✅ Переведено: {translated}")
        return translated

    except Exception as e:
        logger.error(f"❌ Ошибка OpenAI перевода: {e}")
        return text

# ---------------------------------------------------------------------
# OpenCV fallback (когда Replicate не работает)
# ---------------------------------------------------------------------
def opencv_fallback(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Запасной вариант без Replicate.
    Логика: внутри маски размываем + лёгкий inpaint, чтобы не было “грязи”.
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    result = image_bgr.copy()

    # Размываем только область маски (так меньше “мыла”, чем у полного inpaint на огромной маске)
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=OPENCV_BLUR_SIGMA, sigmaY=OPENCV_BLUR_SIGMA)
    result[mask_u8 == 255] = blurred[mask_u8 == 255]

    # Лёгкий inpaint поверх (радиус маленький, чтобы не “плыла” текстура)
    try:
        result = cv2.inpaint(result, mask_u8, inpaintRadius=OPENCV_INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
    except Exception:
        pass

    logger.info("✅ OpenCV fallback (blur + light inpaint)")
    return result


# ---------------------------------------------------------------------
# Replicate FLUX Fill (масочный inpaint)
# ---------------------------------------------------------------------
def flux_inpaint(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Inpaint через Replicate на модели FLUX Fill.
    Гарантия: если FORCE_PRESERVE_OUTSIDE_MASK=True — вне маски возвращаем оригинал пиксель-в-пиксель.
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN не установлен → fallback OpenCV")
        return opencv_fallback(image_bgr, mask_u8)

    try:
        import replicate  # локальный импорт, чтобы проект стартовал даже без replicate в окружении

        # Клиент с явным токеном (на Railway так надёжнее)
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)

        logger.info(f"🚀 Replicate inpaint: {REPLICATE_MODEL}")

        # Важный момент: модель сама не “понимает” вашу бизнес-логику.
        # Мы просим удалить текст/линии/логотипы (внутри маски), восстановить фон, без размытия.
        prompt = (
            "Remove all text, decorative lines and logos in the masked region. "
            "Reconstruct the original background naturally with clean, sharp detail. "
            "Match lighting, texture, and perspective. No blur, no smears, no artifacts, no repeating patterns. "
            "Do not change anything outside the mask. "
        )

        # Изображение в PNG без потерь
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_buf = BytesIO()
        pil_img.save(img_buf, format="PNG", compress_level=0)
        img_buf.seek(0)

        # Маска (белое = инпейнт, чёрное = сохранить)
        pil_mask = Image.fromarray(mask_u8, mode="L")
        mask_buf = BytesIO()
        pil_mask.save(mask_buf, format="PNG", compress_level=0)
        mask_buf.seek(0)

        # ВАЖНО: у flux-fill-pro поля называются image/mask/steps/guidance (не input_image/num_inference_steps).
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

        # output обычно = URL (string)
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

        # Если модель вдруг изменила размер — возвращаем в исходный (иначе будет мыло/скейл)
        if out_bgr.shape[:2] != image_bgr.shape[:2]:
            logger.warning("⚠️ Replicate изменил размер → ресайз обратно (LANCZOS)")
            out_bgr = cv2.resize(out_bgr, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Жёстко сохраняем всё вне маски (решает вашу проблему с “логотипами выше маски”)
        if FORCE_PRESERVE_OUTSIDE_MASK:
            out_bgr = _composite_by_mask(image_bgr, out_bgr, mask_u8)

        logger.info("✅ Replicate inpaint OK")
        return out_bgr

    except Exception as e:
        # Типовая причина у вас: 401 Invalid token → проверяйте переменную окружения REPLICATE_API_TOKEN в Railway.
        logger.error(f"❌ Ошибка Replicate inpaint: {e}")
        return opencv_fallback(image_bgr, mask_u8)


def _composite_by_mask(original_bgr: np.ndarray, edited_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Смешивание по маске: берём edited только там, где mask=255; снаружи — оригинал."""
    m = (mask_u8.astype(np.float32) / 255.0)[:, :, None]
    out = (original_bgr.astype(np.float32) * (1.0 - m) + edited_bgr.astype(np.float32) * m)
    return np.clip(out, 0, 255).astype(np.uint8)


# Совместимость со старым именем (чтобы не менять остальной проект)
def flux_kontext_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ALIАS: раньше было flux-kontext-pro, теперь правильный масочный inpaint = flux-fill-pro."""
    return flux_inpaint(image, mask)


# ---------------------------------------------------------------------
# Градиент (быстро, без покадрового putpixel)
# ---------------------------------------------------------------------
def create_gradient_layer(width: int, height: int,
                          cover_percent: int = GRADIENT_COVER_PERCENT) -> Image.Image:
    """
    Создаёт RGBA-слой градиента для нижних cover_percent%.
    Низ: alpha=255, нижняя половина градиента — 100% непрозрачная,
    верхняя половина — плавный уход в 0.
    """
    cover_percent = int(np.clip(cover_percent, 1, 100))
    start_row = int(height * (1 - cover_percent / 100))
    grad_h = max(1, height - start_row)

    y = np.arange(height, dtype=np.float32)
    t = (y - start_row) / float(grad_h)   # 0 вверху градиента → 1 внизу
    t = np.clip(t, 0.0, 1.0)

    # Нижняя часть — 100% непрозрачная
    # Базовая граница (по доле): на какой высоте начинается 100% чёрный слой.
    base_solid_from = 1.0 - float(np.clip(GRADIENT_SOLID_FRACTION, 0.0, 1.0))
    # Поднимаем границу вверх на фиксированное кол-во пикселей (чтобы скрыть артефакты под градиентом).
    raise_t = float(np.clip(GRADIENT_SOLID_RAISE_PX, 0, height)) / float(grad_h)
    solid_from = float(np.clip(base_solid_from - raise_t, 0.0, 1.0))

    # Преобразуем к шкале “верхняя часть до границы”
    top_part = np.clip(t / max(solid_from, 1e-6), 0.0, 1.0)
    alpha = np.where(
        t >= solid_from,
        255.0,
        255.0 * (top_part ** float(GRADIENT_INTENSITY_CURVE)),
    ).astype(np.uint8)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha[:, None]  # только альфа, цвет = чёрный

    logger.info(f"✨ Градиент: cover={cover_percent}%, start_row={start_row}, solid_from={solid_from:.3f}, raise_px={GRADIENT_SOLID_RAISE_PX}")
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------
# Текст: подбор размера и отрисовка со “stretch”
# ---------------------------------------------------------------------
def calculate_adaptive_font_size(text: str, font_path: str, max_width: int,
                                 initial_size: int, min_size: int = FONT_SIZE_MIN,
                                 stretch_width: float = TEXT_STRETCH_WIDTH) -> tuple:
    """
    Автоподбор размера шрифта под ширину с учётом будущего растяжения по ширине.
    Возвращает: (size, font, lines)
    """
    size = int(initial_size)

    while size >= min_size:
        try:
            font = ImageFont.truetype(font_path, size)
            words = text.split()
            lines = []
            cur = []

            for w in words:
                test = " ".join(cur + [w])
                bbox = font.getbbox(test)
                w0 = bbox[2] - bbox[0]
                # ВАЖНО: учитываем будущий stretch по ширине
                if int(w0 * stretch_width) <= max_width:
                    cur.append(w)
                else:
                    if cur:
                        lines.append(" ".join(cur))
                        cur = [w]
                    else:
                        lines.append(w)
                        cur = []

            if cur:
                lines.append(" ".join(cur))

            # Проверяем, что каждая строка влезет после stretch
            fits = True
            for ln in lines:
                bbox = font.getbbox(ln)
                w0 = bbox[2] - bbox[0]
                if int(w0 * stretch_width) > max_width:
                    fits = False
                    break

            if fits:
                return size, font, lines

        except Exception as e:
            logger.error(f"Ошибка шрифта {size}: {e}")

        size -= 2

    font = ImageFont.truetype(font_path, min_size)
    return min_size, font, [text]


def draw_text_with_stretch(base_image: Image.Image,
                           x: int, y: int,
                           text: str,
                           font: ImageFont.FreeTypeFont,
                           fill_color: tuple,
                           outline_color: tuple,
                           stretch_width: float = TEXT_STRETCH_WIDTH,
                           stretch_height: float = TEXT_STRETCH_HEIGHT,
                           shadow_offset: int = TEXT_SHADOW_OFFSET) -> int:
    """
    Рисует текст с тенью+обводкой, затем растягивает общий “слой текста”.
    Возвращает итоговую высоту нарисованного (после stretch).
    """
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Запас по размеру, чтобы не обрезать тень/обводку
    pad = max(6, shadow_offset + TEXT_OUTLINE_THICKNESS * 2)
    temp_w = int(tw * (stretch_width + 1.0)) + pad * 2
    temp_h = int(th * (stretch_height + 1.0)) + pad * 2

    temp = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(temp)

    # Рисуем ближе к левому/верхнему с паддингом
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

    # Обрезаем по контенту
    bb = temp.getbbox()
    if not bb:
        return th

    crop = temp.crop(bb)

    # Растягиваем
    sw = max(1, int(crop.width * stretch_width))
    sh = max(1, int(crop.height * stretch_height))
    crop = crop.resize((sw, sh), Image.Resampling.LANCZOS)

    # Позиция: x,y считаются как “верхний левый” примерно под центрирование строк
    base_image.paste(crop, (x, y), crop)
    return sh

def _split_manual_lines(text: str) -> list:
    """Если текст уже содержит ручные переносы (\n) — сохраняем их, иначе возвращаем []."""
    if not text:
        return []
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines if len(lines) >= 2 else []

def _estimate_fixed_line_height(font: ImageFont.FreeTypeFont) -> int:
    """Фиксированная высота строки (чтобы межстрочный не 'плясал' от букв/кропа)."""
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

    manual_lines = _split_manual_lines(title)
    if manual_lines:
        title_lines = manual_lines
        size = int(FONT_SIZE_MODE1)
        title_font = ImageFont.truetype(FONT_PATH, size)
        while size >= FONT_SIZE_MIN:
            title_font = ImageFont.truetype(FONT_PATH, size)
            ok = True
            for ln in title_lines:
                bb = title_font.getbbox(ln)
                ln_w = bb[2] - bb[0]
                if int(ln_w * TEXT_STRETCH_WIDTH) > max_text_width:
                    ok = False
                    break
            if ok:
                break
            size -= 2
    else:
        _, title_font, title_lines = calculate_adaptive_font_size(
            title, FONT_PATH, max_text_width, FONT_SIZE_MODE1, stretch_width=TEXT_STRETCH_WIDTH
        )    # высота заголовка (фиксированная, чтобы межстрочный не плясал)
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

    # Линии по центру лого
    line_y = logo_y + logo_h // 2
    line_left_start = logo_x - LOGO_LINE_LENGTH - 10
    line_right_start = logo_x + logo_w + 10

    draw.line([(line_left_start, line_y), (line_left_start + LOGO_LINE_LENGTH, line_y)], fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX)
    draw.line([(line_right_start, line_y), (line_right_start + LOGO_LINE_LENGTH, line_y)], fill=COLOR_TURQUOISE, width=LOGO_LINE_THICKNESS_PX)

    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)

    # Заголовок
    cur_y = start_y + logo_h + SPACING_LOGO_TO_TITLE
    for ln in title_lines:
        bb = title_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        draw_text_with_stretch(image, x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h + LINE_SPACING

    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """Режим 2: только заголовок (UPPERCASE)."""
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()

    manual_lines = _split_manual_lines(title)
    if manual_lines:
        title_lines = manual_lines
        # Единый размер шрифта для всех строк: подбираем под самую широкую строку (чтобы не было разного кегля).
        size = int(FONT_SIZE_MODE2)
        title_font = ImageFont.truetype(FONT_PATH, size)
        while size >= FONT_SIZE_MIN:
            title_font = ImageFont.truetype(FONT_PATH, size)
            ok = True
            for ln in title_lines:
                bb = title_font.getbbox(ln)
                ln_w = bb[2] - bb[0]
                if int(ln_w * TEXT_STRETCH_WIDTH) > max_text_width:
                    ok = False
                    break
            if ok:
                break
            size -= 2
    else:
        _, title_font, title_lines = calculate_adaptive_font_size(
            title, FONT_PATH, max_text_width, FONT_SIZE_MODE2, stretch_width=TEXT_STRETCH_WIDTH
        )

    line_h = _estimate_fixed_line_height(title_font)
    total_h = line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)

    start_y = height - SPACING_BOTTOM - total_h
    cur_y = start_y
    for ln in title_lines:
        bb = title_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        draw_text_with_stretch(image, x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h + LINE_SPACING

    return image


def render_mode3_content(image: Image.Image, title_translated: str, subtitle_translated: str) -> Image.Image:
    """Режим 3: заголовок + подзаголовок (оба UPPERCASE)."""
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()
    subtitle = (subtitle_translated or "").upper()

    manual_title = _split_manual_lines(title)
    if manual_title:
        title_lines = manual_title
        size = int(FONT_SIZE_MODE3_TITLE)
        title_font = ImageFont.truetype(FONT_PATH, size)
        while size >= FONT_SIZE_MIN:
            title_font = ImageFont.truetype(FONT_PATH, size)
            ok = True
            for ln in title_lines:
                bb = title_font.getbbox(ln)
                ln_w = bb[2] - bb[0]
                if int(ln_w * TEXT_STRETCH_WIDTH) > max_text_width:
                    ok = False
                    break
            if ok:
                break
            size -= 2
        title_size = size
    else:
        title_size, title_font, title_lines = calculate_adaptive_font_size(
            title, FONT_PATH, max_text_width, FONT_SIZE_MODE3_TITLE, stretch_width=TEXT_STRETCH_WIDTH
        )

        subtitle_initial = int(title_size * 0.80)
    manual_sub = _split_manual_lines(subtitle)
    if manual_sub:
        subtitle_lines = manual_sub
        size = int(subtitle_initial)
        subtitle_font = ImageFont.truetype(FONT_PATH, size)
        while size >= FONT_SIZE_MIN:
            subtitle_font = ImageFont.truetype(FONT_PATH, size)
            ok = True
            for ln in subtitle_lines:
                bb = subtitle_font.getbbox(ln)
                ln_w = bb[2] - bb[0]
                if int(ln_w * TEXT_STRETCH_WIDTH) > max_text_width:
                    ok = False
                    break
            if ok:
                break
            size -= 2
    else:
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
    for ln in title_lines:
        bb = title_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        draw_text_with_stretch(image, x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += title_line_h + LINE_SPACING

    cur_y += SPACING_TITLE_TO_SUBTITLE

    for ln in subtitle_lines:
        bb = subtitle_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        draw_text_with_stretch(image, x, cur_y, ln, subtitle_font, COLOR_WHITE, COLOR_OUTLINE)
        cur_y += sub_line_h + LINE_SPACING

    return image


# ---------------------------------------------------------------------
# ОСНОВНОЙ WORKFLOW
# ---------------------------------------------------------------------
def process_full_workflow(image_bgr: np.ndarray, mode: int) -> tuple:
    """
    Полный workflow для режимов 1,2,3.

    Режимы:
    1 — лого + заголовок
    2 — только заголовок
    3 — заголовок + подзаголовок
    """
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


# Совместимость (в проекте может где-то вызываться старое имя)
def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Алиас для inpaint."""
    return flux_inpaint(image, mask)
