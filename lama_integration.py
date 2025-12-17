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
FLUX_GUIDANCE = float(os.getenv("FLUX_GUIDANCE", "20"))  # 10..35 для inpaint (выше = больше артефактов)
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
FONT_SIZE_MODE1 = 54             # Заголовок в режиме 1 (лого)
FONT_SIZE_MODE2 = 52             # Заголовок в режиме 2 (только текст)
FONT_SIZE_MODE3_TITLE = 52       # Заголовок в режиме 3
FONT_SIZE_MODE3_SUBTITLE = 50    # Подзаголовок в режиме 3
FONT_SIZE_LOGO = 24              # Размер @neurostep.media
FONT_SIZE_MIN = 44               # Минимальный размер при автоподборе (уменьшить = мельче)

# ============== ОТСТУПЫ И РАССТОЯНИЯ ==============
SPACING_BOTTOM = 120             # Отступ снизу до композиции
SPACING_LOGO_TO_TITLE = 6        # Между логотипом и заголовком
SPACING_TITLE_TO_SUBTITLE = 10   # Между заголовком и подзаголовком
LINE_SPACING = 12                # Между строками
LOGO_LINE_LENGTH = 300           # Длина линий возле лого
LOGO_LINE_THICKNESS_PX = 2   # толщина полос возле логотипа (@neurostep.media)

# ============== МАСКА / OCR ==============
MASK_BOTTOM_PERCENT = int(os.getenv("MASK_BOTTOM_PERCENT", "32"))  # % снизу чистим (маска)
OCR_BOTTOM_PERCENT = int(os.getenv("OCR_BOTTOM_PERCENT", str(MASK_BOTTOM_PERCENT)))  # OCR-зона снизу (по умолчанию = маске)

# ============== ГРАДИЕНТ ==============
# Градиент покрывает ТОЛЬКО нижние MASK_BOTTOM_PERCENT, как вы описали
GRADIENT_COVER_PERCENT = int(os.getenv("GRADIENT_COVER_PERCENT", str(MASK_BOTTOM_PERCENT)))  # по умолчанию = маске
GRADIENT_SOLID_FRACTION = 0.50   # какая часть градиента снизу 100% непрозрачная (0.5 = нижняя половина)
GRADIENT_SOLID_RAISE_PX = int(os.getenv("GRADIENT_SOLID_RAISE_PX", "120"))  # ↑ границу "чёрной основы" на N px (скрыть артефакты)
GRADIENT_INTENSITY_CURVE = 1.6   # плавность в верхней половине (больше = резче переход)

# ============== РАСТЯЖЕНИЕ ТЕКСТА ==============
TEXT_STRETCH_HEIGHT = 1.25       # +25% по высоте
TEXT_STRETCH_WIDTH = 1.15        # +10% по ширине

# ============== ТЕНИ / ОБВОДКИ ==============
TEXT_SHADOW_OFFSET = 2           # Смещение тени (больше = дальше тень)
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
    """Редактор заголовков для обложек (СНГ), формат: 3 строки."""
    if not OPENAI_API_KEY or not text:
        logger.warning("⚠️ OPENAI_API_KEY не установлен или нет текста")
        return text

    try:
        logger.info(f"🌐 Заголовок (LLM): {text}")

        system_prompt = """Ты редактор заголовков для обложек (СНГ) в стиле Wealth: спокойно, уверенно, без кликбейта.
Задача: сделать короткий заголовок, который хорошо выглядит крупным КАПСОМ в 3 строки.

Правила:
1) Верни РОВНО 3 строки. Разделитель строк: \n.
2) Каждая строка короткая. Если не влезает — перефразируй.
3) Избегай длинных слов (желательно до 12–13 букв). Заменяй на короткие синонимы.
4) billion→МЛРД., million→МЛН. (в капсе).
5) Никаких обращений к зрителю: никаких "ВАС/ТЕБЯ", никаких "заставит", "шок", "рот", "не поверите".
6) Если вход уже на русском — улучши и сократи, не перевод.
7) Верни только заголовок, без кавычек, без пояснений, без точки в конце.
"""

        resp = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Заголовок для обложки: {text}"},
            ],
            temperature=0.2,
            max_tokens=80,
        )

        out = (resp.choices[0].message.content or "").strip()

        # Нормализация: убрать пустые строки/пробелы, сохранить переносы
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if len(lines) >= 3:
            out = "\n".join(lines[:3])
        else:
            out = "\n".join(lines) if lines else text

        logger.info(f"✅ Заголовок: {out}")
        return out

    except Exception as e:
        logger.error(f"❌ Ошибка OpenAI translate: {e}")
        return text

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
            "Do not change anything outside the mask."
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
                "guidance": float(np.clip(FLUX_GUIDANCE, 1.5, 35)),
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
def split_manual_lines(text: str) -> list[str]:
    """Сохраняет ручные переносы строк. Пустые строки убирает."""
    if not text:
        return []
    return [ln.strip() for ln in str(text).splitlines() if ln.strip()]


def compute_fixed_line_height(font: ImageFont.FreeTypeFont,
                              stretch_height: float = TEXT_STRETCH_HEIGHT,
                              extra_pad: int = 0) -> int:
    """Фиксированная высота строки по метрикам шрифта (стабильный межстрочник)."""
    ascent, descent = font.getmetrics()
    base = int((ascent + descent) * float(stretch_height))
    return max(1, base + int(extra_pad))



def calculate_adaptive_font_size(text: str,
                                 font_path: str,
                                 max_width: int,
                                 initial_size: int,
                                 min_size: int = FONT_SIZE_MIN,
                                 stretch_width: float = TEXT_STRETCH_WIDTH) -> tuple:
    """
    Автоподбор размера шрифта под ширину с учётом будущего растяжения по ширине.
    ВАЖНО: если в тексте есть \n — считаем строки ручными и не делаем автоперенос по словам.
    Возвращает: (size, font, lines)
    """
    size = int(initial_size)
    text = text or ""

    manual_lines = split_manual_lines(text)
    has_manual = ("\n" in text) and (len(manual_lines) >= 2)

    while size >= min_size:
        try:
            font = ImageFont.truetype(font_path, size)

            # 1) Ручные строки: единый размер шрифта под самую длинную строку
            if has_manual:
                max_line_w = 0
                for ln in manual_lines:
                    bb = font.getbbox(ln)
                    ln_w = max(0, bb[2] - bb[0])
                    max_line_w = max(max_line_w, int(ln_w * stretch_width))
                if max_line_w <= max_width:
                    return size, font, manual_lines

            # 2) Автоперенос: только если нет ручных переносов
            else:
                words = text.split()
                lines = []
                cur = []

                for w in words:
                    cand = " ".join(cur + [w])
                    bb = font.getbbox(cand)
                    cand_w = int((bb[2] - bb[0]) * stretch_width)
                    if cand_w <= max_width or not cur:
                        cur.append(w)
                    else:
                        lines.append(" ".join(cur))
                        cur = [w]
                if cur:
                    lines.append(" ".join(cur))

                # Каждая строка должна влезать
                ok = True
                for ln in lines:
                    bb = font.getbbox(ln)
                    ln_w = int((bb[2] - bb[0]) * stretch_width)
                    if ln_w > max_width:
                        ok = False
                        break
                if ok:
                    return size, font, lines

        except Exception as e:
            logger.error(f"Ошибка шрифта {size}: {e}")

        size -= 2

    font = ImageFont.truetype(font_path, min_size)
    fallback_lines = split_manual_lines(text) or [text]
    return min_size, font, fallback_lines

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


# ---------------------------------------------------------------------
# РЕНДЕРЫ РЕЖИМОВ
# ---------------------------------------------------------------------
def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    """Режим 1: Лого + линии + заголовок (UPPERCASE). Уважает ручные переносы (\n)."""
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()

    _, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE1, stretch_width=TEXT_STRETCH_WIDTH
    )

    # Фиксированный межстрочный (не пляшет от букв/кропа)
    title_line_h = compute_fixed_line_height(
        title_font, TEXT_STRETCH_HEIGHT, extra_pad=(TEXT_SHADOW_OFFSET * 2 + 6)
    )
    total_title_h = title_line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)

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

    # Линии возле лого
    line_y = logo_y + logo_h // 2
    half_line = LOGO_LINE_LENGTH // 2
    left_x1 = logo_x - 20 - half_line
    left_x2 = logo_x - 20
    right_x1 = logo_x + logo_w + 20
    right_x2 = right_x1 + half_line
    draw.line((left_x1, line_y, left_x2, line_y), fill=COLOR_WHITE, width=LOGO_LINE_THICKNESS_PX)
    draw.line((right_x1, line_y, right_x2, line_y), fill=COLOR_WHITE, width=LOGO_LINE_THICKNESS_PX)

    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)

    # Заголовок
    cur_y = start_y + logo_h + SPACING_LOGO_TO_TITLE
    for ln in title_lines:
        bb = title_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        _ = draw_text_with_stretch(image, x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += title_line_h + LINE_SPACING

    return image

def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """Режим 2: только заголовок (UPPERCASE). Уважает ручные переносы (\n)."""
    image = image.convert("RGBA")
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)

    title = (title_translated or "").upper()

    _, title_font, title_lines = calculate_adaptive_font_size(
        title, FONT_PATH, max_text_width, FONT_SIZE_MODE2, stretch_width=TEXT_STRETCH_WIDTH
    )

    line_h = compute_fixed_line_height(title_font, TEXT_STRETCH_HEIGHT, extra_pad=(TEXT_SHADOW_OFFSET * 2 + 6))
    total_h = line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)

    start_y = height - SPACING_BOTTOM - total_h

    cur_y = start_y
    for ln in title_lines:
        bb = title_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        _ = draw_text_with_stretch(image, x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += line_h + LINE_SPACING

    return image

def render_mode3_content(image: Image.Image, title_translated: str, subtitle_translated: str) -> Image.Image:
    """Режим 3: заголовок + подзаголовок (оба UPPERCASE). Уважает ручные переносы (\n)."""
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

    title_line_h = compute_fixed_line_height(title_font, TEXT_STRETCH_HEIGHT, extra_pad=(TEXT_SHADOW_OFFSET * 2 + 6))
    subtitle_line_h = compute_fixed_line_height(subtitle_font, TEXT_STRETCH_HEIGHT, extra_pad=(TEXT_SHADOW_OFFSET * 2 + 6))

    title_block_h = title_line_h * len(title_lines) + max(0, (len(title_lines) - 1) * LINE_SPACING)
    subtitle_block_h = subtitle_line_h * len(subtitle_lines) + max(0, (len(subtitle_lines) - 1) * LINE_SPACING)

    total_h = title_block_h + SPACING_TITLE_TO_SUBTITLE + subtitle_block_h
    start_y = height - SPACING_BOTTOM - total_h

    cur_y = start_y
    for ln in title_lines:
        bb = title_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        _ = draw_text_with_stretch(image, x, cur_y, ln, title_font, COLOR_TURQUOISE, COLOR_OUTLINE)
        cur_y += title_line_h + LINE_SPACING

    cur_y += SPACING_TITLE_TO_SUBTITLE

    for ln in subtitle_lines:
        bb = subtitle_font.getbbox(ln)
        ln_w = bb[2] - bb[0]
        x = (width - int(ln_w * TEXT_STRETCH_WIDTH)) // 2
        _ = draw_text_with_stretch(image, x, cur_y, ln, subtitle_font, COLOR_WHITE, COLOR_OUTLINE)
        cur_y += subtitle_line_h + LINE_SPACING

    return image

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
