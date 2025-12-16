"""
Complete Workflow (SIMPLIFIED):
1. OCR (Google Vision API on bottom 35%) - для получения текста
2. Remove EVERYTHING in bottom 35% (FLUX Kontext Pro) - маска всегда 35%
3. Translate & adapt (OpenAI GPT-4)
4. Apply gradient LAYER on top
5. Render text on top of gradient
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

# Configuration
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')
GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
REPLICATE_MODEL = 'black-forest-labs/flux-kontext-pro'

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Colors
COLOR_TURQUOISE = (0, 206, 209)  # #00CED1 (PIL uses RGB)
COLOR_WHITE = (255, 255, 255)
COLOR_OUTLINE = (60, 60, 60)  # #3C3C3C

# Font sizes - МЕНЯЙ ЭТИ ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ
FONT_SIZE_MODE1 = 52
FONT_SIZE_MODE2 = 46
FONT_SIZE_MODE3_TITLE = 44
FONT_SIZE_MODE3_SUBTITLE = 40
FONT_SIZE_LOGO = 22
FONT_SIZE_MIN = 36

# Spacing - МЕНЯЙ ЭТИ ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ
SPACING_BOTTOM = 140          # Отступ снизу
SPACING_LOGO_TO_TITLE = 1   # Расстояние от лого до заголовка
SPACING_TITLE_TO_SUBTITLE = 10  # Расстояние между заголовком и подзаголовком
LINE_SPACING = 2             # Расстояние между строками текста
LOGO_LINE_LENGTH = 300        # Длина линий возле лого

# Layout
TEXT_WIDTH_PERCENT = 0.9

# Text stretch settings
TEXT_STRETCH_MULTIPLIER = 1.9  # Вертикальное растягивание текста (1.9 = 90% растяжения)
TEXT_PADDING_PERCENT = 0.3     # Padding для хвостиков букв (30% от размера шрифта)
def effective_line_spacing() -> int:
    """
    LINE_SPACING is defined in "base" pixels, but we stretch text vertically
    (TEXT_STRETCH_MULTIPLIER). If we don't scale the gap too, small LINE_SPACING
    changes look like they "do nothing".
    """
    return max(0, int(LINE_SPACING * TEXT_STRETCH_MULTIPLIER))

# Font path
FONT_PATH = '/app/fonts/WaffleSoft.otf'


def google_vision_ocr(image: np.ndarray, crop_bottom_percent: int = 35) -> dict:
    """
    OCR using Google Vision API on bottom portion of image
    Returns: dict with 'text', 'lines'
    """
    if not GOOGLE_VISION_API_KEY:
        logger.warning("⚠️ GOOGLE_VISION_API_KEY not set")
        return {'text': '', 'lines': []}
    
    try:
        # Crop bottom portion
        height, width = image.shape[:2]
        crop_start = int(height * (1 - crop_bottom_percent / 100))
        cropped = image[crop_start:, :]
        
        logger.info(f"🔍 OCR on bottom {crop_bottom_percent}% (rows {crop_start}-{height})")
        
        # Convert to RGB and encode to base64
        image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Call Google Vision API
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
            logger.warning("⚠️ No OCR results")
            return {'text': '', 'lines': []}
        
        response_data = result['responses'][0]
        
        if 'textAnnotations' not in response_data:
            logger.warning("⚠️ No text detected")
            return {'text': '', 'lines': []}
        
        annotations = response_data['textAnnotations']
        
        # First annotation is full text
        full_text = annotations[0]['description']
        logger.info(f"📝 Detected text: {full_text}")
        
        # Extract lines
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        return {
            'text': full_text,
            'lines': lines
        }
        
    except Exception as e:
        logger.error(f"❌ Google Vision OCR error: {e}")
        return {'text': '', 'lines': []}


def openai_translate(text: str, context: str = "") -> str:
    """
    Translate and adapt text using OpenAI GPT-4
    """
    if not OPENAI_API_KEY or not text:
        logger.warning("⚠️ OPENAI_API_KEY not set or no text")
        return text
    
    try:
        logger.info(f"🌐 Translating: {text}")
        
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
        logger.info(f"✅ Translated: {translated}")
        
        return translated
        
    except Exception as e:
        logger.error(f"❌ OpenAI translation error: {e}")
        return text


def opencv_fallback(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV fallback inpainting"""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    logger.info("✅ OpenCV fallback inpainting")
    return result


def flux_kontext_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    FLUX Kontext Pro - remove everything in masked area
    """
    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN not set, using OpenCV")
        return opencv_fallback(image, mask)
    
    try:
        import replicate
        
        logger.info("🚀 FLUX - removing content in masked area")
        
        # Convert full image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Use mask
        pil_mask = Image.fromarray(mask)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        prompt = "Remove all text, lines, logos and restore natural background seamlessly"
        
        logger.info("📤 Sending to FLUX...")
        
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "input_image": img_buffer,
                "mask": mask_buffer,
                "output_format": "png",
                "go_fast": False,
                "num_inference_steps": 50
            }
        )
        
        # Get result
        if hasattr(output, 'read'):
            result_bytes = output.read()
        elif isinstance(output, str):
            response = requests.get(output, timeout=60)
            result_bytes = response.content
        elif isinstance(output, list) and len(output) > 0:
            response = requests.get(output[0], timeout=60)
            result_bytes = response.content
        else:
            logger.error(f"❌ Unknown output: {type(output)}")
            return opencv_fallback(image, mask)
        
        result_pil = Image.open(BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert('RGB'))
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # Resize back to original if needed
        if result_bgr.shape[:2] != image.shape[:2]:
            logger.warning(f"⚠️ FLUX changed size, resizing back")
            result_bgr = cv2.resize(result_bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
        logger.info("✅ FLUX done!")
        return result_bgr
        
    except Exception as e:
        logger.error(f"❌ FLUX error: {e}")
        return opencv_fallback(image, mask)


def create_gradient_layer(width: int, height: int, gradient_start_percent: int = 30,
                          lift_black: int = 40) -> Image.Image:
    """
    Classic vertical gradient:
    - bottom: solid black
    - going up: smoothly fades to transparent
    - top: fully transparent

    Args:
        width: Image width
        height: Image height
        gradient_start_percent: where the fade starts from the TOP (e.g. 55 = start fade at 55% height)
        lift_black: how many pixels to lift the fully-black bottom block upward
                   (keeps a small "solid" base for maximum readability)
    """
    # Where the fade starts (above this it's fully transparent)
    start_y = int(height * (gradient_start_percent / 100))

    # Where solid black begins
    solid_black_start = max(0, min(height, height - lift_black))

    # Degenerate cases (avoid division by zero)
    if solid_black_start <= start_y:
        # Just put solid black at the very bottom part
        alpha_1d = np.zeros((height,), dtype=np.uint8)
        alpha_1d[solid_black_start:] = 255
    else:
        y = np.arange(height, dtype=np.float32)

        # t: 0 at start_y, 1 at solid_black_start
        t = (y - float(start_y)) / float(solid_black_start - start_y)
        t = np.clip(t, 0.0, 1.0)

        # Soft fade: slow start at top, faster near bottom
        # (gamma > 1 makes the upper part more "gentle")
        gamma = 1.9
        t = t ** gamma

        alpha_1d = (255.0 * t).astype(np.uint8)
        alpha_1d[:start_y] = 0
        alpha_1d[solid_black_start:] = 255

    # Expand to 2D alpha mask and build RGBA image
    alpha = np.repeat(alpha_1d[:, None], width, axis=1)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 3] = alpha

    logger.info(
        f"✨ Creating CLASSIC vertical gradient: start_y={start_y}, solid_black_start={solid_black_start}, "
        f"lift_black={lift_black}, start%={gradient_start_percent}"
    )
    return Image.fromarray(rgba, 'RGBA')
def calculate_adaptive_font_size(text: str, font_path: str, max_width: int, 
                                  initial_size: int, min_size: int = 20) -> tuple:
    """
    Calculate font size that fits text within max_width
    Returns: (font_size, font_object, lines)
    """
    font_size = initial_size
    
    while font_size >= min_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            # Split into lines and check width
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
            
            # Check if all lines fit
            fits = all(
                font.getbbox(line)[2] - font.getbbox(line)[0] <= max_width
                for line in lines
            )
            
            if fits:
                return font_size, font, lines
            
        except Exception as e:
            logger.error(f"Font error at size {font_size}: {e}")
        
        font_size -= 2
    
    # Last resort
    font = ImageFont.truetype(font_path, min_size)
    return min_size, font, [text]


def calculate_stretched_height(font: ImageFont.FreeTypeFont, text: str) -> int:
    """
    Calculate final stretched height for a line of text
    This MUST match the actual rendering in draw_sharp_stretched_text
    
    Returns: stretched height in pixels
    """
    bbox = font.getbbox(text)
    base_height = bbox[3] - bbox[1]
    
    # Add padding (same as in draw_sharp_stretched_text)
    padding = int(font.size * TEXT_PADDING_PERCENT)
    height_with_padding = base_height + padding * 2
    
    # Apply stretch (same as in draw_sharp_stretched_text)
    stretched_height = int(height_with_padding * TEXT_STRETCH_MULTIPLIER)
    
    return stretched_height


def draw_sharp_stretched_text(image: Image.Image, x: int, y: int, 
                               text: str, font: ImageFont.FreeTypeFont,
                               fill_color: tuple, outline_color: tuple,
                               shadow_offset: int = 2) -> int:
    """
    Draw super sharp text with 3x rendering + vertical stretch x2
    
    Returns: height of drawn text (stretched)
    """
    # Get text size
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # ADD PADDING for descenders (хвостики букв)
    padding = int(font.size * TEXT_PADDING_PERCENT)
    text_height_with_padding = text_height + padding * 2
    
    # Create temporary image 3x for sharpness
    scale = 3
    temp_width = text_width * scale
    temp_height = text_height_with_padding * scale
    
    temp = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
                                   
    # Draw with padding offset
    y_offset = padding * scale
    
    # Font 3x
    font_3x = ImageFont.truetype(font.path, font.size * scale)
    
    # Draw with 3x resolution
    # Shadow
    temp_draw.text((shadow_offset * scale, y_offset + shadow_offset * scale), text, 
                   font=font_3x, fill=(0, 0, 0, 128))
    
    # Outline (8 directions)
    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        temp_draw.text((dx * scale, y_offset + dy * scale), text, 
                       font=font_3x, fill=outline_color)
    
    # Main text
    temp_draw.text((0, y_offset), text, font=font_3x, fill=fill_color)
        
    # Downscale to original size WITH PADDING (for sharpness)
    temp = temp.resize((text_width, text_height_with_padding), Image.LANCZOS)

    # ⭐ STRETCH VERTICALLY x2
    stretched_height = int(text_height_with_padding * TEXT_STRETCH_MULTIPLIER)
    temp_stretched = temp.resize((text_width, stretched_height), Image.LANCZOS)
    
    # Paste stretched text into image
    image.paste(temp_stretched, (x, y), temp_stretched)
    
    return stretched_height


def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Mode 1: Logo + 2 lines + Title (UPPERCASE)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    line_gap = effective_line_spacing()
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    
    # Calculate title size and position
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE1
    )
    
    # Calculate heights WITH proper padding and stretch
    title_heights = []
    for line in title_lines:
        stretched = calculate_stretched_height(title_font, line)
        title_heights.append(stretched)
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * line_gap
    
    # Logo
    logo_font = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOGO)
    logo_text = "@neurostep.media"
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_height = logo_bbox[3] - logo_bbox[1]
    
    # Total height
    total_height = logo_height + SPACING_LOGO_TO_TITLE + total_title_height
    
    # Start Y position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw logo
    logo_x = (width - logo_width) // 2
    logo_y = start_y
    
    # Logo lines
    line_y = logo_y + logo_height // 2
    line_left_start = logo_x - LOGO_LINE_LENGTH - 10
    line_right_start = logo_x + logo_width + 10
    
    draw.line([(line_left_start, line_y), (line_left_start + LOGO_LINE_LENGTH, line_y)],
              fill=COLOR_TURQUOISE, width=1)
    draw.line([(line_right_start, line_y), (line_right_start + LOGO_LINE_LENGTH, line_y)],
              fill=COLOR_TURQUOISE, width=1)
    
    # Logo text
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=COLOR_WHITE)
    
    # Draw title
    title_y = start_y + logo_height + SPACING_LOGO_TO_TITLE
    
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        actual_height = draw_sharp_stretched_text(
            image, line_x, title_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE, shadow_offset=2
        )
        if i < len(title_lines) - 1:
            title_y += actual_height + line_gap
        else:
            title_y += actual_height

    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Mode 2: Title only (no logo) (UPPERCASE)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    line_gap = effective_line_spacing()
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    
    # Calculate title
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE2
    )
    
    # Calculate heights WITH proper padding and stretch
    title_heights = []
    for line in title_lines:
        stretched = calculate_stretched_height(title_font, line)
        title_heights.append(stretched)
    
    total_height = sum(title_heights) + (len(title_lines) - 1) * line_gap
    
    # Start position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw title
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        actual_height = draw_sharp_stretched_text(
            image, line_x, current_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE, shadow_offset=2
        )
        if i < len(title_lines) - 1:
            current_y += actual_height + line_gap
        else:
            current_y += actual_height

    return image


def render_mode3_content(image: Image.Image, title_translated: str, 
                         subtitle_translated: str) -> Image.Image:
    """
    Mode 3: Title + Subtitle (BOTH UPPERCASE)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    line_gap = effective_line_spacing()
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    subtitle_translated = subtitle_translated.upper()
    
    # Calculate title
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE3_TITLE
    )
    
    # Calculate subtitle
    subtitle_initial_size = int(title_size * 0.8)
    subtitle_size, subtitle_font, subtitle_lines = calculate_adaptive_font_size(
        subtitle_translated, FONT_PATH, max_text_width, subtitle_initial_size
    )
    
    # Calculate heights WITH proper padding and stretch
    title_heights = []
    for line in title_lines:
        stretched = calculate_stretched_height(title_font, line)
        title_heights.append(stretched)
    
    subtitle_heights = []
    for line in subtitle_lines:
        stretched = calculate_stretched_height(subtitle_font, line)
        subtitle_heights.append(stretched)
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * line_gap
    total_subtitle_height = sum(subtitle_heights) + (len(subtitle_lines) - 1) * line_gap
    
    total_height = total_title_height + SPACING_TITLE_TO_SUBTITLE + total_subtitle_height
    
    # Start position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw title
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        actual_height = draw_sharp_stretched_text(
            image, line_x, current_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE, shadow_offset=2
        )
        if i < len(title_lines) - 1:
            current_y += actual_height + line_gap
        else:
            current_y += actual_height

    # Draw subtitle
    current_y += SPACING_TITLE_TO_SUBTITLE
    
    for i, line in enumerate(subtitle_lines):
        line_bbox = subtitle_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        actual_height = draw_sharp_stretched_text(
            image, line_x, current_y, line, subtitle_font,
            COLOR_WHITE, COLOR_OUTLINE, shadow_offset=2
        )
        if i < len(subtitle_lines) - 1:
            current_y += actual_height + line_gap
        else:
            current_y += actual_height

    return image


def process_full_workflow(image: np.ndarray, mode: int) -> tuple:
    """
    Full workflow for modes 1, 2, 3
    
    LOGIC:
    1. OCR → get text for translation
    2. FLUX removes EVERYTHING in bottom 35% (text, lines, logo, gradient)
    3. Translate text
    4. Apply CLASSIC vertical gradient LAYER on top of clean image (separate layer)
    5. Render text on top of gradient
    
    Returns: (result_image, ocr_data)
    """
    logger.info("=" * 60)
    logger.info(f"🚀 FULL WORKFLOW - MODE {mode}")
    logger.info("=" * 60)
    
    height, width = image.shape[:2]
    
    # ========================================
    # STEP 1: OCR (just to get text)
    # ========================================
    logger.info("📋 STEP 1: OCR (Google Vision)")
    ocr_data = google_vision_ocr(image, crop_bottom_percent=35)
    
    if not ocr_data['text']:
        logger.warning("⚠️ No text detected")
        return image, ocr_data
    
    # ========================================
    # STEP 2: Create SIMPLE mask = bottom 35%
    # This removes EVERYTHING: text, lines, logo, gradient
    # ========================================
    logger.info("📋 STEP 2: Create mask (bottom 35%)")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_start = int(height * 0.65)  # 35% from bottom
    mask[mask_start:, :] = 255
    
    logger.info(f"📐 Mask: rows {mask_start}-{height} (bottom 35%)")
    
    # ========================================
    # STEP 3: FLUX removes everything in mask
    # ========================================
    logger.info("📋 STEP 3: Remove content (FLUX Kontext Pro)")
    clean_image = flux_kontext_inpaint(image, mask)
    
    # ========================================
    # STEP 4: Translate
    # ========================================
    logger.info("📋 STEP 4: Translate (OpenAI)")
    
    if mode == 3:
        lines = ocr_data['lines']
        if len(lines) >= 2:
            title = ' '.join(lines[:-1])
            subtitle = lines[-1]
        else:
            title = ocr_data['text']
            subtitle = ""
        
        title_translated = openai_translate(title)
        subtitle_translated = openai_translate(subtitle) if subtitle else ""
    else:
        title_translated = openai_translate(ocr_data['text'])
        subtitle_translated = ""
    
    # ========================================
    # STEP 5: Convert to PIL and apply CLASSIC vertical gradient LAYER
    # ========================================
    logger.info("📋 STEP 5: Apply CLASSIC vertical gradient LAYER")
    
    clean_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(clean_rgb).convert('RGBA')
    
    actual_width, actual_height = pil_image.size
    logger.info(f"📐 Image size: {actual_width}x{actual_height}")
    
    # Create CLASSIC vertical gradient as separate layer
    # gradient_start_percent: где начинается градиент в ЦЕНТРЕ (выше = раньше)
    # lift_black: насколько поднять черное дно снизу
    gradient_layer = create_gradient_layer(
        actual_width, actual_height, 
        gradient_start_percent=75,  # В центре градиент начинается на 75% от верха
        lift_black=120  # Черное дно поднято на 120px
    )
    
    # IMPORTANT LAYER ORDER:
    #   1) clean background (RGBA)
    #   2) gradient layer (RGBA)
    #   3) text (last/top layer)
    #
    # Therefore: we composite the gradient BEFORE any text rendering.
    pil_image = Image.alpha_composite(pil_image, gradient_layer)
    
    logger.info("✅ CLASSIC vertical gradient layer applied")
    
    # ========================================
    # STEP 6: Render text ON TOP of gradient
    # ========================================
    logger.info(f"📋 STEP 6: Render Text (Mode {mode})")
    
    if mode == 1:
        pil_image = render_mode1_logo(pil_image, title_translated)
    elif mode == 2:
        pil_image = render_mode2_text(pil_image, title_translated)
    elif mode == 3:
        pil_image = render_mode3_content(pil_image, title_translated, subtitle_translated)
    
    # Convert back to BGR
    result_rgb = np.array(pil_image.convert('RGB'))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    logger.info("=" * 60)
    logger.info("✅ WORKFLOW COMPLETED!")
    logger.info("=" * 60)
    
    return result_bgr, ocr_data


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compatibility function"""
    return flux_kontext_inpaint(image, mask)
