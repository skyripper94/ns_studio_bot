"""
Complete Workflow (FINAL FIX):
1. OCR (Google Vision API on bottom 35%)
2. Remove EVERYTHING in bottom 35% (FLUX Kontext Pro) - WITH ROI CROP
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

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Colors
COLOR_TURQUOISE = (0, 206, 209)  # #00CED1 (PIL uses RGB)
COLOR_WHITE = (255, 255, 255)
COLOR_OUTLINE = (60, 60, 60)  # #3C3C3C

# Font sizes
FONT_SIZE_MODE1 = 64
FONT_SIZE_MODE2 = 62
FONT_SIZE_MODE3_TITLE = 62
FONT_SIZE_MODE3_SUBTITLE = 58
FONT_SIZE_LOGO = 22
FONT_SIZE_MIN = 36

# Spacing
SPACING_BOTTOM = 120
SPACING_LOGO_TO_TITLE = 2
SPACING_TITLE_TO_SUBTITLE = 10
LINE_SPACING = 8
LOGO_LINE_LENGTH = 300

# Layout
TEXT_WIDTH_PERCENT = 0.9

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
"Northrop B-2 Spirit → Бомбордировщик Northrop B-2 Spirit"
"Mars' Perseverance Rover → Марсоход Perseverance"
"Aircraft → Истребитель"
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
    FLUX Kontext Pro - MAXIMUM STRENGTH text removal
    """
    if not REPLICATE_API_TOKEN:
        logger.error("❌ REPLICATE_API_TOKEN NOT SET! Using fallback...")
        return opencv_fallback(image, mask)
    
    try:
        import replicate
        
        logger.info("🚀 FLUX MAXIMUM - aggressive text removal")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Convert mask
        pil_mask = Image.fromarray(mask)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # MAXIMUM STRENGTH PROMPT
        prompt = """COMPLETELY ERASE and REMOVE ALL: text, letters, numbers, words, symbols, logos, 
        watermarks, lines, decorations, gradients, overlays, ANY foreground elements in masked area.
        Fill ONLY with clean, natural background texture matching surrounding unmasked area.
        CRITICAL: Absolutely NO text or overlays should remain. Pure background restoration only."""
        
        logger.info("📤 Sending to FLUX with MAXIMUM parameters...")
        
        # Initialize client with token
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        
        # Run with MAXIMUM parameters
        output = client.run(
            "black-forest-labs/flux-kontext-pro",
            input={
                "prompt": prompt,
                "input_image": img_buffer,
                "mask": mask_buffer,
                "output_format": "png",
                "go_fast": False,
                "num_inference_steps": 100,  # MAXIMUM steps
                "guidance_scale": 15.0,      # MAXIMUM guidance
                "strength": 1.0,             # MAXIMUM strength
                "prompt_strength": 1.0,      # MAXIMUM prompt adherence
                "seed": 42                   # Fixed seed for consistency
            }
        )
        
        logger.info("⏳ Processing with FLUX...")
        
        # Get result
        if hasattr(output, 'read'):
            result_bytes = output.read()
        elif isinstance(output, str):
            response = requests.get(output, timeout=90)
            result_bytes = response.content
        elif isinstance(output, list) and len(output) > 0:
            response = requests.get(output[0], timeout=90)
            result_bytes = response.content
        else:
            logger.error(f"❌ Unexpected output type: {type(output)}")
            return opencv_fallback(image, mask)
        
        result_pil = Image.open(BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert('RGB'))
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if result_bgr.shape[:2] != image.shape[:2]:
            logger.warning(f"⚠️ Resizing from {result_bgr.shape[:2]} to {image.shape[:2]}")
            result_bgr = cv2.resize(result_bgr, (image.shape[1], image.shape[0]), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        logger.info("✅ FLUX MAXIMUM completed!")
        return result_bgr
        
    except Exception as e:
        logger.error(f"❌ FLUX error: {e}", exc_info=True)
        return opencv_fallback(image, mask)


def flux_kontext_inpaint_bottom_roi(image: np.ndarray, mask_start: int, overlap_px: int = 150) -> np.ndarray:
    """
    Inpaint ONLY bottom ROI to preserve top logos
    Enhanced overlap and blending for seamless result
    """
    h, w = image.shape[:2]
    mask_start = int(np.clip(mask_start, 0, h))
    
    logger.info(f"🔧 Processing bottom ROI with overlap")
    
    # ROI with overlap for context
    roi_start = max(0, mask_start - overlap_px)
    roi = image[roi_start:h, :]
    roi_h = h - roi_start
    
    logger.info(f"📐 ROI: rows {roi_start}-{h} (overlap: {overlap_px}px)")
    
    # Create mask for ROI (mask only actual bottom part)
    mask_roi = np.zeros((roi_h, w), dtype=np.uint8)
    local_mask_start = mask_start - roi_start
    if local_mask_start < roi_h:
        mask_roi[local_mask_start:, :] = 255
    
    # Save ROI for debugging
    cv2.imwrite("/tmp/debug_roi.png", roi)
    cv2.imwrite("/tmp/debug_roi_mask.png", mask_roi)
    
    # Process ROI with FLUX
    roi_clean = flux_kontext_inpaint(roi, mask_roi)
    
    # Enhanced blending
    out = image.copy()
    
    # Create smooth blend in overlap zone
    if overlap_px > 0 and local_mask_start > 0:
        blend_height = min(overlap_px, local_mask_start)
        
        # Cosine interpolation for smoother blend
        alpha = np.zeros((blend_height, 1, 1), dtype=np.float32)
        for i in range(blend_height):
            # Cosine curve for smoother transition
            t = i / float(blend_height)
            alpha[i] = 0.5 * (1 + np.cos(np.pi * (1 - t)))
        
        # Apply blend
        blend_start = mask_start - blend_height
        blend_end = mask_start
        
        orig_part = image[blend_start:blend_end].astype(np.float32)
        new_part = roi_clean[:blend_height].astype(np.float32)
        
        blended = orig_part * (1.0 - alpha) + new_part * alpha
        out[blend_start:blend_end] = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Copy clean bottom part
    out[mask_start:h] = roi_clean[local_mask_start:]
    
    logger.info("✅ ROI processing with blend completed")
    return out


def create_gradient_layer(width: int, height: int, start_percent: int = 55) -> Image.Image:
    """
    Create gradient as a separate RGBA layer
    Transparent at top, black at bottom
    """
    gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    
    start_row = int(height * (1 - start_percent / 100))
    
    # Smoother gradient with more steps
    for y in range(start_row, height):
        progress = (y - start_row) / (height - start_row)
        # Exponential curve for smoother gradient
        alpha = int(255 * (progress ** 0.8))
        
        # Draw horizontal line with alpha
        draw.rectangle([(0, y), (width, y+1)], fill=(0, 0, 0, alpha))
    
    logger.info(f"✨ Created gradient layer from row {start_row} ({start_percent}%)")
    return gradient


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


def draw_text_with_effects(draw: ImageDraw.Draw, x: int, y: int, 
                            text: str, font: ImageFont.FreeTypeFont,
                            fill_color: tuple, outline_color: tuple,
                            shadow_offset: int = 2) -> int:
    """
    Draw text with shadow and outline
    Returns: height of drawn text
    """
    # Shadow
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 128))
    
    # Outline (8 directions)
    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    
    # Main text
    draw.text((x, y), text, font=font, fill=fill_color)
    
    bbox = font.getbbox(text)
    return bbox[3] - bbox[1]


def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Mode 1: Logo + 2 lines + Title (UPPERCASE)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    
    # Calculate title size and position
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE1
    )
    
    # Calculate heights
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        title_heights.append(bbox[3] - bbox[1])
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    
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
        
        draw_text_with_effects(draw, line_x, title_y, line, title_font,
                               COLOR_TURQUOISE, COLOR_OUTLINE)
        
        title_y += title_heights[i] + LINE_SPACING
    
    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Mode 2: Title only (no logo) (UPPERCASE)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    
    # Calculate title
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE2
    )
    
    # Calculate heights
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        title_heights.append(bbox[3] - bbox[1])
    
    total_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    
    # Start position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw title
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        draw_text_with_effects(draw, line_x, current_y, line, title_font,
                               COLOR_TURQUOISE, COLOR_OUTLINE)
        
        current_y += title_heights[i] + LINE_SPACING
    
    return image


def render_mode3_content(image: Image.Image, title_translated: str, 
                         subtitle_translated: str) -> Image.Image:
    """
    Mode 3: Title + Subtitle (BOTH UPPERCASE)
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
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
    
    # Calculate heights
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        title_heights.append(bbox[3] - bbox[1])
    
    subtitle_heights = []
    for line in subtitle_lines:
        bbox = subtitle_font.getbbox(line)
        subtitle_heights.append(bbox[3] - bbox[1])
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    total_subtitle_height = sum(subtitle_heights) + (len(subtitle_lines) - 1) * LINE_SPACING
    
    total_height = total_title_height + SPACING_TITLE_TO_SUBTITLE + total_subtitle_height
    
    # Start position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw title
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        draw_text_with_effects(draw, line_x, current_y, line, title_font,
                               COLOR_TURQUOISE, COLOR_OUTLINE)
        
        current_y += title_heights[i] + LINE_SPACING
    
    # Draw subtitle
    current_y += SPACING_TITLE_TO_SUBTITLE
    
    for i, line in enumerate(subtitle_lines):
        line_bbox = subtitle_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        draw_text_with_effects(draw, line_x, current_y, line, subtitle_font,
                               COLOR_WHITE, COLOR_OUTLINE)
        
        current_y += subtitle_heights[i] + LINE_SPACING
    
    return image


def process_full_workflow(image: np.ndarray, mode: int) -> tuple:
    """
    Full workflow - FINAL VERSION with ROI
    
    Returns: (result_image, ocr_data)
    """
    logger.info("=" * 60)
    logger.info(f"🚀 FULL WORKFLOW FINAL - MODE {mode}")
    logger.info(f"📊 API TOKENS:")
    logger.info(f"   REPLICATE: {'✅' if REPLICATE_API_TOKEN else '❌ MISSING'}")
    logger.info(f"   GOOGLE_VISION: {'✅' if GOOGLE_VISION_API_KEY else '❌ MISSING'}")
    logger.info(f"   OPENAI: {'✅' if OPENAI_API_KEY else '❌ MISSING'}")
    logger.info("=" * 60)
    
    height, width = image.shape[:2]
    
    # STEP 1: OCR
    logger.info("📋 STEP 1: OCR")
    ocr_data = google_vision_ocr(image, crop_bottom_percent=35)
    
    if not ocr_data['text']:
        logger.warning("⚠️ No text detected")
        return image, ocr_data
    
    # STEP 2: Remove text with ROI (preserve top logos)
    logger.info("📋 STEP 2: Remove text (FLUX with ROI)")
    mask_start = int(height * 0.65)
    clean_image = flux_kontext_inpaint_bottom_roi(image, mask_start=mask_start, overlap_px=150)
    
    # STEP 3: Translate
    logger.info("📋 STEP 3: Translate")
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
    
    # STEP 4: Apply gradient
    logger.info("📋 STEP 4: Apply gradient")
    clean_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(clean_rgb).convert('RGBA')
    
    gradient_layer = create_gradient_layer(pil_image.width, pil_image.height, start_percent=55)
    pil_image = Image.alpha_composite(pil_image, gradient_layer)
    
    # STEP 5: Render text
    logger.info(f"📋 STEP 5: Render text (Mode {mode})")
    if mode == 1:
        pil_image = render_mode1_logo(pil_image, title_translated)
    elif mode == 2:
        pil_image = render_mode2_text(pil_image, title_translated)
    elif mode == 3:
        pil_image = render_mode3_content(pil_image, title_translated, subtitle_translated)
    
    # Convert back
    result_rgb = np.array(pil_image.convert('RGB'))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    logger.info("✅ WORKFLOW COMPLETED!")
    return result_bgr, ocr_data


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compatibility function"""
    return flux_kontext_inpaint(image, mask)
