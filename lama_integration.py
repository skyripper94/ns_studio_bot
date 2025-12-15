"""
Complete Workflow:
1. OCR (Google Vision API on bottom 35%)
2. Remove text (FLUX Kontext Pro)
3. Translate & adapt (OpenAI GPT-4)
4. Render gradient + text with custom styling
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
COLOR_SHADOW = (0, 0, 0, 128)  # Semi-transparent black

# Font sizes (ORIGINAL SIZES - before *1.25)
FONT_SIZE_MODE1 = 50  # Original
FONT_SIZE_MODE2 = 48  # Original
FONT_SIZE_MODE3_TITLE = 46  # Original
FONT_SIZE_MODE3_SUBTITLE = 38  # Original
FONT_SIZE_LOGO = 18
FONT_SIZE_MIN = 32

# Spacing
SPACING_BOTTOM = 100
SPACING_LOGO_TO_TITLE = 6
SPACING_TITLE_TO_SUBTITLE = 10
LINE_SPACING = 34
LOGO_LINE_LENGTH = 300

# Layout
TEXT_WIDTH_PERCENT = 0.9

# Font path
FONT_PATH = '/app/fonts/WaffleSoft.otf'


def google_vision_ocr(image: np.ndarray, crop_bottom_percent: int = 35) -> dict:
    """
    OCR using Google Vision API on bottom portion of image
    Returns: dict with 'text' and 'lines'
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
        
        # First annotation is full text
        full_text = response_data['textAnnotations'][0]['description']
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
    Universal stable pipeline:
    - Run FLUX on FULL image (no crop => no crop-edge artifacts).
    - Build mask_flux = base bottom mask + SAFE leak band mask.
    - After FLUX: copy back ONLY masked pixels (prevents any top/logo/global edits).
    - SAFE leak mask uses text-like detector + edge-suppression + CC filtering
      so it does NOT grab objects (prevents mirrored/warped planes etc).
    """

    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN not set, using OpenCV")
        return opencv_fallback(image, mask)

    try:
        import replicate

        h, w = image.shape[:2]

        # ---------- 1) Robust boundary_y from incoming mask ----------
        mask_bin = (mask > 0).astype(np.uint8)
        row_frac = mask_bin.mean(axis=1)

        row_threshold = 0.08
        stable_rows = 12

        boundary_y = None
        for y in range(0, h - stable_rows):
            if row_frac[y] >= row_threshold and np.all(row_frac[y:y + stable_rows] >= row_threshold):
                boundary_y = y
                break

        if boundary_y is None:
            boundary_y = int(h * 0.65)
            logger.warning("⚠️ Could not detect boundary from mask reliably, fallback to 65% height")

        # ---------- 2) Base mask (your bottom area) ----------
        base_mask = (mask > 0).astype(np.uint8) * 255  # should be bottom region already

        # ---------- 3) SAFE leak mask in a narrow band above boundary ----------
        # band size (universal): enough to catch faint "ghost" text, not enough to hit object
        leak_up = max(140, int(0.14 * h))     # 140..~250 typically
        leak_down = 50

        band_y0 = max(0, boundary_y - leak_up)
        band_y1 = min(h, boundary_y + leak_down)

        if band_y1 <= band_y0:
            mask_flux = base_mask
        else:
            band = image[band_y0:band_y1, :].copy()

            # --- text-like candidate map (tophat/blackhat on L channel) ---
            lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
            L, _, _ = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            Lc = clahe.apply(L)

            k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
            tophat = cv2.morphologyEx(Lc, cv2.MORPH_TOPHAT, k)
            blackhat = cv2.morphologyEx(Lc, cv2.MORPH_BLACKHAT, k)
            cand = cv2.max(tophat, blackhat)
            cand = cv2.GaussianBlur(cand, (0, 0), 1.2)

            # threshold for faint watermark
            thr = np.percentile(cand, 88)
            thr = max(thr, 10)
            m = (cand >= thr).astype(np.uint8) * 255

            # --- EDGE SUPPRESSION (key anti-plane/anti-rover) ---
            # We remove strong edges (objects) from leak detection.
            gx = cv2.Sobel(Lc, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(Lc, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)

            edge_hi = np.percentile(mag, 92)  # strong edges (objects)
            keep_low_edges = (mag < edge_hi).astype(np.uint8) * 255

            m = cv2.bitwise_and(m, keep_low_edges)

            # connect broken letters a bit (but not enough to make a full rectangle)
            m = cv2.morphologyEx(
                m, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
            )
            m = cv2.morphologyEx(
                m, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)), iterations=1
            )

            # --- CC FILTER: keep "text-ish" components only ---
            num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
            filtered = np.zeros_like(m)

            band_h = band.shape[0]
            band_area = float(band_h * w)

            min_area = max(10, int(band_area * 0.00002))
            max_area = int(band_area * 0.03)  # prevents huge blobs

            for i in range(1, num):
                x, y, cw, ch, area = stats[i]

                if area < min_area or area > max_area:
                    continue
                if ch < 6 or cw < 18:
                    continue

                ar = cw / float(ch)
                # watermark text lines are usually wider than tall
                if ar < 1.2 or ar > 80.0:
                    continue

                fill = area / float(max(1, cw * ch))
                if fill < 0.01 or fill > 0.85:
                    continue

                filtered[labels == i] = 255

            # expand slightly to cover halos
            filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

            leak_mask = np.zeros((h, w), dtype=np.uint8)
            leak_mask[band_y0:band_y1, :] = filtered

            # ---------- 4) Final mask for FLUX ----------
            mask_flux = cv2.bitwise_or(base_mask, leak_mask)

        # If mask is empty, just return original
        if mask_flux.mean() < 0.1:
            logger.warning("⚠️ mask_flux is empty-ish, returning original")
            return image

        # ---------- 5) Encode FULL image + FULL mask ----------
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        pil_mask = Image.fromarray(mask_flux)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format="PNG")
        mask_buffer.seek(0)

        prompt = (
            "Remove only the text in the masked area. Preserve everything outside the mask. "
            "Naturally restore the background. No global edits."
        )

        logger.info(
            f"📤 FLUX FULLFRAME boundary_y={boundary_y}, band={band_y0}-{band_y1}, "
            f"mask_mean={mask_flux.mean():.2f}"
        )

        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "input_image": img_buffer,
                "mask": mask_buffer,
                "output_format": "png",
                "go_fast": False,
                "num_inference_steps": 28
            }
        )

        # ---------- 6) Read result ----------
        if hasattr(output, "read"):
            result_bytes = output.read()
        elif isinstance(output, str):
            result_bytes = requests.get(output, timeout=60).content
        elif isinstance(output, list) and len(output) > 0:
            result_bytes = requests.get(output[0], timeout=60).content
        else:
            logger.error(f"❌ Unknown output: {type(output)}")
            return opencv_fallback(image, mask)

        result_pil = Image.open(BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert("RGB"))
        flux_result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # ---------- 7) Copy ONLY masked pixels back ----------
        final = image.copy()
        copy_mask = mask_flux > 0
        for c in range(3):
            final[:, :, c][copy_mask] = flux_result[:, :, c][copy_mask]

        logger.info("✅ FLUX done! Copied ONLY masked pixels (universal, no global artifacts).")
        return final

    except Exception as e:
        logger.error(f"❌ FLUX error: {e}")
        return opencv_fallback(image, mask)


def create_gradient(width: int, height: int, start_percent: int = 65) -> np.ndarray:
    """
    Gradient with solid black base at bottom (100px) + smooth fade above
    """
    gradient = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA
    
    start_row = int(height * (1 - start_percent / 100))
    black_base_height = 160  # Черная основа снизу
    
    for y in range(height):
        if y >= height - black_base_height:
            # Solid black at bottom (100px)
            alpha = 255
        elif y >= start_row:
            # Smooth gradient from start to black base
            progress = (y - start_row) / (height - black_base_height - start_row)
            alpha = int(255 * (progress ** 0.7))
        else:
            alpha = 0
        
        gradient[y, :] = [0, 0, 0, alpha]
    
    logger.info(f"✨ Gradient: {start_percent}% start + 100px black base")
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


def draw_sharp_stretched_text(image: Image.Image, x: int, y: int, 
                               text: str, font: ImageFont.FreeTypeFont,
                               fill_color: tuple, outline_color: tuple,
                               shadow_offset: int = 2):
    """Draw super sharp text with 3x rendering + 25% vertical stretch"""
    
    # Get text bounding box WITH offset
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create temp 3x
    scale = 3
    temp = Image.new('RGBA', (text_width * scale, text_height * scale), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    
    # Font 3x
    font_3x = ImageFont.truetype(font.path, font.size * scale)
    
    # Get bbox for 3x font and calculate offset
    bbox_3x = font_3x.getbbox(text)
    offset_x = -bbox_3x[0]  # Компенсация смещения
    offset_y = -bbox_3x[1]  # Компенсация смещения
    
    # Shadow
    temp_draw.text((offset_x + shadow_offset * scale, offset_y + shadow_offset * scale), 
                   text, font=font_3x, fill=(0, 0, 0, 128))
    
    # Outline
    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        temp_draw.text((offset_x + dx * scale, offset_y + dy * scale), 
                       text, font=font_3x, fill=outline_color)
    
    # Main text
    temp_draw.text((offset_x, offset_y), text, font=font_3x, fill=fill_color)
    
    # Downscale
    temp = temp.resize((text_width, text_height), Image.LANCZOS)
    
    # Stretch +25%
    stretched_height = int(text_height * 2.0)
    temp_stretched = temp.resize((text_width, stretched_height), Image.LANCZOS)
    
    # Paste
    image.paste(temp_stretched, (x, y), temp_stretched)


def render_mode1_logo(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Mode 1: Logo + 2 lines + Title (UPPERCASE)
    """
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    
    # Calculate title size and position
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE1
    )
    
    # Calculate total height needed (with 25% stretch)
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        stretched_height = int(line_height * 1.25)  # +25% vertical
        title_heights.append(stretched_height)
    
    total_title_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    
    # Logo
    logo_font = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOGO)
    logo_text = "@neurostep.media"
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_height = logo_bbox[3] - logo_bbox[1]
    
    # Total construction height
    total_height = logo_height + SPACING_LOGO_TO_TITLE + total_title_height
    
    # Start Y position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw logo
    draw = ImageDraw.Draw(image, 'RGBA')
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
    
    # Draw title with sharp + stretched rendering
    title_y = start_y + logo_height + SPACING_LOGO_TO_TITLE
    
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        draw_sharp_stretched_text(
            image, line_x, title_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE, shadow_offset=2
        )
        
        title_y += title_heights[i] + LINE_SPACING
    
    return image


def render_mode2_text(image: Image.Image, title_translated: str) -> Image.Image:
    """
    Mode 2: Title only (no logo) (UPPERCASE)
    """
    width, height = image.size
    max_text_width = int(width * TEXT_WIDTH_PERCENT)
    
    # Convert to UPPERCASE
    title_translated = title_translated.upper()
    
    # Calculate title
    title_size, title_font, title_lines = calculate_adaptive_font_size(
        title_translated, FONT_PATH, max_text_width, FONT_SIZE_MODE2
    )
    
    # Total height (with 25% stretch)
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        stretched_height = int(line_height * 1.25)
        title_heights.append(stretched_height)
    
    total_height = sum(title_heights) + (len(title_lines) - 1) * LINE_SPACING
    
    # Start position
    start_y = height - SPACING_BOTTOM - total_height
    
    # Draw title
    current_y = start_y
    for i, line in enumerate(title_lines):
        line_bbox = title_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        draw_sharp_stretched_text(
            image, line_x, current_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE, shadow_offset=2
        )
        
        current_y += title_heights[i] + LINE_SPACING
    
    return image


def render_mode3_content(image: Image.Image, title_translated: str, 
                         subtitle_translated: str) -> Image.Image:
    """
    Mode 3: Title + Subtitle (BOTH UPPERCASE)
    """
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
    
    # Total height (with 25% stretch)
    title_heights = []
    for line in title_lines:
        bbox = title_font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        stretched_height = int(line_height * 1.25)
        title_heights.append(stretched_height)
    
    subtitle_heights = []
    for line in subtitle_lines:
        bbox = subtitle_font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        stretched_height = int(line_height * 1.25)
        subtitle_heights.append(stretched_height)
    
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
        
        draw_sharp_stretched_text(
            image, line_x, current_y, line, title_font,
            COLOR_TURQUOISE, COLOR_OUTLINE, shadow_offset=2
        )
        
        current_y += title_heights[i] + LINE_SPACING
    
    # Draw subtitle
    current_y += SPACING_TITLE_TO_SUBTITLE
    
    for i, line in enumerate(subtitle_lines):
        line_bbox = subtitle_font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        draw_sharp_stretched_text(
            image, line_x, current_y, line, subtitle_font,
            COLOR_WHITE, COLOR_OUTLINE, shadow_offset=2
        )
        
        current_y += subtitle_heights[i] + LINE_SPACING
    
    return image


def process_full_workflow(image: np.ndarray, mode: int) -> tuple:
    """
    Full workflow for modes 1, 2, 3
    Returns: (result_image, ocr_data)
    """
    logger.info("=" * 60)
    logger.info(f"🚀 FULL WORKFLOW - MODE {mode}")
    logger.info("=" * 60)
    
    # Step 1: OCR on bottom 35%
    logger.info("📋 STEP 1: OCR (Google Vision)")
    ocr_data = google_vision_ocr(image, crop_bottom_percent=35)
    
    if not ocr_data['text']:
        logger.warning("⚠️ No text detected")
        return image, ocr_data
    
    # Step 2: Create mask for bottom 35%
    logger.info("📋 STEP 2: Create Mask (Bottom 35%)")
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_start = int(height * 0.65)
    mask[mask_start:, :] = 255
    logger.info(f"📐 Mask: rows {mask_start}-{height} (35% bottom)")
    
    # Step 3: Remove text with FLUX
    logger.info("📋 STEP 3: Remove Text (FLUX Kontext Pro)")
    clean_image = flux_kontext_inpaint(image, mask)
    
    # Step 4: Translate
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
    
    # Step 5: Create gradient
    logger.info("📋 STEP 5: Create Gradient")
    
    # Convert FLUX result to PIL
    clean_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(clean_rgb).convert('RGBA')
    
    # Get actual image size
    actual_width, actual_height = pil_image.size
    logger.info(f"📐 Image size: {actual_width}x{actual_height}")
    
    # Create smooth gradient (55% coverage, NO black bar)
    gradient = create_gradient(actual_width, actual_height, start_percent=55)
    gradient_image = Image.fromarray(gradient, 'RGBA')
    
    # Apply gradient
    pil_image = Image.alpha_composite(pil_image, gradient_image)
    
    # Step 6: Render text
    logger.info(f"📋 STEP 6: Render Text (Mode {mode})")
    
    if mode == 1:
        pil_image = render_mode1_logo(pil_image, title_translated)
    elif mode == 2:
        pil_image = render_mode2_text(pil_image, title_translated)
    elif mode == 3:
        pil_image = render_mode3_content(pil_image, title_translated, subtitle_translated)
    
    # Convert back
    result_rgb = np.array(pil_image.convert('RGB'))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    logger.info("=" * 60)
    logger.info("✅ WORKFLOW COMPLETED!")
    logger.info("=" * 60)
    
    return result_bgr, ocr_data


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compatibility function"""
    return flux_kontext_inpaint(image, mask)
