"""
FULL WORKFLOW:
1. OCR - recognize text from image
2. Remove text using FLUX Kontext Pro
3. Translate EN -> RU
4. Add translated text back
"""

import os
import logging
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import easyocr
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# Configuration
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')
REPLICATE_MODEL = 'black-forest-labs/flux-kontext-pro'

# Initialize OCR reader (once)
_ocr_reader = None

def get_ocr_reader():
    """Initialize EasyOCR reader (lazy loading)"""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("🔍 Initializing EasyOCR...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
        logger.info("✅ EasyOCR ready")
    return _ocr_reader


def recognize_text(image: np.ndarray) -> list:
    """
    Recognize text from image using OCR
    Returns: list of dicts with text and coordinates
    """
    try:
        reader = get_ocr_reader()
        
        # Convert BGR to RGB for OCR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        logger.info("🔍 Running OCR...")
        results = reader.readtext(image_rgb)
        
        # Parse results
        text_data = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Filter low confidence
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                text_data.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'x_min': int(min(x_coords)),
                    'y_min': int(min(y_coords)),
                    'x_max': int(max(x_coords)),
                    'y_max': int(max(y_coords))
                })
                
                logger.info(f"📝 Found: '{text}' (confidence: {confidence:.2f})")
        
        return text_data
        
    except Exception as e:
        logger.error(f"❌ OCR error: {e}")
        return []


def create_text_mask(image: np.ndarray, text_data: list, gradient_height_percent: int = 40) -> np.ndarray:
    """
    Create mask for text removal
    - Covers detected text boxes
    - Extends to cover gradient area (bottom 35-40%)
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not text_data:
        # No text detected - mask bottom gradient area
        gradient_start = int(height * (1 - gradient_height_percent / 100))
        mask[gradient_start:, :] = 255
        logger.info(f"📐 No text detected, masking bottom {gradient_height_percent}%")
        return mask
    
    # Find text area boundaries
    min_y = min(item['y_min'] for item in text_data)
    max_y = max(item['y_max'] for item in text_data)
    
    # Extend mask to cover gradient area (bottom 35-40%)
    gradient_start = int(height * (1 - gradient_height_percent / 100))
    mask_start = min(min_y, gradient_start)
    mask_end = height  # Always to bottom
    
    # Fill mask area
    mask[mask_start:mask_end, :] = 255
    
    logger.info(f"📐 Text mask: rows {mask_start}-{mask_end} (gradient area: bottom {gradient_height_percent}%)")
    
    return mask


def translate_text(text: str, source: str = 'en', target: str = 'ru') -> str:
    """Translate text using Google Translate"""
    try:
        translator = GoogleTranslator(source=source, target=target)
        translated = translator.translate(text)
        logger.info(f"🌐 Translated: '{text}' → '{translated}'")
        return translated
    except Exception as e:
        logger.error(f"❌ Translation error: {e}")
        return text  # Return original if translation fails


def opencv_fallback(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV fallback inpainting"""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    logger.info("✅ OpenCV fallback inpainting done")
    return result


def flux_kontext_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove text using FLUX Kontext Pro
    """
    if not REPLICATE_API_TOKEN:
        logger.warning("⚠️ REPLICATE_API_TOKEN not set, using OpenCV fallback")
        return opencv_fallback(image, mask)
    
    try:
        import replicate
        
        logger.info(f"🚀 Starting FLUX Kontext Pro...")
        
        # Convert image to BytesIO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Convert mask to BytesIO
        pil_mask = Image.fromarray(mask)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # FLUX Kontext Pro prompt for inpainting
        prompt = "seamless clean background, smooth gradient, no text, no logos, professional photo editing, high quality"
        
        logger.info(f"📤 Sending to FLUX Kontext Pro...")
        logger.info(f"🎨 Prompt: {prompt}")
        
        # Run FLUX Kontext Pro
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "input_image": img_buffer,
                "mask": mask_buffer,
                "output_format": "png"
            }
        )
        
        # Get result
        if hasattr(output, 'read'):
            result_bytes = output.read()
        elif isinstance(output, str):
            import requests
            logger.info("📥 Downloading result...")
            response = requests.get(output, timeout=60)
            result_bytes = response.content
        elif isinstance(output, list) and len(output) > 0:
            import requests
            logger.info("📥 Downloading result...")
            response = requests.get(output[0], timeout=60)
            result_bytes = response.content
        else:
            logger.error(f"❌ Unknown output format: {type(output)}")
            return opencv_fallback(image, mask)
        
        # Convert to numpy array
        result_pil = Image.open(BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert('RGB'))
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        logger.info("✅ FLUX Kontext Pro inpainting done!")
        return result_bgr
        
    except ImportError:
        logger.error("❌ replicate library not installed!")
        return opencv_fallback(image, mask)
    
    except Exception as e:
        logger.error(f"❌ FLUX Kontext Pro error: {e}")
        logger.info("Using OpenCV fallback")
        return opencv_fallback(image, mask)


def add_text_to_image(image: np.ndarray, texts: list, font_size: int = 40) -> np.ndarray:
    """
    Add translated text back to image
    texts: list of dicts with 'text' and position info
    """
    try:
        # Convert to PIL for text rendering
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
            logger.warning("⚠️ Using default font")
        
        height, width = image.shape[:2]
        
        for item in texts:
            text = item['translated_text']
            
            # Calculate text position (bottom center)
            # Get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position at bottom center
            x = (width - text_width) // 2
            y = height - text_height - 30  # 30px from bottom
            
            # Draw text with shadow for better visibility
            # Shadow
            draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 255))
            # Main text (white)
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
            
            logger.info(f"✍️ Added text: '{text}' at ({x}, {y})")
        
        # Convert back to OpenCV
        result_rgb = np.array(pil_image)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        return result_bgr
        
    except Exception as e:
        logger.error(f"❌ Text rendering error: {e}")
        return image


def process_image_full_workflow(image: np.ndarray, gradient_percent: int = 40, translate: bool = True) -> tuple:
    """
    FULL WORKFLOW:
    1. Recognize text (OCR)
    2. Create mask for gradient area
    3. Remove text using FLUX Kontext Pro
    4. Translate text EN->RU
    5. Add translated text back
    
    Returns: (processed_image, recognized_texts)
    """
    logger.info("=" * 50)
    logger.info("🚀 Starting FULL WORKFLOW")
    logger.info("=" * 50)
    
    # Step 1: OCR
    logger.info("📋 STEP 1: Text Recognition")
    text_data = recognize_text(image)
    
    if not text_data:
        logger.warning("⚠️ No text detected, will mask gradient area only")
    
    # Step 2: Create mask
    logger.info("📋 STEP 2: Mask Creation")
    mask = create_text_mask(image, text_data, gradient_percent)
    
    # Step 3: Remove text
    logger.info("📋 STEP 3: Text Removal (FLUX Kontext Pro)")
    result = flux_kontext_inpaint(image, mask)
    
    # Step 4 & 5: Translate and add text
    if text_data and translate:
        logger.info("📋 STEP 4: Translation")
        for item in text_data:
            item['translated_text'] = translate_text(item['text'], 'en', 'ru')
        
        logger.info("📋 STEP 5: Add Translated Text")
        result = add_text_to_image(result, text_data)
    
    logger.info("=" * 50)
    logger.info("✅ WORKFLOW COMPLETED!")
    logger.info("=" * 50)
    
    return result, text_data


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compatibility function for existing telegram_bot.py
    Calls FLUX Kontext Pro
    """
    return flux_kontext_inpaint(image, mask)


def remove_text(image_path: str, mask_path: str, output_path: str) -> bool:
    """
    Compatibility function for CLI usage
    """
    try:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            logger.error("❌ Failed to load files")
            return False
        
        result = flux_kontext_inpaint(image, mask)
        cv2.imwrite(output_path, result)
        
        logger.info(f"✅ Result saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
