"""
Интеграция с Replicate API используя ОФИЦИАЛЬНЫЙ SDK
Намного проще и надежнее чем HTTP API!
"""

import os
import logging
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Конфигурация
REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY', '')
REPLICATE_MODEL = os.getenv('REPLICATE_MODEL', 'ideogram-ai/ideogram-v2')


def opencv_fallback(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV fallback если API не работает"""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Двойной проход для лучшего качества
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Сглаживание краёв
    kernel = np.ones((3,3), np.uint8)
    mask_blurred = cv2.GaussianBlur(mask, (5,5), 0)
    edges = cv2.dilate(mask, kernel, iterations=2) - mask
    edges_blurred = cv2.GaussianBlur(edges.astype(np.float32), (7,7), 0)
    
    for i in range(3):
        result[:,:,i] = np.where(
            edges_blurred > 0,
            cv2.GaussianBlur(result[:,:,i], (5,5), 0),
            result[:,:,i]
        )
    
    logger.info("✅ OpenCV fallback inpainting выполнен")
    return result


def replicate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Удаляет области по маске используя Replicate SDK
    """
    if not REPLICATE_API_KEY:
        logger.warning("⚠️ REPLICATE_API_KEY не установлен, используем OpenCV fallback")
        return opencv_fallback(image, mask)
    
    try:
        import replicate
        
        logger.info(f"🚀 Запуск Replicate SDK (модель: {REPLICATE_MODEL})...")
        
        # Конвертируем изображение в BytesIO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Конвертируем маску в BytesIO
        pil_mask = Image.fromarray(mask)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # Запускаем модель через SDK
        logger.info("📤 Отправка через Replicate SDK...")
        
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "prompt": "clean background, no text, no logos, seamless fill",
                "image": img_buffer,
                "mask": mask_buffer,
                "magic_prompt_option": "Off"  # Для ideogram моделей
            }
        )
        
        # Получаем результат
        # output может быть URL или FileOutput объект
        if hasattr(output, 'read'):
            # Это FileOutput объект
            result_bytes = output.read()
        elif isinstance(output, str):
            # Это URL, загружаем
            import requests
            logger.info("📥 Загрузка результата...")
            response = requests.get(output, timeout=30)
            result_bytes = response.content
        elif isinstance(output, list) and len(output) > 0:
            # Список URL
            import requests
            logger.info("📥 Загрузка результата...")
            response = requests.get(output[0], timeout=30)
            result_bytes = response.content
        else:
            logger.error(f"❌ Неизвестный формат output: {type(output)}")
            return opencv_fallback(image, mask)
        
        # Конвертируем в numpy array
        result_pil = Image.open(BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert('RGB'))
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        logger.info("✅ Replicate SDK inpainting выполнен успешно!")
        return result_bgr
        
    except ImportError:
        logger.error("❌ Библиотека replicate не установлена!")
        logger.error("Добавьте 'replicate' в requirements.txt")
        return opencv_fallback(image, mask)
    
    except Exception as e:
        logger.error(f"❌ Ошибка Replicate SDK: {e}")
        logger.info("Используем OpenCV fallback")
        return opencv_fallback(image, mask)


def remove_text(image_path: str, mask_path: str, output_path: str) -> bool:
    """
    Основная функция удаления текста
    """
    try:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            logger.error("❌ Не удалось загрузить файлы")
            return False
        
        result = replicate_inpaint(image, mask)
        cv2.imwrite(output_path, result)
        
        logger.info(f"✅ Результат сохранён: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return False
