"""
Интеграция с Replicate API для запуска LaMa модели
Универсальная версия - работает с любой моделью inpainting на Replicate
"""

import os
import logging
import numpy as np
import cv2
import requests
import time
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Конфигурация из переменных окружения
REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY', '')
# Модель по умолчанию: cjwbw/lama (стабильная LaMa модель)
REPLICATE_MODEL = os.getenv('REPLICATE_MODEL', 'cjwbw/lama')


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
    Удаляет области по маске используя Replicate API
    """
    if not REPLICATE_API_KEY:
        logger.warning("⚠️ REPLICATE_API_KEY не установлен, используем OpenCV fallback")
        return opencv_fallback(image, mask)
    
    try:
        logger.info(f"🚀 Запуск Replicate API (модель: {REPLICATE_MODEL})...")
        
        # Конвертируем изображение в base64
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Конвертируем маску в base64
        pil_mask = Image.fromarray(mask)
        mask_buffer = BytesIO()
        pil_mask.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Создаём prediction используя упрощённый формат
        logger.info("📤 Отправка запроса...")
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Bearer {REPLICATE_API_KEY}",
                "Content-Type": "application/json",
                "Prefer": "wait"  # Ждём результата синхронно
            },
            json={
                "model": REPLICATE_MODEL,
                "input": {
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                }
            },
            timeout=120  # 2 минуты на обработку
        )
        
        if response.status_code not in [200, 201]:
            logger.error(f"❌ Replicate ошибка: {response.status_code} - {response.text}")
            return opencv_fallback(image, mask)
        
        result_data = response.json()
        
        # Проверяем статус
        status = result_data.get('status')
        
        if status == 'succeeded':
            result_url = result_data.get('output')
            
            if not result_url:
                logger.error("❌ Replicate вернул пустой output")
                return opencv_fallback(image, mask)
            
            # Если output это список, берём первый элемент
            if isinstance(result_url, list):
                result_url = result_url[0]
            
            # Загружаем результат
            logger.info("📥 Загрузка результата...")
            result_response = requests.get(result_url, timeout=30)
            
            if result_response.status_code != 200:
                logger.error(f"❌ Ошибка загрузки: {result_response.status_code}")
                return opencv_fallback(image, mask)
            
            # Конвертируем в numpy array
            result_pil = Image.open(BytesIO(result_response.content))
            result_rgb = np.array(result_pil.convert('RGB'))
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            
            logger.info("✅ Replicate (LaMa) inpainting выполнен успешно!")
            return result_bgr
            
        elif status == 'failed':
            error = result_data.get('error', 'Unknown error')
            logger.error(f"❌ Replicate failed: {error}")
            return opencv_fallback(image, mask)
        
        else:
            # Если статус processing, пробуем подождать
            prediction_id = result_data.get('id')
            get_url = result_data.get('urls', {}).get('get')
            
            if not get_url:
                logger.error("❌ Нет URL для проверки статуса")
                return opencv_fallback(image, mask)
            
            logger.info("⏳ Ожидание обработки...")
            
            # Ждём до 90 секунд
            for attempt in range(90):
                time.sleep(1)
                
                status_response = requests.get(
                    get_url,
                    headers={"Authorization": f"Bearer {REPLICATE_API_KEY}"},
                    timeout=10
                )
                
                if status_response.status_code != 200:
                    logger.error(f"❌ Ошибка проверки статуса")
                    return opencv_fallback(image, mask)
                
                status_data = status_response.json()
                current_status = status_data.get('status')
                
                if current_status == 'succeeded':
                    result_url = status_data.get('output')
                    
                    if isinstance(result_url, list):
                        result_url = result_url[0]
                    
                    result_response = requests.get(result_url, timeout=30)
                    result_pil = Image.open(BytesIO(result_response.content))
                    result_rgb = np.array(result_pil.convert('RGB'))
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                    
                    logger.info(f"✅ Replicate выполнен за {attempt+1}с")
                    return result_bgr
                
                elif current_status == 'failed':
                    logger.error(f"❌ Replicate failed")
                    return opencv_fallback(image, mask)
                
                # Показываем прогресс
                if attempt % 10 == 0:
                    logger.info(f"⏳ Обработка... {attempt}s")
            
            logger.error("❌ Timeout")
            return opencv_fallback(image, mask)
        
    except requests.exceptions.Timeout:
        logger.error("❌ Timeout")
        return opencv_fallback(image, mask)
    
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
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
