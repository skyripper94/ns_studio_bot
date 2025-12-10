"""
Интеграция с LaMa для удаления текста
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import torch
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Путь к модели
MODEL_PATH = Path(os.getenv('LAMA_MODEL_PATH', '/app/models/big-lama'))


class LamaInpainter:
    """Класс для работы с LaMa моделью"""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Используем устройство: {self.device}")
    
    def load_model(self):
        """Загружает LaMa модель"""
        if self.model is not None:
            return True
        
        try:
            # Проверяем наличие модели
            if not MODEL_PATH.exists():
                logger.warning(f"Модель не найдена в {MODEL_PATH}")
                logger.info("Попытка скачать модель...")
                self._download_model()
            
            # Загружаем модель
            logger.info("Загружаем LaMa модель...")
            
            # TODO: Здесь будет реальная загрузка LaMa
            # Для Railway деплоя используем упрощенную версию
            
            # Пока используем простой OpenCV inpainting как fallback
            self.model = "opencv_fallback"
            logger.info("✅ Модель загружена (OpenCV fallback)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def _download_model(self):
        """Скачивает модель с HuggingFace"""
        import urllib.request
        import zipfile
        
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        model_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
        zip_path = MODEL_PATH.parent / "big-lama.zip"
        
        logger.info(f"Скачиваем модель с {model_url}")
        
        # Скачиваем
        urllib.request.urlretrieve(model_url, zip_path)
        
        # Распаковываем
        logger.info("Распаковываем модель...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH.parent)
        
        # Удаляем архив
        zip_path.unlink()
        logger.info("✅ Модель скачана")
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Удаляет области по маске
        
        Args:
            image: исходное изображение (BGR)
            mask: маска (255 = удалить, 0 = оставить)
            
        Returns:
            result: обработанное изображение
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Убеждаемся что маска правильного типа
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # Используем комбинацию методов для лучшего качества
            # Сначала NS (лучше для текстур)
            result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
            
            # Затем дополнительный проход TELEA для сглаживания
            result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            
            # Небольшое размытие только в области маски для сглаживания границ
            mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
            mask_normalized = mask_blur.astype(float) / 255.0
            
            # Применяем легкое размытие только в замененных областях
            blurred = cv2.bilateralFilter(result, 5, 50, 50)
            result = (result * (1 - mask_normalized[:,:,np.newaxis]) + 
                     blurred * mask_normalized[:,:,np.newaxis]).astype(np.uint8)
            
            logger.info("✅ Inpainting выполнен (улучшенный OpenCV)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка inpainting: {e}")
            return image
    
    def process_file(self, image_path: Path, mask_path: Path, output_path: Path):
        """
        Обрабатывает файл изображения
        
        Args:
            image_path: путь к изображению
            mask_path: путь к маске
            output_path: путь для сохранения результата
        """
        # Читаем изображение и маску
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError("Не удалось загрузить изображение или маску")
        
        # Обрабатываем
        result = self.inpaint(image, mask)
        
        # Сохраняем
        cv2.imwrite(str(output_path), result)
        logger.info(f"Результат сохранен: {output_path}")


# Глобальный экземпляр
_inpainter = None

def get_inpainter() -> LamaInpainter:
    """Получить глобальный инстанс инпейнтера"""
    global _inpainter
    if _inpainter is None:
        _inpainter = LamaInpainter()
    return _inpainter