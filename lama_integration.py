"""
Полная интеграция LaMa для максимального качества (10/10)
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import torch
from PIL import Image
import cv2
import yaml

logger = logging.getLogger(__name__)

# Путь к модели
MODEL_PATH = Path(os.getenv('LAMA_MODEL_PATH', '/app/models/big-lama'))


class LamaInpainterFull:
    """Полная интеграция с LaMa для максимального качества"""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Используем устройство: {self.device}")
    
    def load_model(self):
        """Загружает полную LaMa модель"""
        if self.model is not None:
            return True
        
        try:
            # Проверяем наличие модели
            if not MODEL_PATH.exists():
                logger.warning(f"Модель не найдена в {MODEL_PATH}")
                return False
            
            logger.info("Загружаем полную LaMa модель...")
            
            # Добавляем путь к LaMa в sys.path
            lama_path = MODEL_PATH.parent.parent / 'lama'
            if lama_path.exists():
                sys.path.insert(0, str(lama_path))
            
            # Импортируем LaMa модули
            try:
                from saicinpainting.training.trainers import load_checkpoint
                from omegaconf import OmegaConf
            except ImportError:
                logger.warning("Не удалось импортировать LaMa модули")
                return False
            
            # Загружаем конфигурацию
            config_path = MODEL_PATH / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = OmegaConf.create(yaml.safe_load(f))
            else:
                logger.warning("config.yaml не найден")
                return False
            
            # Загружаем checkpoint
            checkpoint_path = MODEL_PATH / 'models' / 'best.ckpt'
            if not checkpoint_path.exists():
                checkpoint_path = MODEL_PATH / 'big-lama.ckpt'
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint не найден: {checkpoint_path}")
                return False
            
            self.model = load_checkpoint(config, str(checkpoint_path), strict=False, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)
            
            logger.info("✅ Полная LaMa модель загружена")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки LaMa: {e}")
            return False
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Удаляет области используя полную LaMa
        
        Args:
            image: исходное изображение (BGR)
            mask: маска (255 = удалить, 0 = оставить)
            
        Returns:
            result: обработанное изображение
        """
        if self.model is None:
            if not self.load_model():
                # Fallback на OpenCV если LaMa не загрузилась
                logger.warning("Используем OpenCV fallback")
                return self._opencv_inpaint(image, mask)
        
        try:
            # Конвертируем в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Нормализуем изображение [0, 1]
            image_norm = image_rgb.astype(np.float32) / 255.0
            
            # Нормализуем маску [0, 1]
            mask_norm = mask.astype(np.float32) / 255.0
            if len(mask_norm.shape) == 2:
                mask_norm = mask_norm[:, :, np.newaxis]
            
            # Конвертируем в torch tensors
            image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Инпейнтинг
            with torch.no_grad():
                batch = {
                    'image': image_tensor,
                    'mask': mask_tensor
                }
                result_tensor = self.model(batch)
                
                # Извлекаем результат
                if isinstance(result_tensor, dict):
                    result_tensor = result_tensor.get('inpainted', result_tensor.get('predicted_image', result_tensor))
                
                # Конвертируем обратно в numpy
                result_np = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
            
            # Конвертируем обратно в BGR
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            logger.info("✅ LaMa inpainting выполнен")
            return result_bgr
            
        except Exception as e:
            logger.error(f"❌ Ошибка LaMa inpainting: {e}")
            # Fallback на OpenCV
            return self._opencv_inpaint(image, mask)
    
    def _opencv_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback OpenCV inpainting"""
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Улучшенный OpenCV метод
        result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
        result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        return result


# Глобальный экземпляр
_inpainter_full = None

def get_full_inpainter() -> LamaInpainterFull:
    """Получить глобальный инстанс полного инпейнтера"""
    global _inpainter_full
    if _inpainter_full is None:
        _inpainter_full = LamaInpainterFull()
    return _inpainter_full