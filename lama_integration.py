"""
Интеграция с IOPaint для качественного удаления текста
IOPaint - готовая обертка над LaMa с простым API
"""

import os
import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Пробуем импортировать IOPaint
try:
    from iopaint.model_manager import ModelManager
    from iopaint.schema import Config, HDStrategy, LDMSampler
    IOPAINT_AVAILABLE = True
    logger.info("✅ IOPaint доступен")
except ImportError:
    IOPAINT_AVAILABLE = False
    logger.warning("⚠️ IOPaint недоступен, используем OpenCV fallback")


class LamaInpainter:
    """Инпейнтер с IOPaint (LaMa) или OpenCV fallback"""
    
    def __init__(self):
        self.model = None
        self.device = 'cpu'  # IOPaint автоматически выбирает GPU если есть
        
        if IOPAINT_AVAILABLE:
            self._init_iopaint()
        else:
            logger.info("Используем OpenCV fallback")
    
    def _init_iopaint(self):
        """Инициализация IOPaint модели"""
        try:
            # Конфигурация IOPaint
            config = Config(
                ldm_steps=25,
                ldm_sampler=LDMSampler.ddim,
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=128,
                hd_strategy_crop_trigger_size=800,
                hd_strategy_resize_limit=1024,
            )
            
            # Загружаем LaMa модель
            self.model = ModelManager(
                name="lama",
                device=self.device,
            )
            
            self.config = config
            logger.info("✅ IOPaint LaMa модель загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки IOPaint: {e}")
            self.model = None
    
    def load_model(self):
        """Совместимость со старым API"""
        if self.model is None and IOPAINT_AVAILABLE:
            self._init_iopaint()
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Удаляет области по маске используя IOPaint (LaMa) или OpenCV
        
        Args:
            image: исходное изображение (BGR)
            mask: маска (255 = удалить, 0 = оставить)
            
        Returns:
            result: обработанное изображение
        """
        # Если IOPaint доступен и модель загружена
        if IOPAINT_AVAILABLE and self.model is not None:
            try:
                return self._iopaint_inpaint(image, mask)
            except Exception as e:
                logger.error(f"❌ Ошибка IOPaint inpainting: {e}")
                logger.info("Переключаемся на OpenCV fallback")
                return self._opencv_inpaint(image, mask)
        else:
            # Fallback на улучшенный OpenCV
            return self._opencv_inpaint(image, mask)
    
    def _iopaint_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """IOPaint инпейнтинг (высокое качество)"""
        # Конвертируем BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в PIL Image
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(mask)
        
        # Инпейнтинг через IOPaint
        result_pil = self.model(pil_image, pil_mask, self.config)
        
        # Конвертируем обратно в numpy BGR
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        logger.info("✅ IOPaint (LaMa) inpainting выполнен")
        return result_bgr
    
    def _opencv_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Улучшенный OpenCV инпейнтинг (fallback)"""
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        try:
            # Двойной проход для лучшего качества
            result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
            result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            
            # Легкое размытие в области маски для сглаживания границ
            mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
            mask_normalized = mask_blur.astype(float) / 255.0
            
            blurred = cv2.bilateralFilter(result, 5, 50, 50)
            result = (result * (1 - mask_normalized[:,:,np.newaxis]) + 
                     blurred * mask_normalized[:,:,np.newaxis]).astype(np.uint8)
            
            logger.info("✅ OpenCV inpainting выполнен")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка OpenCV inpainting: {e}")
            return image


# Глобальный инстанс
_inpainter = None

def get_inpainter() -> LamaInpainter:
    """Получить глобальный инстанс инпейнтера"""
    global _inpainter
    if _inpainter is None:
        _inpainter = LamaInpainter()
    return _inpainter
