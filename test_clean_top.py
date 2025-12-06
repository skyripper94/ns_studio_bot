#!/usr/bin/env python3
"""
Тестовый скрипт для функции clean_top_yellow_artifacts.
Использование: python test_clean_top.py input.jpg
Результат сохраняется в output.png
"""

import sys
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def clean_top_yellow_artifacts(img: Image.Image, logo_y_estimate=None):
    """
    Очищает узкую полосу 30-50px над логотипом от жёлтых/коричневых артефактов
    и накладывает мягкий вертикальный градиент.
    """
    w, h = img.size
    
    # Оцениваем положение логотипа (примерно 15-20% от верха для нижнего градиента)
    if logo_y_estimate is None:
        # Логотип обычно рисуется в зоне fade_top + смещение
        # Приблизительно это 40-50% высоты изображения
        logo_y_estimate = int(h * 0.45)
    
    # Работаем только с полосой 30-50px НАД логотипом
    detection_zone_height = 50  # высота зоны детекции
    zone_start = max(0, logo_y_estimate - detection_zone_height - 30)  # 30px запас сверху лого
    zone_end = max(50, logo_y_estimate - 30)  # до 30px перед логотипом
    
    # Конвертируем в numpy для анализа цвета
    img_array = np.array(img)
    detection_zone = img_array[zone_start:zone_end, :, :]
    zone_height = zone_end - zone_start
    
    if zone_height <= 0:
        print("✓ Skip artifact cleaning (invalid zone)")
        return img
    
    # Создаём маску для жёлтых/коричневых оттенков в RGB
    mask = np.zeros((zone_height, w), dtype=np.uint8)
    
    for y in range(zone_height):
        for x in range(w):
            r, g, b = detection_zone[y, x]
            # Условия для жёлтых/коричневых оттенков
            is_yellow = (r > 150 and g > 120 and b < 100 and r - b > 50)
            is_brown = (r > 100 and g > 70 and b < 80 and r > g > b)
            
            if is_yellow or is_brown:
                mask[y, x] = 255
    
    # Расширяем маску для захвата размытых краёв
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))  # меньшее расширение
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(5))  # смягчение
    mask_array = np.array(mask_pil)
    
    # Если есть артефакты — удаляем их
    if mask_array.max() > 10:
        # Получаем доминирующий цвет зоны (медиана без жёлтых областей)
        clean_pixels = detection_zone[mask_array < 128]
        if len(clean_pixels) > 0:
            dominant_color = np.median(clean_pixels, axis=0).astype(np.uint8)
        else:
            # Fallback: тёмный оттенок
            dominant_color = np.array([35, 35, 40], dtype=np.uint8)
        
        # Заменяем артефакты на доминирующий цвет
        img_rgba = img.convert("RGBA")
        fill_layer = Image.new("RGBA", (w, h), tuple(dominant_color) + (255,))
        
        # Применяем только в зоне детекции
        full_mask = Image.new("L", (w, h), 0)
        full_mask.paste(mask_pil, (0, zone_start))
        
        result = Image.composite(fill_layer, img_rgba, full_mask)
        img = result.convert("RGB")
        print(f"✓ Cleaned yellow artifacts in zone [{zone_start}:{zone_end}]px, color: {dominant_color}")
        
        # Накладываем лёгкий градиент ТОЛЬКО в зоне очистки + 20px сверху
        img_rgba = img.convert("RGBA")
        gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(gradient)
        
        # Градиент от начала зоны до её конца + небольшой запас
        gradient_start = max(0, zone_start - 20)
        gradient_height = (zone_end - gradient_start) + 30  # +30px запас вниз
        steps = max(30, gradient_height)
        
        for i in range(steps):
            t = i / steps
            # Квадратичное затухание (темнее вверху, прозрачно внизу)
            alpha = int(160 * ((1 - t) ** 2.5))  # более резкое затухание
            y = gradient_start + int(i * gradient_height / steps)
            if 0 <= y < h:
                d.rectangle([(0, y), (w, y+2)], fill=(10, 10, 15, alpha))
        
        # Применяем градиент
        result = Image.alpha_composite(img_rgba, gradient)
        print(f"✓ Applied localized gradient [{gradient_start}:{gradient_start+gradient_height}]px")
        return result.convert("RGB")
    else:
        print(f"✓ No yellow artifacts detected in zone [{zone_start}:{zone_end}]px")
        return img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_clean_top.py input.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = "output.png"
    
    print(f"Loading {input_path}...")
    img = Image.open(input_path).convert("RGB")
    print(f"Image size: {img.size}")
    
    print("Cleaning top zone...")
    result = clean_top_yellow_artifacts(img)
    
    print(f"Saving to {output_path}...")
    result.save(output_path, format="PNG", quality=95)
    print(f"✅ Done! Check {output_path}")
