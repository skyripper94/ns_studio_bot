#!/usr/bin/env python3
"""
Тестовый скрипт для функции clean_top_yellow_artifacts.
Использование: python test_clean_top.py input.jpg
Результат сохраняется в output.png
"""

import sys
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def clean_top_yellow_artifacts(img: Image.Image):
    """
    Очищает верхнюю часть изображения (до 30% высоты) от жёлтых/коричневых артефактов
    и накладывает мягкий вертикальный градиент сверху вниз.
    """
    w, h = img.size
    # Работаем только с верхними 30% изображения
    top_zone_height = int(h * 0.30)
    
    # Конвертируем в numpy для анализа цвета
    img_array = np.array(img)
    top_zone = img_array[:top_zone_height, :, :]
    
    # Создаём маску для жёлтых/коричневых оттенков в RGB
    # Жёлтый: R высокий, G высокий, B низкий
    # Коричневый: R > G > B, всё средние-высокие значения
    mask = np.zeros((top_zone_height, w), dtype=np.uint8)
    
    for y in range(top_zone_height):
        for x in range(w):
            r, g, b = top_zone[y, x]
            # Условия для жёлтых/коричневых оттенков
            is_yellow = (r > 150 and g > 120 and b < 100 and r - b > 50)
            is_brown = (r > 100 and g > 70 and b < 80 and r > g > b)
            
            if is_yellow or is_brown:
                mask[y, x] = 255
    
    # Расширяем маску для захвата размытых краёв
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))  # расширение
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(8))  # смягчение
    mask_array = np.array(mask_pil)
    
    # Если есть артефакты — удаляем их
    if mask_array.max() > 10:
        # Получаем доминирующий цвет верхней зоны (медиана без жёлтых областей)
        clean_pixels = top_zone[mask_array < 128]
        if len(clean_pixels) > 0:
            dominant_color = np.median(clean_pixels, axis=0).astype(np.uint8)
        else:
            # Fallback: тёмно-серый
            dominant_color = np.array([40, 40, 45], dtype=np.uint8)
        
        # Заменяем артефакты на доминирующий цвет
        img_rgba = img.convert("RGBA")
        fill_layer = Image.new("RGBA", (w, h), tuple(dominant_color) + (255,))
        
        # Применяем только в верхней зоне по маске
        full_mask = Image.new("L", (w, h), 0)
        full_mask.paste(mask_pil, (0, 0))
        
        result = Image.composite(fill_layer, img_rgba, full_mask)
        img = result.convert("RGB")
        print(f"✓ Cleaned yellow artifacts in top {top_zone_height}px, dominant color: {dominant_color}")
    else:
        print("✓ No yellow artifacts detected in top zone")
    
    # Накладываем мягкий вертикальный градиент сверху вниз
    img_rgba = img.convert("RGBA")
    gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(gradient)
    
    # Градиент в верхних 20% высоты (от тёмного к прозрачному)
    gradient_height = int(h * 0.20)
    steps = max(50, gradient_height)
    
    for i in range(steps):
        t = i / steps
        # Квадратичное затухание (темнее вверху, прозрачно внизу)
        alpha = int(180 * ((1 - t) ** 2))
        y = int(i * gradient_height / steps)
        if y < h:
            d.rectangle([(0, y), (w, y+2)], fill=(15, 15, 20, alpha))
    
    # Применяем градиент
    result = Image.alpha_composite(img_rgba, gradient)
    print(f"✓ Applied soft top gradient ({gradient_height}px)")
    
    return result.convert("RGB")

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
