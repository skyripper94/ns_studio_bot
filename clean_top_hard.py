#!/usr/bin/env python3
"""
Жёсткая очистка верхней части изображения от жёлтых полос и текстуры.
Использование: python clean_top_hard.py input.jpg [--debug]
Результат: output.png
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# OpenCV опционален
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


def clean_top_hard(
    img: Image.Image,
    max_top_ratio: float = 0.30,
    line_band_top_ratio: float = 0.04,
    band_above_header: int = 60,
    feather: int = 28,
    inpaint_radius: int = 4,
    debug: bool = False,
    debug_prefix: str = "debug"
) -> Image.Image:
    """
    Жёстко очищает верхнюю часть изображения от жёлтых полос и текстуры.
    
    Args:
        img: Входное изображение (PIL Image RGB)
        max_top_ratio: Максимальная доля высоты для обработки (0.30 = 30%)
        line_band_top_ratio: Верхняя полоса для тонких линий (0.04 = 4%)
        band_above_header: Зона очистки выше заголовка (px)
        feather: Размер пера для плавного перехода (px)
        inpaint_radius: Радиус inpaint (3-5)
        debug: Сохранять отладочные изображения
        debug_prefix: Префикс для отладочных файлов
    
    Returns:
        Очищенное изображение (PIL Image RGB)
    """
    w, h = img.size
    max_top_px = int(h * max_top_ratio)
    line_band_top_px = int(h * line_band_top_ratio)
    
    # Конвертируем в numpy (RGB)
    img_np = np.array(img)
    
    # ==================== ШАГ 1: Поиск базовой линии заголовка ====================
    # Ищем бирюзовый текст (H≈175-195 в OpenCV HSV, т.е. ~170-180)
    baseline_y = int(h * 0.70)  # fallback
    
    if CV2_AVAILABLE:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # Бирюзовый/cyan: H=80-100 в OpenCV (0-180 scale), S>50, V>100
        cyan_mask = cv2.inRange(hsv, (80, 50, 100), (100, 255, 255))
        
        # Ищем самую верхнюю строку с бирюзовыми пикселями
        rows_with_cyan = np.where(cyan_mask.any(axis=1))[0]
        if len(rows_with_cyan) > 0:
            # Берём верхнюю границу бирюзового текста
            baseline_y = int(rows_with_cyan[0])
            print(f"✓ Found cyan text baseline at y={baseline_y}")
        else:
            print(f"⚠️ Cyan text not found, using fallback baseline y={baseline_y}")
    else:
        # Fallback без OpenCV: ищем яркие голубые пиксели
        for y in range(max_top_px, h):
            row = img_np[y, :, :]
            # Бирюзовый: B > R, G > R, B > 150
            cyan_pixels = np.sum((row[:, 2] > row[:, 0]) & (row[:, 1] > row[:, 0]) & (row[:, 2] > 150))
            if cyan_pixels > w * 0.05:  # минимум 5% ширины
                baseline_y = y
                print(f"✓ Found approximate baseline at y={baseline_y}")
                break
    
    # ==================== ШАГ 2: Определение целевой зоны очистки ====================
    # Зона 1: от (baseline - 60px) до (baseline - 2px)
    zone_header_start = max(0, baseline_y - band_above_header)
    zone_header_end = max(0, baseline_y - 2)
    
    # Зона 2: верхний край 0-4% для тонких линий
    zone_top_start = 0
    zone_top_end = line_band_top_px
    
    # Объединяем зоны (берём всё от 0 до zone_header_end, но ограничиваем max_top_ratio)
    clean_zone_start = 0
    clean_zone_end = min(zone_header_end, max_top_px)
    
    print(f"[Zones] Header zone: {zone_header_start}-{zone_header_end}px, Top band: 0-{zone_top_end}px")
    print(f"[Clean zone] 0-{clean_zone_end}px (max allowed: {max_top_px}px)")
    
    if clean_zone_end <= 0:
        print("⚠️ Clean zone is empty, skipping")
        return img
    
    # ==================== ШАГ 3: Построение маски грязных пикселей ====================
    # Жёлтый/оранжевый в HSV: H≈10-42 (в OpenCV 0-180 scale это 5-21), S>40, V>40
    
    clean_zone = img_np[clean_zone_start:clean_zone_end, :, :]
    zone_h = clean_zone_end - clean_zone_start
    
    if CV2_AVAILABLE:
        zone_hsv = cv2.cvtColor(clean_zone, cv2.COLOR_RGB2HSV)
        
        # Маска жёлтых/оранжевых пикселей (H=5-25 в OpenCV, S>40, V>40)
        yellow_mask = cv2.inRange(zone_hsv, (5, 40, 40), (25, 255, 255))
        
        # Добавляем коричневые оттенки (H=0-10)
        brown_mask = cv2.inRange(zone_hsv, (0, 30, 30), (10, 255, 200))
        dirty_mask = cv2.bitwise_or(yellow_mask, brown_mask)
        
        # ==================== ШАГ 4: Морфологическая обработка маски ====================
        # Open (3x3) — убираем шум
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dirty_mask = cv2.morphologyEx(dirty_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Close (7x7) — заполняем дыры
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dirty_mask = cv2.morphologyEx(dirty_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Top-hat для горизонтальных полос (25x1)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horiz_lines = cv2.morphologyEx(zone_hsv[:, :, 2], cv2.MORPH_TOPHAT, kernel_horiz)
        horiz_mask = cv2.threshold(horiz_lines, 30, 255, cv2.THRESH_BINARY)[1]
        dirty_mask = cv2.bitwise_or(dirty_mask, horiz_mask)
        
        # Dilate (5x1) — расширяем маску горизонтально
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        dirty_mask = cv2.dilate(dirty_mask, kernel_dilate, iterations=2)
        
        mask_coverage = np.sum(dirty_mask > 0) / dirty_mask.size * 100
        print(f"[Mask] Dirty pixels: {mask_coverage:.2f}% of clean zone")
        
    else:
        # Fallback без OpenCV: простая маска по RGB
        dirty_mask = np.zeros((zone_h, w), dtype=np.uint8)
        for y in range(zone_h):
            for x in range(w):
                r, g, b = clean_zone[y, x]
                # Жёлтый: R > 150, G > 100, B < 100, R - B > 50
                is_yellow = (r > 150 and g > 100 and b < 100 and r - b > 50)
                # Оранжевый/коричневый: R > G > B, R > 100
                is_orange = (r > 100 and g > 70 and b < 90 and r > g > b)
                if is_yellow or is_orange:
                    dirty_mask[y, x] = 255
        
        # Простое расширение маски через PIL
        mask_pil = Image.fromarray(dirty_mask)
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(3))
        dirty_mask = np.array(mask_pil)
        
        mask_coverage = np.sum(dirty_mask > 0) / dirty_mask.size * 100
        print(f"[Mask] Dirty pixels (fallback): {mask_coverage:.2f}% of clean zone")
    
    # ==================== ШАГ 5: Получение цвета заливки ====================
    # Берём средний цвет из полосы baseline - 40 ... baseline - 20 px
    sample_start = max(0, baseline_y - 40)
    sample_end = max(10, baseline_y - 20)
    
    if sample_end > sample_start and sample_end <= h:
        sample_zone = img_np[sample_start:sample_end, :, :]
        # Исключаем жёлтые пиксели из выборки
        sample_flat = sample_zone.reshape(-1, 3)
        
        # Фильтруем: убираем жёлтые (R > G > B and R - B > 40)
        non_yellow = sample_flat[~((sample_flat[:, 0] > sample_flat[:, 1]) & 
                                    (sample_flat[:, 1] > sample_flat[:, 2]) & 
                                    (sample_flat[:, 0] - sample_flat[:, 2] > 40))]
        
        if len(non_yellow) > 0:
            fill_color = np.median(non_yellow, axis=0).astype(np.uint8)
        else:
            fill_color = np.median(sample_flat, axis=0).astype(np.uint8)
    else:
        fill_color = np.array([30, 30, 35], dtype=np.uint8)
    
    print(f"[Fill color] RGB: {tuple(fill_color)}")
    
    # ==================== ШАГ 6: Очистка (inpaint или заливка) ====================
    result_zone = clean_zone.copy()
    
    if CV2_AVAILABLE and np.sum(dirty_mask > 0) > 0:
        # Inpaint с OpenCV
        result_zone_bgr = cv2.cvtColor(result_zone, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(result_zone_bgr, dirty_mask, inpaint_radius, cv2.INPAINT_TELEA)
        result_zone = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        print(f"✓ Inpaint applied (radius={inpaint_radius})")
    else:
        # Fallback: заливка по маске
        mask_bool = dirty_mask > 128
        result_zone[mask_bool] = fill_color
        print("✓ Fill applied (fallback)")
    
    # Если маска слабая (< 5%), всё равно затемняем зону заливкой
    if mask_coverage < 5.0:
        print("⚠️ Weak mask, applying full zone darkening")
        # Создаём затемнённую версию
        dark_factor = 0.85
        darkened = (result_zone.astype(np.float32) * dark_factor).astype(np.uint8)
        result_zone = darkened
    
    # ==================== ШАГ 7: Градиент сверху ====================
    # Создаём градиент от тёмного (верх) к прозрачному (низ)
    gradient_height = max(clean_zone_end, feather * 2)
    
    # Цвет градиента: на 10-15% темнее fill_color
    gradient_top_color = (fill_color.astype(np.float32) * 0.85).astype(np.uint8)
    
    # Применяем вертикальный градиент
    for y in range(gradient_height):
        if y >= zone_h:
            break
        # t: 0 вверху (полностью тёмный) → 1 внизу (прозрачный)
        t = y / gradient_height
        # Квадратичное затухание для более плавного перехода
        alpha = (1 - t) ** 2
        
        # Смешиваем с градиентным цветом
        result_zone[y] = (
            result_zone[y].astype(np.float32) * (1 - alpha * 0.7) + 
            gradient_top_color.astype(np.float32) * (alpha * 0.7)
        ).astype(np.uint8)
    
    print(f"✓ Gradient applied (height={gradient_height}px)")
    
    # ==================== ШАГ 8: Перо на нижней границе ====================
    feather_start = max(0, zone_h - feather)
    for y in range(feather_start, zone_h):
        # t: 0 в начале пера → 1 в конце (граница с оригиналом)
        t = (y - feather_start) / feather
        # Плавное перо
        blend = t ** 1.5
        
        # Смешиваем с оригиналом
        orig_row = clean_zone[y]
        result_zone[y] = (
            result_zone[y].astype(np.float32) * (1 - blend) + 
            orig_row.astype(np.float32) * blend
        ).astype(np.uint8)
    
    print(f"✓ Feather applied ({feather}px)")
    
    # ==================== ШАГ 9: Собираем результат ====================
    result_np = img_np.copy()
    result_np[clean_zone_start:clean_zone_end, :, :] = result_zone
    
    result_img = Image.fromarray(result_np)
    
    # ==================== ШАГ 10: Отладочные изображения ====================
    if debug:
        # Маска
        mask_img = Image.fromarray(dirty_mask)
        mask_img.save(f"{debug_prefix}_mask.png")
        print(f"✓ Saved {debug_prefix}_mask.png")
        
        # Debug с показом зон
        debug_np = img_np.copy()
        # Рисуем границы зон
        # Красная линия — baseline
        if baseline_y < h:
            debug_np[baseline_y, :] = [255, 0, 0]
        # Зелёная линия — верхняя граница очистки
        if zone_header_start < h:
            debug_np[zone_header_start, :] = [0, 255, 0]
        # Синяя линия — нижняя граница очистки
        if clean_zone_end < h:
            debug_np[clean_zone_end, :] = [0, 0, 255]
        
        debug_img = Image.fromarray(debug_np)
        debug_img.save(f"{debug_prefix}_debug.png")
        print(f"✓ Saved {debug_prefix}_debug.png")
    
    print("✓ Top cleaning completed")
    return result_img


# ==================== CLI ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_top_hard.py input.jpg [--debug]")
        print("Result: output.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    debug_mode = "--debug" in sys.argv
    output_path = "output.png"
    
    print(f"Loading {input_path}...")
    img = Image.open(input_path).convert("RGB")
    print(f"Image size: {img.size}")
    print(f"OpenCV available: {CV2_AVAILABLE}")
    print(f"Debug mode: {debug_mode}")
    
    result = clean_top_hard(
        img,
        max_top_ratio=0.30,
        line_band_top_ratio=0.04,
        band_above_header=60,
        feather=28,
        inpaint_radius=4,
        debug=debug_mode,
        debug_prefix="clean_top"
    )
    
    print(f"Saving to {output_path}...")
    result.save(output_path, format="PNG")
    print(f"✅ Done! Check {output_path}")
