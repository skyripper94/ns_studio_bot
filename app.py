from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io, base64, os
import numpy as np

# OpenCV опционален: если поставится на Railway — используем, нет — работаем через PIL.
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

app = Flask(__name__)

# ----------------------------- Fonts -----------------------------

def get_font(size, weight="bold"):
    if weight == "bold":
        paths = [
            os.path.join(os.path.dirname(__file__), "fonts", "gotham_bold.otf"),
            os.path.join(os.path.dirname(__file__), "fonts", "Exo2-Bold.ttf"),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        paths = [
            os.path.join(os.path.dirname(__file__), "fonts", "gotham_medium.otf"),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for p in paths:
        try:
            if os.path.exists(p):
                print(f"✓ Font loaded: {p} (size: {size})")
                return ImageFont.truetype(p, size)
        except Exception as e:
            print(f"✗ Font load fail {p}: {e}")
    print(f"⚠️ Default font (size: {size})")
    return ImageFont.load_default()

# ----------------------------- Helpers -----------------------------

def calculate_adaptive_gradient(img, long_text=False):
    """Возвращает процент высоты фейда снизу (мягкий тёплый градиент Instagram-style)."""
    w, h = img.size
    bottom = img.crop((0, h//2, w, h)).convert("L")
    arr = np.array(bottom, dtype=np.uint8)
    avg = float(arr.mean())

    # Увеличенная зона градиента для более плавного Instagram-style перехода
    if avg > 150:      gp = 0.48
    elif avg > 100:    gp = 0.45
    else:              gp = 0.42

    if long_text:
        gp = max(gp, 0.36)

    print(f"[Adaptive Gradient] Brightness: {avg:.0f}, Gradient: {gp*100:.0f}%")
    return gp

def clean_top_yellow_artifacts(img: Image.Image, logo_y_estimate=None):
    """
    Точечно очищает жёлтые артефакты от размытия текста.
    Используется ПОСЛЕ удаления старого текста, ПЕРЕД нанесением нового.
    Работает в узкой зоне 25-70px над предполагаемым логотипом.
    """
    w, h = img.size
    
    # Используем переданную оценку или берём 52% высоты (где обычно логотип)
    if logo_y_estimate is None:
        logo_y_estimate = int(h * 0.52)
    
    # Узкая зона очистки: только там где артефакты от старого текста
    band_above_header = 70  # px над заголовком для поиска артефактов
    clean_zone_start = max(0, logo_y_estimate - band_above_header)
    clean_zone_end = max(10, logo_y_estimate - 25)  # до 25px перед логотипом
    
    zone_h = clean_zone_end - clean_zone_start
    
    if zone_h <= 10:
        print("⚠️ Clean zone too small, skipping")
        return img
    
    # Конвертируем в numpy (RGB)
    img_np = np.array(img)
    clean_zone = img_np[clean_zone_start:clean_zone_end, :, :]
    
    # ШАГ 1: Построение маски жёлтых/оранжевых пикселей
    if CV2_AVAILABLE:
        zone_hsv = cv2.cvtColor(clean_zone, cv2.COLOR_RGB2HSV)
        
        # Только явно жёлтые/оранжевые пиксели (не трогаем нормальные оттенки)
        # Жёлтый: H=10-30, S>50, V>60 (более строгие критерии)
        yellow_mask = cv2.inRange(zone_hsv, (10, 50, 60), (30, 255, 255))
        
        # Оранжевый: H=5-15, S>60, V>70
        orange_mask = cv2.inRange(zone_hsv, (5, 60, 70), (15, 255, 255))
        
        dirty_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        
        # Лёгкая морфология: только убираем шум
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dirty_mask = cv2.morphologyEx(dirty_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Детекция горизонтальных полос (жёлтые линии)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horiz_lines = cv2.morphologyEx(zone_hsv[:, :, 2], cv2.MORPH_TOPHAT, kernel_horiz)
        horiz_mask = cv2.threshold(horiz_lines, 30, 255, cv2.THRESH_BINARY)[1]
        dirty_mask = cv2.bitwise_or(dirty_mask, horiz_mask)
        
        # Минимальное расширение только горизонтально (для линий)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        dirty_mask = cv2.dilate(dirty_mask, kernel_dilate, iterations=2)
        
        mask_coverage = np.sum(dirty_mask > 0) / dirty_mask.size * 100
        
    else:
        # Fallback без OpenCV
        dirty_mask = np.zeros((zone_h, w), dtype=np.uint8)
        for y in range(zone_h):
            for x in range(w):
                r, g, b = clean_zone[y, x]
                is_yellow = (r > 140 and g > 90 and b < 95 and r - b > 45)
                is_orange = (r > 95 and g > 65 and b < 85 and r > g > b)
                is_beige = (r > 110 and g > 95 and b > 70 and r - b > 30 and r - b < 80)
                if is_yellow or is_orange or is_beige:
                    dirty_mask[y, x] = 255
        
        mask_pil = Image.fromarray(dirty_mask)
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(4))
        dirty_mask = np.array(mask_pil)
        mask_coverage = np.sum(dirty_mask > 0) / dirty_mask.size * 100
    
    print(f"[Clean Zone] {clean_zone_start}-{clean_zone_end}px (above logo at ~{logo_y_estimate}px)")
    print(f"[Mask] Dirty pixels: {mask_coverage:.2f}%")
    
    # ШАГ 2: Получение цвета заливки (из чистой зоны чуть выше)
    sample_start = max(0, clean_zone_start - 50)
    sample_end = max(10, clean_zone_start - 10)
    
    if sample_end > sample_start and sample_end <= h:
        sample_zone = img_np[sample_start:sample_end, :, :]
        sample_flat = sample_zone.reshape(-1, 3)
        # Фильтруем жёлтые пиксели
        non_yellow = sample_flat[~((sample_flat[:, 0] > sample_flat[:, 1]) & 
                                    (sample_flat[:, 1] > sample_flat[:, 2]) & 
                                    (sample_flat[:, 0] - sample_flat[:, 2] > 35))]
        if len(non_yellow) > 100:
            fill_color = np.median(non_yellow, axis=0).astype(np.uint8)
        else:
            fill_color = np.median(sample_flat, axis=0).astype(np.uint8)
    else:
        fill_color = np.array([30, 32, 38], dtype=np.uint8)
    
    print(f"[Fill color] RGB: {tuple(fill_color)}")
    
    # ШАГ 3: Очистка (только inpaint, БЕЗ градиента)
    result_zone = clean_zone.copy()
    
    if CV2_AVAILABLE and np.sum(dirty_mask > 0) > 0:
        # Inpaint с умеренным радиусом — только замазываем артефакты
        result_zone_bgr = cv2.cvtColor(result_zone, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(result_zone_bgr, dirty_mask, 5, cv2.INPAINT_TELEA)
        result_zone = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        print(f"✓ Inpaint applied (radius=5)")
    else:
        # Заливка по маске
        mask_bool = dirty_mask > 128
        result_zone[mask_bool] = fill_color
        print("✓ Fill applied")
    
    # ШАГ 4: Перо на границах (только верх и низ, БЕЗ градиента!)
    feather = 12
    
    # Верхнее перо (если есть место)
    if clean_zone_start > 0:
        for y in range(min(feather, zone_h)):
            t = y / feather
            blend = t ** 1.5
            if clean_zone_start + y < h:
                orig_row = img_np[clean_zone_start + y]
                result_zone[y] = (
                    orig_row.astype(np.float32) * (1 - blend) + 
                    result_zone[y].astype(np.float32) * blend
                ).astype(np.uint8)
    
    # Нижнее перо
    feather_start = max(0, zone_h - feather)
    for y in range(feather_start, zone_h):
        t = (y - feather_start) / feather
        blend = t ** 1.5
        orig_row = clean_zone[y]
        result_zone[y] = (
            result_zone[y].astype(np.float32) * (1 - blend) + 
            orig_row.astype(np.float32) * blend
        ).astype(np.uint8)
    
    # ШАГ 5: Собираем результат
    result_np = img_np.copy()
    result_np[clean_zone_start:clean_zone_end, :, :] = result_zone
    
    print(f"✓ Top cleaned with inpaint only (no gradient)")
    return Image.fromarray(result_np)

def build_mask_from_boxes(size, boxes):
    """Возвращает бинарную маску (PIL L) по boundingBoxes."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    for b in boxes:
        v = b.get("vertices", [])
        if len(v) < 4: 
            continue
        xs = [vi.get("x", 0) for vi in v]
        ys = [vi.get("y", 0) for vi in v]
        pad_x, pad_y = 12, 14  # Уменьшенный padding для более точной маски
        x1, y1 = max(0, min(xs)-pad_x), max(0, min(ys)-pad_y)
        x2, y2 = min(w, max(xs)+pad_x), min(h, max(ys)+pad_y)
        # Только нижняя половина — убираем оверкилл
        if y1 < h*0.45:
            continue
        d.rectangle([(x1,y1),(x2,y2)], fill=255)
    # МИНИМАЛЬНОЕ размытие только для сглаживания ступенек
    return mask.filter(ImageFilter.GaussianBlur(1))  # было 4 - слишком много

def inpaint_or_soft_cover(img: Image.Image, boxes):
    """
    Удаляет старый текст: размытие области + тёмный градиент для маскировки.
    Стратегия: 1) размываем текст 2) накладываем мягкий градиент поверх.
    """
    if not boxes:
        return img

    w, h = img.size
    mask = build_mask_from_boxes((w,h), boxes)
    
    # Проверяем, есть ли что удалять
    mask_arr = np.array(mask)
    if mask_arr.max() == 0:
        print("✓ No text to remove (mask empty)")
        return img

    # ШАГ 1: Размываем ТОЛЬКО внутри boundingBoxes!
    # Создаём размытую версию изображения
    blurred = img.filter(ImageFilter.GaussianBlur(25))
    
    # МИНИМАЛЬНОЕ размытие маски - только 2px для сглаживания краёв!
    mask_soft = mask.filter(ImageFilter.GaussianBlur(2))  # было 12 - слишком много!
    img_no_text = Image.composite(blurred, img, mask_soft)
    
    print("✓ Text removed with blur only (no gradient artifacts)")
    return img_no_text

def draw_soft_warm_fade(img: Image.Image, percent: float):
    """Мягкий градиент с однотонным черным внизу и плавным переходом вверх."""
    w, h = img.size
    g_h = int(h * percent)
    y0 = h - g_h

    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    # Однотонный черный внизу (40% градиента)
    solid_black_height = int(g_h * 0.40)
    solid_black_start = h - solid_black_height
    d.rectangle([(0, solid_black_start), (w, h)], fill=(0, 0, 0, 255))

    # Плавный градиент в верхней части (60% градиента) - более длинный и мягкий
    gradient_zone_height = g_h - solid_black_height
    steps = max(1, int(gradient_zone_height * 4))  # еще больше шагов для плавности

    for i in range(steps):
        t = i / steps
        # Более плавная cubic функция для мягкого перехода
        alpha = int(255 * (t ** 3))
        y = y0 + int(i * gradient_zone_height / steps)
        d.rectangle([(0, y), (w, y+1)], fill=(0, 0, 0, alpha))

    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    print(f"✓ Smooth gradient: {percent*100:.0f}% height, solid bottom 40%")
    return out, y0, g_h  # вернём позицию начала фейда

# ----------------------------- Text Layout -----------------------------

def wrap_text(text, font, max_width, draw, tracking=-1):
    words, lines, cur = text.split(), [], []
    for w in words:
        probe = " ".join(cur+[w])
        width = 0
        for ch in probe:
            bb = draw.textbbox((0,0), ch, font=font)
            cw = bb[2]-bb[0]
            width += cw if ch==" " else cw + tracking
        if width > max_width and cur:
            lines.append(" ".join(cur)); cur=[w]
        else:
            cur.append(w)
    if cur: lines.append(" ".join(cur))
    return lines

def draw_with_tracking(draw, xy, text, font, color, tracking=-1):
    x, y = xy
    for ch in text:
        bb = draw.textbbox((0,0), ch, font=font)
        cw = bb[2]-bb[0]
        if ch != " ":
            draw.text((x,y), ch, font=font, fill=color)
            x += cw + tracking
        else:
            x += cw

def draw_stretched_line(img, text, font, color, center_x, y, tracking=-1, stretch_factor=1.25):
    """
    Рисует строку текста с вертикальным растяжением (буквы выше).
    stretch_factor=1.15 означает 115% высоты (на 15% выше).
    """
    # Создаём временный draw для измерения
    temp_draw = ImageDraw.Draw(img)
    
    # Измеряем ширину строки с учётом tracking
    line_width = 0
    for ch in text:
        bb = temp_draw.textbbox((0,0), ch, font=font)
        cw = bb[2] - bb[0]
        line_width += cw if ch == " " else cw + tracking
    
    # Измеряем высоту строки
    bb = temp_draw.textbbox((0,0), text, font=font)
    line_height = bb[3] - bb[1]
    
    # Создаём временное RGBA изображение для текста
    padding = 10
    temp_width = line_width + padding * 2
    temp_height = line_height + padding * 2
    temp_img = Image.new("RGBA", (temp_width, temp_height), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Рисуем текст с tracking во временное изображение
    x = padding
    for ch in text:
        bb = temp_draw.textbbox((0,0), ch, font=font)
        cw = bb[2] - bb[0]
        if ch != " ":
            temp_draw.text((x, padding), ch, font=font, fill=color)
            x += cw + tracking
        else:
            x += cw
    
    # Растягиваем по вертикали
    new_height = int(temp_height * stretch_factor)
    stretched = temp_img.resize((temp_width, new_height), Image.Resampling.LANCZOS)
    
    # Вычисляем позицию для вставки (центрируем по x)
    paste_x = center_x - line_width // 2 - padding
    # Корректируем y с учётом увеличенной высоты (чтобы базовая линия осталась примерно там же)
    height_diff = new_height - temp_height
    paste_y = y - padding - height_diff  # сдвигаем вверх на разницу высот
    
    # Вставляем с альфа-каналом
    img.paste(stretched, (paste_x, paste_y), stretched)
    
    return new_height - padding * 2  # возвращаем реальную высоту растянутого текста

def draw_title_subtitle(img, draw, title, subtitle, start_y, width):
    cyan = (0,188,212)
    white = (255,255,255)
    y = start_y
    
    # Коэффициент вертикального растяжения букв (1.15 = на 15% выше)
    stretch_factor = 1.15

    if title:
        t = title.upper()
        has_logo = (subtitle == "")
        size = 72 if has_logo and len(t)<=30 else (60 if has_logo else (44 if len(t)<=30 else 38))
        fnt = get_font(size, "bold")
        tracking = -1
        lines = wrap_text(t, fnt, width*0.88, draw, tracking)
        while len(lines)>3 and size>32:
            size -= 3; fnt = get_font(size, "bold")
            lines = wrap_text(t, fnt, width*0.88, draw, tracking)
        
        # Фиксированный межстрочный интервал независимо от размера шрифта
        line_spacing = 6
        
        # Вычисляем высоту одной строки (используем первую строку как эталон)
        bb = draw.textbbox((0,0), lines[0] if lines else "A", font=fnt)
        line_height = bb[3]-bb[1]
        
        # Высота растянутой строки
        stretched_line_height = int(line_height * stretch_factor)
        
        center_x = width // 2
        for i, line in enumerate(lines):
            # Используем растянутую высоту строки для позиционирования
            line_y = y + i * (stretched_line_height + line_spacing)
            draw_stretched_line(img, line, fnt, cyan, center_x, line_y, tracking, stretch_factor)
        
        # Переходим к следующему элементу
        y += len(lines) * (stretched_line_height + line_spacing) - line_spacing
        if subtitle:
            y += 8  # уменьшенный отступ между заголовком и подзаголовком
        print(f"[Title] Lines:{len(lines)}, Size:{size}px, Mode:{'LOGO' if has_logo else 'NO-LOGO'}, Stretch:{stretch_factor}")

    if subtitle:
        fnt = get_font(32, "medium")
        tracking = -1
        lines = wrap_text(subtitle, fnt, width*0.88, draw, tracking)
        
        # Вычисляем высоту одной строки subtitle
        bb = draw.textbbox((0,0), lines[0] if lines else "A", font=fnt)
        sub_line_height = bb[3]-bb[1]
        stretched_sub_height = int(sub_line_height * stretch_factor)
        sub_line_spacing = 5
        
        center_x = width // 2
        for i, line in enumerate(lines):
            # Используем растянутую высоту строки
            line_y = y + i * (stretched_sub_height + sub_line_spacing)
            draw_stretched_line(img, line, fnt, white, center_x, line_y, tracking, stretch_factor)
        print(f"[Subtitle] Lines:{len(lines)}, Stretch:{stretch_factor}")

# ----------------------------- API -----------------------------

@app.route("/process", methods=["POST"])
def process_image():
    try:
        data = request.json or {}
        img_b64   = data.get("image","")
        title     = data.get("title","")
        subtitle  = data.get("subtitle","")
        boxes     = data.get("boundingBoxes", []) or data.get("boundingboxes", [])
        add_logo  = bool(data.get("addLogo", False))
        
        # ✅ НОВЫЙ РЕЖИМ "last" - только title, без logo и без subtitle
        mode = data.get("mode", "").lower()
        is_last_mode = (mode == "last")

        # Определяем режим работы
        if is_last_mode:
            subtitle = ""  # В режиме last — только заголовок
            add_logo = False
            print("[Processing] LAST MODE: title only, no logo, no subtitle")
        elif add_logo:
            subtitle = ""  # в режиме лого — только заголовок
            print("[Processing] LOGO MODE: subtitle disabled")
        else:
            print("[Processing] NORMAL MODE: title + subtitle")

        if not title and not subtitle:
            title = data.get("text","ЗАГОЛОВОК")

        print(f"[Processing] Title:'{title}', Subtitle:'{subtitle}', Logo:{add_logo}, Mode:{mode or 'normal'}")

        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        w, h = img.size
        print(f"[Image] Size: {w}x{h}")

        # Лёгкий pre-tone
        img = ImageEnhance.Sharpness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(0.98)

        # 1) Удаляем старый текст (если есть координаты)
        if boxes:
            img = inpaint_or_soft_cover(img, boxes)
            # ОТКЛЮЧЕНО: clean_top_yellow_artifacts создавала мыльную полосу по середине
            # img = clean_top_yellow_artifacts(img, logo_y_estimate=int(h * 0.52))

        # 2) Мягкий фейд снизу (индивидуально для каждого режима)
        long_text = (len(title)>25) or (len(subtitle)>40)
        gp = calculate_adaptive_gradient(img, long_text)
        
        # Настройки градиента для каждого режима
        if add_logo:
            # LOGO режим: градиент высокий - закрываем жёлтые полоски
            gp = 0.92  # ФИКСИРОВАННЫЙ высокий градиент - 92%
        elif is_last_mode:
            # LAST режим: градиент ВЫШЕ (было 0.76)
            gp = 0.82
        else:
            # NORMAL режим: градиент НИЖЕ (было +12%)
            gp = min(gp + 0.05, 0.55)
            
        img, fade_top, fade_h = draw_soft_warm_fade(img, gp)

        d = ImageDraw.Draw(img)

        # 3) Вычисляем высоту текстового блока для центрирования
        def calculate_text_height(title, subtitle, width, draw):
            total_h = 0
            if title:
                t = title.upper()
                has_logo = (subtitle == "")
                size = 72 if has_logo and len(t)<=30 else (60 if has_logo else (48 if len(t)<=30 else 42))
                fnt = get_font(size, "bold")
                lines = wrap_text(t, fnt, width*0.88, draw, -1)
                while len(lines)>3 and size>32:
                    size -= 3
                    fnt = get_font(size, "bold")
                    lines = wrap_text(t, fnt, width*0.88, draw, -1)
                
                # Используем высоту первой строки как эталон для всех
                if lines:
                    bb = draw.textbbox((0,0), lines[0], font=fnt)
                    line_height = bb[3]-bb[1]
                    total_h = len(lines) * line_height + (len(lines)-1) * 6
                
                if subtitle:
                    total_h += 8  # отступ между title и subtitle
            if subtitle:
                fnt = get_font(32, "medium")
                lines = wrap_text(subtitle, fnt, width*0.88, draw, -1)
                if lines:
                    bb = draw.textbbox((0,0), lines[0], font=fnt)
                    sub_line_height = bb[3]-bb[1]
                    total_h += len(lines) * sub_line_height + (len(lines)-1) * 5
            return total_h

        text_height = calculate_text_height(title, subtitle, w, d)
        
        # 4) Логотип и центрирование конструкции
        # ✅ НАСТРАИВАЕМЫЕ СМЕЩЕНИЯ ДЛЯ КАЖДОГО РЕЖИМА
        if add_logo:
            # РЕЖИМ LOGO: логотип + title (текст НИЖЕ)
            logo_text = "@neurostep.media"
            f = get_font(18, "bold")
            bb = d.textbbox((0,0), logo_text, font=f)
            lw, lh = bb[2]-bb[0], bb[3]-bb[1]
            
            # Общая высота конструкции: логотип + отступ + текст
            total_construction_h = lh + 2 + text_height
            
            # ✅ Центрируем + смещаем ВНИЗ на 190px (было 160) - текст ниже
            construction_top = fade_top + (fade_h - total_construction_h) // 2 + 190
            
            # Рисуем логотип
            lx = (w-lw)//2
            ly = construction_top
            d.text((lx+1, ly+1), logo_text, font=f, fill=(0,0,0,150))
            d.text((lx, ly), logo_text, font=f, fill=(255,255,255,255))
            # Бирюзовые линии
            line_y = ly + lh//2
            line_len = 185
            d.rectangle([(lx-8-line_len, line_y), (lx-8, line_y+1)], fill=(0,188,212,255))
            d.rectangle([(lx+lw+8, line_y), (lx+lw+8+line_len, line_y+1)], fill=(0,188,212,255))
            print(f"✓ Logo at ({lx},{ly}), Construction height: {total_construction_h}px")
            start_y = ly + lh + 2
            
        elif is_last_mode:
            # ✅ РЕЖИМ LAST: только title, смещаем ВНИЗ на 160px (было 140) - текст ниже
            start_y = fade_top + (fade_h - text_height) // 2 + 160
            print(f"✓ LAST MODE: title only, offset +160px")
            
        else:
            # РЕЖИМ NORMAL: title + subtitle, смещаем ВНИЗ на 120px (было 140) - текст выше
            start_y = fade_top + (fade_h - text_height) // 2 + 120
            print(f"✓ NORMAL MODE: title + subtitle, offset +120px")

        # Гарантия, что текст не упрётся в край (с большим запасом)
        bottom_guard = h - int(fade_h*0.25)
        start_y = min(start_y, bottom_guard)

        # 5) Текст
        draw_title_subtitle(img, d, title, subtitle, start_y, w)

        # 6) Мини-полоска внизу с внутренней тенью
        bar_h = int(h*0.012)
        bar_y0 = h - bar_h
        
        # Рисуем основную полоску
        d.rectangle([(0, bar_y0), (w, h)], fill=(0,150,170,255))
        
        # Добавляем минимальную внутреннюю тень сверху (1-2px)
        shadow_h = 2
        for i in range(shadow_h):
            t = i / shadow_h
            # Затухание от темного к прозрачному
            shadow_alpha = int(40 * (1 - t))  # минимальная тень
            # Рисуем темную линию
            shadow_y = bar_y0 + i
            if shadow_y < h:
                for x in range(w):
                    r, g, b = (0, 150, 170)
                    # Затемняем цвет
                    dark_r = max(0, r - shadow_alpha)
                    dark_g = max(0, g - shadow_alpha)
                    dark_b = max(0, b - shadow_alpha)
                    d.point((x, shadow_y), fill=(dark_r, dark_g, dark_b))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        print("✓ Image processed successfully")
        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}, 500

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "version": "NEUROSTEP_v13.0_HARD_CLEAN",
        "opencv_available": CV2_AVAILABLE,
        "features": [
            "✅ NEW: Hard clean top zone (0-30%) with inpaint + morphology",
            "✅ NEW: Auto-detect cyan baseline for precise zone targeting",
            "✅ NEW: Horizontal line detection with top-hat morphology",
            "✅ NEW: Feathered gradient transition (28px)",
            "'last' mode - title only without logo",
            "Three modes: logo, normal (title+subtitle), last (title only)",
            "Text removal: blur + dark gradient overlay",
            "Instagram-style warm gradient (42-48%)",
            "Safety bottom guard; tiny bottom bar (1.2%)",
        ],
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)