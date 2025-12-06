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
    # Смягчаем края маски минимально, чтобы не было «ступенек»
    return mask.filter(ImageFilter.GaussianBlur(4))

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

    # ШАГ 1: Размываем область с текстом для удаления букв
    # Создаём сильно размытую версию изображения
    blurred = img.filter(ImageFilter.GaussianBlur(15))
    
    # Смешиваем оригинал и размытие по маске (убираем текст)
    mask_soft = mask.filter(ImageFilter.GaussianBlur(6))
    img_no_text = Image.composite(blurred, img, mask_soft)
    
    # ШАГ 2: Накладываем тёмный градиент поверх для маскировки артефактов
    img_rgba = img_no_text.convert("RGBA")
    darken_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(darken_layer)
    
    # Для каждого бокса рисуем локальный градиент
    for b in boxes:
        v = b.get("vertices", [])
        if len(v) < 4: 
            continue
        xs = [vi.get("x", 0) for vi in v]
        ys = [vi.get("y", 0) for vi in v]
        
        # Расширяем область на 25px для плавного перехода
        pad = 25
        x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
        x2, y2 = min(w, max(xs)+pad), min(h, max(ys)+pad)
        
        # Пропускаем верхнюю половину изображения
        if y1 < h*0.45:
            continue
        
        box_h = y2 - y1
        # Рисуем вертикальный градиент сверху вниз (от прозрачного к тёмному)
        steps = max(25, box_h // 2)
        for i in range(steps):
            t = i / steps
            # Плавное нарастание прозрачности (кубическая функция для более мягкого перехода)
            alpha = int(140 * (t ** 3))  # максимум 140 (более тёмный для лучшей маскировки)
            y_line = y1 + int(i * box_h / steps)
            if y_line < y2:
                d.rectangle([(x1, y_line), (x2, y_line+2)], fill=(0, 0, 0, alpha))
    
    # Размываем градиент для ультра-мягких краёв
    darken_layer = darken_layer.filter(ImageFilter.GaussianBlur(10))
    
    # Применяем затемнение только в области маски
    darken_masked = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    darken_masked.paste(darken_layer, mask=mask_soft)
    
    # Композитим слои
    result = Image.alpha_composite(img_rgba, darken_masked)
    
    print("✓ Text removed with blur + soft dark gradient overlay")
    return result.convert("RGB")

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

def draw_title_subtitle(img, draw, title, subtitle, start_y, width):
    cyan = (0,188,212)
    white = (255,255,255)
    y = start_y

    if title:
        t = title.upper()
        has_logo = (subtitle == "")
        size = 72 if has_logo and len(t)<=30 else (60 if has_logo else (48 if len(t)<=30 else 42))
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
        
        for i, line in enumerate(lines):
            # ширина с учётом tracking
            lw = 0
            for ch in line:
                bb = draw.textbbox((0,0), ch, font=fnt)
                cw = bb[2]-bb[0]
                lw += cw if ch==" " else cw + tracking
            x = (width - lw)//2
            # Используем фиксированную высоту строки для всех линий
            line_y = y + i * (line_height + line_spacing)
            draw_with_tracking(draw, (x,line_y), line, fnt, cyan, tracking)
        
        # Переходим к следующему элементу
        y += len(lines) * (line_height + line_spacing) - line_spacing
        if subtitle:
            y += 8  # уменьшенный отступ между заголовком и подзаголовком
        print(f"[Title] Lines:{len(lines)}, Size:{size}px, Mode:{'LOGO' if has_logo else 'NO-LOGO'}")

    if subtitle:
        fnt = get_font(32, "medium")
        tracking = -1
        lines = wrap_text(subtitle, fnt, width*0.88, draw, tracking)
        
        # Вычисляем высоту одной строки subtitle
        bb = draw.textbbox((0,0), lines[0] if lines else "A", font=fnt)
        sub_line_height = bb[3]-bb[1]
        sub_line_spacing = 5
        
        for i, line in enumerate(lines):
            lw = 0
            for ch in line:
                bb = draw.textbbox((0,0), ch, font=fnt)
                cw = bb[2]-bb[0]
                lw += cw if ch==" " else cw + tracking
            x = (width - lw)//2
            # Используем фиксированную высоту строки
            line_y = y + i * (sub_line_height + sub_line_spacing)
            draw_with_tracking(draw, (x,line_y), line, fnt, white, tracking)
        print(f"[Subtitle] Lines:{len(lines)}")

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

        # 2) Мягкий фейд снизу (поднимаем выше)
        long_text = (len(title)>25) or (len(subtitle)>40)
        gp = calculate_adaptive_gradient(img, long_text)
        gp = min(gp + 0.10, 0.65)  # поднимаем градиент на 10% выше
        
        # Дополнительные пиксели в зависимости от режима
        extra_pixels = 60 / h
        if not add_logo and not is_last_mode:
            # Режим title + subtitle - поднимаем на 75px
            extra_pixels = 75 / h
        elif is_last_mode:
            # Режим LAST - опускаем ниже (меньше extra_pixels)
            extra_pixels = 40 / h
            
        gp = min(gp + extra_pixels, 0.70)
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
            # РЕЖИМ LOGO: логотип + title
            logo_text = "@neurostep.media"
            f = get_font(18, "bold")
            bb = d.textbbox((0,0), logo_text, font=f)
            lw, lh = bb[2]-bb[0], bb[3]-bb[1]
            
            # Общая высота конструкции: логотип + отступ + текст
            total_construction_h = lh + 2 + text_height
            
            # ✅ Центрируем + смещаем ВНИЗ на 80px (было 60)
            construction_top = fade_top + (fade_h - total_construction_h) // 2 + 80
            
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
            # ✅ РЕЖИМ LAST: только title, смещаем ВНИЗ на 160px
            start_y = fade_top + (fade_h - text_height) // 2 + 160
            print(f"✓ LAST MODE: title only, offset +160px")
            
        else:
            # РЕЖИМ NORMAL: title + subtitle, смещаем ВНИЗ на 140px
            start_y = fade_top + (fade_h - text_height) // 2 + 140
            print(f"✓ NORMAL MODE: title + subtitle, offset +140px")

        # Гарантия, что текст не упрётся в край (с большим запасом)
        bottom_guard = h - int(fade_h*0.25)
        start_y = min(start_y, bottom_guard)

        # 5) Текст
        draw_title_subtitle(img, d, title, subtitle, start_y, w)

        # 6) Мини-полоска внизу (ещё меньше)
        bar_h = int(h*0.012)
        bar_y0 = h - bar_h
        d.rectangle([(0, bar_y0), (w, h)], fill=(0,150,170,255))

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
        "version": "NEUROSTEP_v11.0_LAST_MODE",
        "opencv_available": CV2_AVAILABLE,
        "features": [
            "✅ NEW: 'last' mode - title only without logo",
            "✅ IMPROVED: OpenCV inpaint with radius=2 (cleaner, no blur)",
            "✅ IMPROVED: PIL fallback with blur=4 (sharper)",
            "Three modes: logo, normal (title+subtitle), last (title only)",
            "Configurable vertical offsets per mode",
            "Instagram-style warm gradient (42-48%)",
            "Smoother fade transition with extended gradient zone",
            "Tighter cyan/white typography, tracking -1",
            "Safety bottom guard; tiny bottom bar (1.2%)",
        ],
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)