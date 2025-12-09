# ИСПРАВЛЕНИЯ:
# 1. Ручное управление градиентом через gradientPercent
# 2. Минимальный шрифт в logo mode = 54px
# 3. Увеличенный stretch_factor для logo mode = 1.3
# 4. Ограничение start_y снизу (текст не уезжает)
# 5. Локальный градиент на зону boundingBoxes (закрывает желтое мыло)

# ИНСТРУКЦИЯ:
# 1. Замени весь app.py этим кодом
# 2. Deploy на Railway
# 3. В n8n обнови Check Logo (check_logo_gradient.js)
# 4. В n8n обнови Extract Headline (extract_headline_v10.1_fixed.js)

from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io, base64, os
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

app = Flask(__name__)

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
                return ImageFont.truetype(p, size)
        except:
            pass
    return ImageFont.load_default()

def calculate_adaptive_gradient(img, long_text=False):
    w, h = img.size
    bottom = img.crop((0, h//2, w, h)).convert("L")
    arr = np.array(bottom, dtype=np.uint8)
    avg = float(arr.mean())

    if avg > 150:      gp = 0.48
    elif avg > 100:    gp = 0.45
    else:              gp = 0.42

    if long_text:
        gp = max(gp, 0.36)

    return gp

def build_mask_from_boxes(size, boxes):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    for b in boxes:
        v = b.get("vertices", [])
        if len(v) < 4: 
            continue
        xs = [vi.get("x", 0) for vi in v]
        ys = [vi.get("y", 0) for vi in v]
        pad_x, pad_y = 12, 14
        x1, y1 = max(0, min(xs)-pad_x), max(0, min(ys)-pad_y)
        x2, y2 = min(w, max(xs)+pad_x), min(h, max(ys)+pad_y)
        if y1 < h*0.45:
            continue
        d.rectangle([(x1,y1),(x2,y2)], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(1))

def inpaint_or_soft_cover(img: Image.Image, boxes):
    if not boxes:
        return img

    w, h = img.size
    mask = build_mask_from_boxes((w,h), boxes)
    
    mask_arr = np.array(mask)
    if mask_arr.max() == 0:
        return img

    blurred = img.filter(ImageFilter.GaussianBlur(25))
    mask_soft = mask.filter(ImageFilter.GaussianBlur(2))
    img_no_text = Image.composite(blurred, img, mask_soft)
    
    print("✓ Text removed")
    return img_no_text

# ✅ НОВОЕ: Локальный градиент на зону boundingBoxes
def apply_local_gradient_on_boxes(img: Image.Image, boxes):
    """Накладывает полупрозрачный градиент на зону где был текст."""
    if not boxes:
        return img
    
    w, h = img.size
    
    # Находим общую зону всех boundingBoxes
    all_y = []
    for b in boxes:
        v = b.get("vertices", [])
        if len(v) >= 4:
            ys = [vi.get("y", 0) for vi in v]
            all_y.extend(ys)
    
    if not all_y:
        return img
    
    # Зона: от min_y - 40px до max_y + 20px
    zone_start = max(0, min(all_y) - 40)
    zone_end = min(h, max(all_y) + 20)
    zone_height = zone_end - zone_start
    
    if zone_height < 10:
        return img
    
    # Создаём градиент overlay
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    
    steps = max(20, zone_height)
    for i in range(steps):
        t = i / steps
        # Кубическое затухание для мягкого края сверху, монолит снизу
        # При t=0 (верх): alpha=180 (более непрозрачный)
        # При t=1 (низ): alpha=0 (прозрачный)
        alpha = int(180 * ((1 - t) ** 3))
        y = zone_start + int(i * zone_height / steps)
        if 0 <= y < h:
            d.rectangle([(0, y), (w, y+2)], fill=(5, 5, 10, alpha))
    
    result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    print(f"✓ Local gradient applied on boxes zone [{zone_start}-{zone_end}]px")
    return result

def draw_soft_warm_fade(img: Image.Image, percent: float, offset_down: int = 0, soft_top: bool = False):
    """
    Рисует градиент снизу.
    offset_down: смещение всего градиента вниз в пикселях.
    soft_top: если True, добавляет мягкое рассеивание сверху градиента.
    """
    w, h = img.size
    g_h = int(h * percent)
    y0 = h - g_h + offset_down

    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    solid_black_height = int(g_h * 0.40)
    solid_black_start = h - solid_black_height + offset_down
    d.rectangle([(0, solid_black_start), (w, h)], fill=(0, 0, 0, 255))

    gradient_zone_height = g_h - solid_black_height
    steps = max(1, int(gradient_zone_height * 4))

    for i in range(steps):
        t = i / steps
        
        if soft_top:
            # Мягкое рассеивание: в начале (t близко к 0) ещё больше затухание
            # Используем двойную степень для очень мягкого края
            alpha = int(255 * (t ** 4))
        else:
            alpha = int(255 * (t ** 3))
            
        y = y0 + int(i * gradient_zone_height / steps)
        if y < h:
            d.rectangle([(0, y), (w, y+1)], fill=(0, 0, 0, alpha))

    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return out, y0, g_h

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

def draw_text_with_stroke(draw, text, font, x, y, fill_color, stroke_width=3, stroke_color=(0,0,0,180), tracking=-2):
    """
    Рисует текст с обводкой и плотным tracking.
    """
    # Рисуем обводку (stroke)
    for offset_x in range(-stroke_width, stroke_width + 1):
        for offset_y in range(-stroke_width, stroke_width + 1):
            if offset_x == 0 and offset_y == 0:
                continue
            # Рисуем каждый символ с tracking
            char_x = x
            for ch in text:
                if ch != " ":
                    draw.text((char_x + offset_x, y + offset_y), ch, font=font, fill=stroke_color)
                bb = draw.textbbox((0,0), ch, font=font)
                cw = bb[2] - bb[0]
                char_x += cw if ch == " " else cw + tracking
    
    # Рисуем основной текст
    char_x = x
    for ch in text:
        if ch != " ":
            draw.text((char_x, y), ch, font=font, fill=fill_color)
        bb = draw.textbbox((0,0), ch, font=font)
        cw = bb[2] - bb[0]
        char_x += cw if ch == " " else cw + tracking

def draw_title_subtitle(img, draw, title, subtitle, start_y, width, has_logo=False):
    cyan = (0,188,212)
    white = (255,255,255)
    y = start_y
    
    # ✅ ИСПРАВЛЕНО: Stretch factor увеличен для logo mode
    stretch_factor = 1.3 if has_logo else 1.15

    if title:
        t = title.upper()
        has_logo_mode = (subtitle == "")
        
        # Начальный размер шрифта
        size = 68 if len(t)<=30 else 56
        
        fnt = get_font(size, "bold")
        tracking = -2  # Плотный tracking
        max_width = int(width * 0.85)  # 85% ширины изображения
        lines = wrap_text(t, fnt, max_width, draw, tracking)
        
        # Уменьшаем шрифт если не влезает
        while len(lines) > 3 and size > 36:
            size -= 4
            fnt = get_font(size, "bold")
            lines = wrap_text(t, fnt, max_width, draw, tracking)
        
        # Фиксированный межстрочный интервал
        line_spacing = 4
        
        # Вычисляем высоту строки
        bb = draw.textbbox((0,0), lines[0] if lines else "A", font=fnt)
        line_height = bb[3] - bb[1]
        
        # Рисуем каждую строку с обводкой
        for i, line in enumerate(lines):
            # Вычисляем ширину строки с tracking
            line_width = 0
            for ch in line:
                bb = draw.textbbox((0,0), ch, font=fnt)
                cw = bb[2] - bb[0]
                line_width += cw if ch == " " else cw + tracking
            
            # Центрируем
            x = (width - line_width) // 2
            line_y = y + i * (line_height + line_spacing)
            
            # Рисуем с обводкой
            draw_text_with_stroke(draw, line, fnt, x, line_y, cyan, stroke_width=3, tracking=tracking)
        
        y += len(lines) * (line_height + line_spacing)
        if subtitle:
            y += 10
        print(f"[Title] Lines:{len(lines)}, Size:{size}px, LineHeight:{line_height}px")

    if subtitle:
        fnt = get_font(28, "medium")
        tracking = -1
        max_width = int(width * 0.85)
        lines = wrap_text(subtitle, fnt, max_width, draw, tracking)
        
        bb = draw.textbbox((0,0), lines[0] if lines else "A", font=fnt)
        sub_line_height = bb[3] - bb[1]
        sub_line_spacing = 3
        
        for i, line in enumerate(lines):
            line_width = 0
            for ch in line:
                bb = draw.textbbox((0,0), ch, font=fnt)
                cw = bb[2] - bb[0]
                line_width += cw if ch == " " else cw + tracking
            
            x = (width - line_width) // 2
            line_y = y + i * (sub_line_height + sub_line_spacing)
            
            draw_text_with_stroke(draw, line, fnt, x, line_y, white, stroke_width=2, tracking=tracking)
        
        print(f"[Subtitle] Lines:{len(lines)}")

@app.route("/process", methods=["POST"])
def process_image():
    try:
        data = request.json or {}
        img_b64   = data.get("image","")
        title     = data.get("title","")
        subtitle  = data.get("subtitle","")
        boxes     = data.get("boundingBoxes", []) or data.get("boundingboxes", [])
        add_logo  = bool(data.get("addLogo", False))
        
        # ✅ НОВОЕ: Ручное управление градиентом
        manual_gradient = data.get("gradientPercent", None)
        
        mode = data.get("mode", "").lower()
        is_last_mode = (mode == "last")

        if is_last_mode:
            subtitle = ""
            add_logo = False
        elif add_logo:
            subtitle = ""

        if not title and not subtitle:
            title = data.get("text","ЗАГОЛОВОК")

        print(f"[Processing] Title:'{title}', Subtitle:'{subtitle}', Logo:{add_logo}, Mode:{mode or 'normal'}")
        if manual_gradient:
            print(f"[Processing] Manual gradient: {manual_gradient*100:.0f}%")

        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        w, h = img.size

        img = ImageEnhance.Sharpness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(0.98)

        # 1) Удаляем старый текст
        if boxes:
            img = inpaint_or_soft_cover(img, boxes)
            # ✅ НОВОЕ: Локальный градиент на зону boundingBoxes
            img = apply_local_gradient_on_boxes(img, boxes)

        # 2) Мягкий фейд снизу
        long_text = (len(title)>25) or (len(subtitle)>40)
        
        # ✅ ИСПРАВЛЕНО: Используем manual_gradient если указан
        if manual_gradient is not None:
            gp = manual_gradient
            print(f"[Gradient] Using manual: {gp*100:.0f}%")
        else:
            # Auto-calculate
            gp = calculate_adaptive_gradient(img, long_text)
            
            if add_logo:
                gp = 0.95  # Увеличено с 0.92 для опускания градиента
            elif is_last_mode:
                gp = 0.85  # Увеличено с 0.82 для опускания градиента
            else:
                gp = min(gp + 0.05, 0.55)
            
            print(f"[Gradient] Auto-calculated: {gp*100:.0f}%")
        
        # Мягкий верхний край для logo и last mode
        soft_top_edge = add_logo or is_last_mode
        img, fade_top, fade_h = draw_soft_warm_fade(img, gp, 0, soft_top_edge)

        d = ImageDraw.Draw(img)

        # 3) Вычисляем высоту текстового блока
        def calculate_text_height(title, subtitle, width, draw):
            total_h = 0
            if title:
                t = title.upper()
                size = 68 if len(t)<=30 else 56
                
                fnt = get_font(size, "bold")
                max_width = int(width * 0.85)
                lines = wrap_text(t, fnt, max_width, draw, -2)
                
                while len(lines) > 3 and size > 36:
                    size -= 4
                    fnt = get_font(size, "bold")
                    lines = wrap_text(t, fnt, max_width, draw, -2)
                
                if lines:
                    bb = draw.textbbox((0,0), lines[0], font=fnt)
                    line_height = bb[3]-bb[1]
                    total_h = len(lines) * line_height + (len(lines)-1) * 4
                
                if subtitle:
                    total_h += 10
            if subtitle:
                fnt = get_font(28, "medium")
                max_width = int(width * 0.85)
                lines = wrap_text(subtitle, fnt, max_width, draw, -1)
                if lines:
                    bb = draw.textbbox((0,0), lines[0], font=fnt)
                    sub_line_height = bb[3]-bb[1]
                    total_h += len(lines) * sub_line_height + (len(lines)-1) * 3
            return total_h

        text_height = calculate_text_height(title, subtitle, w, d)
        
        # 4) Логотип и центрирование
        if add_logo:
            logo_text = "@neurostep.media"
            f = get_font(18, "bold")
            bb = d.textbbox((0,0), logo_text, font=f)
            lw, lh = bb[2]-bb[0], bb[3]-bb[1]
            
            total_construction_h = lh + 2 + text_height
            
            # Центрируем конструкцию в середине градиента со смещением вниз
            construction_top = fade_top + (fade_h - total_construction_h) // 2 + 200
            
            lx = (w-lw)//2
            ly = construction_top
            d.text((lx+1, ly+1), logo_text, font=f, fill=(0,0,0,150))
            d.text((lx, ly), logo_text, font=f, fill=(255,255,255,255))
            line_y = ly + lh//2
            line_len = 185
            d.rectangle([(lx-8-line_len, line_y), (lx-8, line_y+1)], fill=(0,188,212,255))
            d.rectangle([(lx+lw+8, line_y), (lx+lw+8+line_len, line_y+1)], fill=(0,188,212,255))
            start_y = ly + lh + 2
            
        elif is_last_mode:
            start_y = fade_top + (fade_h - text_height) // 2 + 160
            
        else:
            start_y = fade_top + (fade_h - text_height) // 2 + 20

        # ✅ ИСПРАВЛЕНО: Ограничиваем start_y снизу чтобы текст не уезжал
        bar_h = int(h*0.012)
        max_start_y = h - bar_h - text_height - 30  # минимум 30px запас снизу
        start_y = min(start_y, max_start_y)
        print(f"[Positioning] start_y={start_y}, text_height={text_height}, max_allowed={max_start_y}")

        # 5) Текст
        draw_title_subtitle(img, d, title, subtitle, start_y, w, add_logo)

        # 6) Мини-полоска внизу
        bar_y0 = h - bar_h
        d.rectangle([(0, bar_y0), (w, h)], fill=(0,150,170,255))
        
        shadow_h = 2
        for i in range(shadow_h):
            t = i / shadow_h
            shadow_alpha = int(40 * (1 - t))
            shadow_y = bar_y0 + i
            if shadow_y < h:
                for x in range(w):
                    r, g, b = (0, 150, 170)
                    dark_r = max(0, r - shadow_alpha)
                    dark_g = max(0, g - shadow_alpha)
                    dark_b = max(0, b - shadow_alpha)
                    d.point((x, shadow_y), fill=(dark_r, dark_g, dark_b))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        print("✓ Image processed")
        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}, 500

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "version": "v14.0_FIXED",
        "features": [
            "Manual gradient control via caption",
            "Fixed Extract Headline (logo words filter)",
            "Logo mode: min font 54px, stretch 1.3x",
            "Text positioning limited to prevent overflow",
            "Local gradient on boundingBoxes zone",
        ],
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)