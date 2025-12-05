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
        pad_x, pad_y = 16, 20
        x1, y1 = max(0, min(xs)-pad_x), max(0, min(ys)-pad_y)
        x2, y2 = min(w, max(xs)+pad_x), min(h, max(ys)+pad_y)
        # Только нижняя половина — убираем оверкилл
        if y1 < h*0.45:
            continue
        d.rectangle([(x1,y1),(x2,y2)], fill=255)
    # Смягчаем края маски, чтобы не было «ступенек»
    return mask.filter(ImageFilter.GaussianBlur(6))

def inpaint_or_soft_cover(img: Image.Image, boxes):
    """Удаляет старый текст. С OpenCV — inpaint; без него — мягкое закрытие PIL."""
    if not boxes:
        return img

    w, h = img.size
    mask = build_mask_from_boxes((w,h), boxes)

    if CV2_AVAILABLE:
        np_img = np.array(img.convert("RGB"))
        np_mask = np.array(mask)
        # Алгоритм TELEA даёт более органичный залив
        repaired = cv2.inpaint(np_img, np_mask, 3, cv2.INPAINT_TELEA)
        print("✓ Inpaint (OpenCV)")
        return Image.fromarray(repaired)

    # Fallback: лёгкий «контентный» бленд — блюрим источник и смешиваем по маске
    blurred = img.filter(ImageFilter.GaussianBlur(7))
    # Добавляем легкий шум, чтобы убрать «пластик»
    noise = np.random.randint(-7, 8, (h, w, 3), dtype=np.int16)
    base = np.array(blurred).astype(np.int16)
    soft = np.clip(base + noise, 0, 255).astype(np.uint8)
    soft_img = Image.fromarray(soft)
    out = Image.composite(soft_img, img, mask)
    print("✓ Soft cover (PIL fallback)")
    return out

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
        
        # Фиксированный межстрочный интервал для режима с логотипом (уменьшен в 4 раза от исходного)
        line_spacing = int(size * 0.2875) if has_logo else 5
        
        for line in lines:
            # ширина с учётом tracking
            lw = 0
            for ch in line:
                bb = draw.textbbox((0,0), ch, font=fnt)
                cw = bb[2]-bb[0]
                lw += cw if ch==" " else cw + tracking
            bb = draw.textbbox((0,0), line, font=fnt)
            lh = bb[3]-bb[1]
            x = (width - lw)//2
            draw_with_tracking(draw, (x,y), line, fnt, cyan, tracking)
            y += lh + line_spacing
        if subtitle:
            y += 16  # увеличенный отступ между заголовком и подзаголовком
        print(f"[Title] Lines:{len(lines)}, Size:{size}px, Mode:{'LOGO' if has_logo else 'NO-LOGO'}")

    if subtitle:
        fnt = get_font(32, "medium")
        tracking = -1
        lines = wrap_text(subtitle, fnt, width*0.88, draw, tracking)
        for line in lines:
            lw = 0
            for ch in line:
                bb = draw.textbbox((0,0), ch, font=fnt)
                cw = bb[2]-bb[0]
                lw += cw if ch==" " else cw + tracking
            bb = draw.textbbox((0,0), line, font=fnt)
            lh = bb[3]-bb[1]
            x = (width - lw)//2
            draw_with_tracking(draw, (x,y), line, fnt, white, tracking)
            y += lh + 5
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

        if add_logo:
            subtitle = ""  # в режиме лого — только заголовок
            print("[Processing] Logo mode: subtitle disabled")

        if not title and not subtitle:
            title = data.get("text","ЗАГОЛОВОК")

        print(f"[Processing] Title:'{title}', Subtitle:'{subtitle}', Logo:{add_logo}")

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
        # Дополнительно поднимаем на 60px через увеличение процента
        extra_pixels = 60 / h
        # Для изображений БЕЗ лого поднимаем ещё на 15px
        if not add_logo:
            extra_pixels = (60 + 15) / h
        gp = min(gp + extra_pixels, 0.70)
        img, fade_top, fade_h = draw_soft_warm_fade(img, gp)

        d = ImageDraw.Draw(img)

        # 3) Раскладка: якоримся ВНУТРИ фейда в верхней части зоны.
        #    Текст должен быть ближе к верхней части градиента, не к краю.
        # Для изображений БЕЗ лого опускаем ниже на 80px (90-10)
        center_offset = 80 if not add_logo else 60
        center_in_fade = fade_top + int(fade_h*0.35) + center_offset
        top_in_fade    = fade_top + int(fade_h*0.22)  # ещё выше для лого

        start_y = center_in_fade  # по умолчанию — в верхней части фейда

        # 4) Логотип (если нужен) — опускаем ближе к тексту
        if add_logo:
            logo_text = "@neurostep.media"
            f = get_font(18, "bold")
            bb = d.textbbox((0,0), logo_text, font=f)
            lw, lh = bb[2]-bb[0], bb[3]-bb[1]
            lx = (w-lw)//2
            ly = fade_top + int(fade_h*0.30) + 91  # опускаем на 91px
            # Тень + белый логотип
            d.text((lx+1, ly+1), logo_text, font=f, fill=(0,0,0,150))
            d.text((lx, ly), logo_text, font=f, fill=(255,255,255,255))
            # Бирюзовые линии
            line_y = ly + lh//2
            line_len = 185
            d.rectangle([(lx-8-line_len, line_y), (lx-8, line_y+1)], fill=(0,188,212,255))
            d.rectangle([(lx+lw+8, line_y), (lx+lw+8+line_len, line_y+1)], fill=(0,188,212,255))
            print(f"✓ Logo at ({lx},{ly})")
            start_y = ly + lh + 2  # только 2px отступ между лого и заголовком

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
        "version": "NEUROSTEP_v10.1_ELEVATED_FADE",
        "features": [
            "OpenCV inpaint (optional) + PIL soft cover fallback",
            "Instagram-style warm gradient (brownish-orange undertone, 42-48%)",
            "Smoother fade transition with extended gradient zone",
            "Text positioned in upper fade area for better visibility",
            "Harmonious spacing: logo→title (14px), title→subtitle (16px)",
            "Adaptive fade height based on image brightness and text length",
            "Tighter cyan/white typography, tracking -1, wrap & line gap",
            "Safety bottom guard; tiny bottom bar (1.2%)",
        ],
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
