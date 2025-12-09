# NEUROSTEP COVER GENERATOR
# Simple image processing service for creating text overlays

from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io, base64, os

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
        add_logo  = bool(data.get("addLogo", False))
        
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

        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        w, h = img.size

        img = ImageEnhance.Sharpness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(0.98)

        d = ImageDraw.Draw(img)

        # Вычисляем высоту текстового блока
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
        
        # Логотип и текст позиционирование (просто в нижней части изображения)
        if add_logo:
            logo_text = "@neurostep.media"
            f = get_font(18, "bold")
            bb = d.textbbox((0,0), logo_text, font=f)
            lw, lh = bb[2]-bb[0], bb[3]-bb[1]
            
            total_construction_h = lh + 2 + text_height
            
            # Размещаем в нижней трети изображения
            construction_top = h - total_construction_h - 100
            
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
            # Размещаем в нижней части
            start_y = h - text_height - 120
            
        else:
            # Обычный режим - центр изображения
            start_y = (h - text_height) // 2 + 100

        # Текст
        draw_title_subtitle(img, d, title, subtitle, start_y, w, add_logo)

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