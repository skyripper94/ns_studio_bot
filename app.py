# NEUROSTEP COVER GENERATOR
# Simple image processing service

from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io, base64, os

app = Flask(__name__)

def get_font(size):
    paths = [
        os.path.join(os.path.dirname(__file__), "fonts", "gotham_bold.otf"),
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except:
            pass
    return ImageFont.load_default()

@app.route("/health")
def health():
    return {"status":"ok", "version":"clean_v1"}

@app.route("/process", methods=["POST"])
def process_image():
    try:
        data = request.json or {}
        img_b64 = data.get("image", "")
        text = data.get("text", "ТЕКСТ")
        
        print(f"[Processing] Text: '{text}'")
        
        # Открываем изображение
        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        w, h = img.size
        
        # Небольшое улучшение
        img = ImageEnhance.Sharpness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        
        # Рисуем текст
        d = ImageDraw.Draw(img)
        font = get_font(48)
        
        # Центрируем текст
        bbox = d.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x = (w - text_w) // 2
        y = (h - text_h) // 2
        
        # Тень
        d.text((x+2, y+2), text, font=font, fill=(0,0,0,150))
        # Основной текст
        d.text((x, y), text, font=font, fill=(255,255,255,255))
        
        # Отправляем результат
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        print("✓ Image processed")
        return send_file(buf, mimetype="image/jpeg")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
