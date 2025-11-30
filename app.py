from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import base64
import os

app = Flask(__name__)

def get_font(size, bold=True):
    if bold:
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # Можно попробовать Black/ExtraBold если есть
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",  
            os.path.join(os.path.dirname(__file__), "fonts", "LiberationSans-Bold.ttf"),
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                print(f"✓ Font loaded: {font_path} (size: {size})")
                return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"✗ Failed to load {font_path}: {e}")
            continue
    
    print(f"⚠️ WARNING: Using default font (size: {size})")
    return ImageFont.load_default()

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        
        # Получаем данные
        image_base64 = data.get('image', '')
        text = data.get('text', 'ЗАГОЛОВОК')
        config = data.get('config', {})
        
        # Параметры ИЗ CONFIG (не жестко прописанные!)
        gradient_percent = config.get('gradientPercent', 45) / 100
        font_size = config.get('fontSize', 40)  # Значение по умолчанию, но берется из config
        
        print(f"Processing: {text}")
        print(f"Config received: gradient={gradient_percent*100}%, fontSize={font_size}")
        
        # Декодируем изображение
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        width, height = img.size
        
        print(f"Image size: {width}x{height}")
        
        # ===== 1. УЛУЧШЕНИЕ ФОТО =====
        # Резкость x3
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(3.5)
        
        # Контраст +20%
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.2)
        
        # Яркость -5% (чуть темнее для контраста с текстом)
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(0.95)
        
        # ===== 2. ГРАДИЕНТ (ПЛАВНЫЙ ПЕРЕХОД) =====
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        gradient_height = int(height * gradient_percent)  # 45% от высоты
        gradient_start = height - gradient_height
        
        # 35% полностью черные (вместо 30%)
        solid_black_height = int(height * 0.35)
        solid_black_start = height - solid_black_height
        
        # Рисуем СПЛОШНОЙ черный (нижние 35%)
        draw_overlay.rectangle(
            [(0, solid_black_start), (width, height)],
            fill=(0, 0, 0, 255)
        )
        
        # ПЛАВНЫЙ градиент в зоне 10% (от 55% до 65% высоты)
        gradient_zone_start = gradient_start
        gradient_zone_height = solid_black_start - gradient_start
        
        for y in range(gradient_zone_start, solid_black_start):
            # Плавный переход с использованием кривой
            progress = (y - gradient_zone_start) / gradient_zone_height
            
            # Используем квадратичную функцию для плавности
            alpha = int(255 * (progress ** 2))
            
            draw_overlay.rectangle(
                [(0, y), (width, y + 1)],
                fill=(0, 0, 0, alpha)
            )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        # ===== 3. ЛОГОТИП "NEUROSTEP" (БЕЗ ОБВОДКИ) =====
        logo_font = get_font(20)
        logo_text = "NEUROSTEP"
        
        # Позиция: самый верх изображения
        logo_y = 20
        
        # Центрируем по X
        bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
        logo_width = bbox[2] - bbox[0]
        logo_x = (width - logo_width) // 2
        
        # Легкая тень (только смещение, без обводки)
        shadow_offset = 2
        draw.text(
            (logo_x + shadow_offset, logo_y + shadow_offset),
            logo_text,
            font=logo_font,
            fill=(0, 0, 0, 180)
        )
        
        # Логотип - белый текст
        draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=(255, 255, 255))
        
        # ===== 4. ОСНОВНОЙ ТЕКСТ (С EMOJI) =====
        text = text.upper()
        
        # Добавляем emoji
        if 'ДОЛЛАРОВ' in text or '$' in text or 'ДЕНЬГ' in text:
            import re
            text = re.sub(r'(\d+[\s\d]*)\s*(ДОЛЛАРОВ?)', r'💰 \1 \2', text)
            if '💰' not in text and ('ДЕНЬГ' in text or 'ДОЛЛАРОВ' in text):
                text = text.replace('ДОЛЛАРОВ', '💰 ДОЛЛАРОВ')
                text = text.replace('ДЕНЬГИ', '💰 ДЕНЬГИ')
        
        if 'НАГРА' in text or 'ПОЛУЧИЛ' in text or 'ПРЕМ' in text:
            text = text.replace('НАГРАДУ', '🎁 НАГРАДУ')
            text = text.replace('ПРЕМИЮ', '🎁 ПРЕМИЮ')
            text = text.replace('ПОЛУЧИЛ', 'ПОЛУЧИЛ 🎁')
        
        if 'НАШЁЛ' in text or 'НАШЕЛ' in text:
            text = text.replace('НАШЁЛ', '💼 НАШЁЛ')
            text = text.replace('НАШЕЛ', '💼 НАШЕЛ')
        
        print(f"Text with emoji: {text}")
        
        main_font = get_font(font_size)
        words = text.split()
        lines = []
        current_line = []
        
        max_width = int(width * 0.88)
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=main_font)
            text_width = bbox[2] - bbox[0]
            
            if text_width > max_width:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        print(f"Text lines: {lines}")
        
        # ВЫТЯНУТО ВВЕРХ: межстрочный 0.85x
        line_spacing = int(font_size * 0.85)
        
        text_start_y = gradient_start + 30
        
        # ОБВОДКА для толщины букв
        outline = 1
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=main_font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            y_pos = text_start_y + i * line_spacing
            
            # Черная обводка
            for dx in range(-outline, outline + 1):
                for dy in range(-outline, outline + 1):
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, y_pos + dy), line, font=main_font, fill=(0, 0, 0))
            
            # Белый текст
            draw.text((text_x, y_pos), line, font=main_font, fill=(255, 255, 255))
        
        # ===== 5. СТРЕЛКА → (НИЖЕ, ЧТОБЫ НЕ НАКЛАДЫВАЛАСЬ) =====
        arrow_size = 100
        arrow_margin = 25
        arrow_x = width - arrow_size - arrow_margin
        arrow_y = height - 40  # Поднял ближе к низу (было 60)
        
        # Линия стрелки (ТОЛСТАЯ - 8px)
        line_width = 8
        draw.line(
            [(arrow_x, arrow_y), (arrow_x + arrow_size - 25, arrow_y)],
            fill=(255, 255, 255),
            width=line_width
        )
        
        # Наконечник (треугольник)
        tip_size = 24
        tip_points = [
            (arrow_x + arrow_size, arrow_y),
            (arrow_x + arrow_size - tip_size, arrow_y - tip_size // 2),
            (arrow_x + arrow_size - tip_size, arrow_y + tip_size // 2),
        ]
        draw.polygon(tip_points, fill=(255, 255, 255))
        
        # ===== СОХРАНЕНИЕ =====
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        print("✓ Image processed successfully")
        return send_file(output, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    fonts_available = []
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"),
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            fonts_available.append(fp)
    
    return {
        'status': 'ok',
        'version': 'NEUROSTEP_v2',
        'features': [
            'UPPERCASE text',
            'Shadow instead of outline',
            'Solid black gradient',
            'Sharp enhancement x3.5',
            'Compact line spacing',
            'Big arrow',
            'Logo with shadow'
        ],
        'fonts': fonts_available
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)