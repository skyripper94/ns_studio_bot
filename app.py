from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import base64
import os

app = Flask(__name__)

def get_font(size, bold=True):
    """Загрузка шрифта с приоритетом Schist Black (основной текст)"""
    if bold:
        font_paths = [
            # Schist Black (локальный файл с пробелом в имени)
            os.path.join(os.path.dirname(__file__), "fonts", "Schist Black.ttf"),
            # Exo2 - сильный локальный fallback
            os.path.join(os.path.dirname(__file__), "fonts", "Exo2-Bold.ttf"),
            # Liberation Sans - системный fallback
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # DejaVu Sans - дополнительный fallback
            os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        font_paths = [
            os.path.join(os.path.dirname(__file__), "fonts", "Schist Black.ttf"),
            os.path.join(os.path.dirname(__file__), "fonts", "Exo2-Bold.ttf"),
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

def get_font_for_logo(size, bold=True):
    """Загрузка шрифта для логотипа - Exo 2 (более плотный для заголовка)"""
    if bold:
        font_paths = [
            # Exo2 - приоритет для логотипа
            os.path.join(os.path.dirname(__file__), "fonts", "Exo2-Bold.ttf"),
            # Fallback
            os.path.join(os.path.dirname(__file__), "fonts", "Montserrat-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        font_paths = [
            os.path.join(os.path.dirname(__file__), "fonts", "Exo2-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                print(f"✓ Logo font loaded: {font_path} (size: {size})")
                return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"✗ Failed to load {font_path}: {e}")
            continue
    
    print(f"⚠️ WARNING: Using default font for logo (size: {size})")
    return ImageFont.load_default()

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        
        # Получаем данные
        image_base64 = data.get('image', '')
        text = data.get('text', 'ЗАГОЛОВОК')
        config = data.get('config', {})
        
        # Параметры ИЗ CONFIG
        gradient_percent = config.get('gradientPercent', 40) / 100
        font_size = config.get('fontSize', 42)
        # Цвет основного текста: можно передать как [r,g,b] в config.textColor
        text_color_cfg = config.get('textColor', None)
        if text_color_cfg and isinstance(text_color_cfg, (list, tuple)) and len(text_color_cfg) == 3:
            text_color = tuple(int(c) for c in text_color_cfg)
        else:
            # default blue (bright)
            text_color = (0, 122, 255)
        
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
        img = sharpness.enhance(3.7)
        
        # Контраст +20%
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.2)
        
        # Яркость -5%
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(0.95)
        
        # ===== 2. ГРАДИЕНТ (ПЛАВНЫЙ ПЕРЕХОД) =====
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        # 35% полностью черные
        solid_black_height = int(height * 0.35)
        solid_black_start = height - solid_black_height
        
        # Рисуем СПЛОШНОЙ черный (нижние 35%)
        draw_overlay.rectangle(
            [(0, solid_black_start), (width, height)],
            fill=(0, 0, 0, 255)
        )
        
        # ПЛАВНЫЙ градиент в зоне 10%
        gradient_zone_start = gradient_start
        gradient_zone_height = solid_black_start - gradient_start
        
        for y in range(gradient_zone_start, solid_black_start):
            progress = (y - gradient_zone_start) / gradient_zone_height
            alpha = int(255 * (progress ** 2))
            
            draw_overlay.rectangle(
                [(0, y), (width, y + 1)],
                fill=(0, 0, 0, alpha)
            )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        # ===== 3. ЛОГОТИП "NEUROSTEP" =====
logo_font = get_font(22)
logo_text = "NEUROSTEP"

logo_y = 20

bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
logo_width = bbox[2] - bbox[0]
logo_x = (width - logo_width) // 2

# Легкая тень
shadow_offset = 2
draw.text(
    (logo_x + shadow_offset, logo_y + shadow_offset),
    logo_text,
    font=logo_font,
    fill=(0, 0, 0, 180)
)

draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=(255, 255, 255))
        
        # ===== 4. ОСНОВНОЙ ТЕКСТ (БЕЗ EMOJI, ВЫТЯНУТЫЕ БУКВЫ) =====
        text = text.upper()
        
        print(f"Text: {text}")
        
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
        
        # Межстрочный интервал
        line_spacing = int(font_size * 0.9)
        
        # Начало текста
        text_start_y = gradient_start + 40
        
        # Рисуем текст с ВЫТЯГИВАНИЕМ
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=main_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (width - text_width) // 2
            
            y_pos = text_start_y + i * line_spacing
            
            print(f"[DEBUG] Line {i}: width={text_width}, height={text_height}")
            
            # Создаем временное изображение для текста
            padding = 40
            temp_img = Image.new('RGBA', (text_width + padding*2, text_height + padding*2), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Черная тень снизу
            temp_draw.text((padding + 3, padding + 3), line, font=main_font, fill=(0, 0, 0, 200))
            
            # Основной текст (можно настроить через config.textColor)
            temp_draw.text((padding, padding), line, font=main_font, fill=text_color)
            
            # ╔══════════════════════════════════════════════════╗
            # ║  ВЫТЯГИВАНИЕ БУКВ ПО ВЕРТИКАЛИ (+15%)           ║
            # ╚══════════════════════════════════════════════════╝
            
            original_width = text_width + padding*2
            original_height = text_height + padding*2
            
            # УВЕЛИЧИВАЕМ ВЫСОТУ НА 15%
            stretched_height = int(original_height * 1.15)
            
            print(f"[DEBUG] Stretching: {original_height}px -> {stretched_height}px (+20%)")
            
            stretched = temp_img.resize(
                (original_width, stretched_height), 
                Image.Resampling.LANCZOS
            )
            
            # Накладываем на основное изображение
            img.paste(stretched, (text_x - padding, y_pos - padding), stretched)
            
            print(f"[DEBUG] Line '{line}' rendered with CYAN glow and 20% stretch")
        
        # (arrow removed)
        
        # Сохраняем
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
        'version': 'NEUROSTEP_v3_CYAN_GLOW',
        'features': [
            'CYAN glow effect',
            '20% vertical text stretch',
            'No emoji',
            'Schist Black (main text)',
            'Exo 2 (logo)',
        ],
        'fonts': fonts_available
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)