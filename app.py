from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

app = Flask(__name__)

def get_font(size):
    """Загрузка шрифта с кириллицей"""
    font_paths = [
        # Ваш шрифт в папке fonts/
        os.path.join(os.path.dirname(__file__), "fonts", "Montserrat-Bold.ttf"),
        # Системные шрифты с кириллицей (приоритетные)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                print(f"Using font: {font_path}")
                return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"Failed to load {font_path}: {e}")
            continue
    
    # Если ничего не нашли - используем дефолтный (без кириллицы)
    print("WARNING: Using default font (no cyrillic)")
    return ImageFont.load_default()

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        
        # Получаем данные
        image_base64 = data.get('image', '')
        text = data.get('text', 'Заголовок')
        config = data.get('config', {})
        
        # Параметры
        gradient_percent = config.get('gradientPercent', 35) / 100
        font_size = config.get('fontSize', 64)
        font_color = config.get('fontColor', '#ffffff')
        
        print(f"Processing image with text: {text}")
        
        # Декодируем изображение
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        width, height = img.size
        
        print(f"Image size: {width}x{height}")
        
        # Создаем overlay для градиента
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Высота градиента (нижние 35%)
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        print(f"Gradient from {gradient_start} to {height}")
        
        # КРИТИЧНО: Делаем СПЛОШНОЙ черный градиент
        # Верх градиента - прозрачный (0%)
        # Низ - полностью черный (100%)
        for y in range(gradient_start, height):
            progress = (y - gradient_start) / gradient_height
            # Делаем более агрессивный переход
            # progress^0.3 = быстрое затемнение
            alpha = int(255 * (progress ** 0.3))
            
            # Черный цвет с нарастающей непрозрачностью
            draw_overlay.rectangle(
                [(0, y), (width, y + 1)],
                fill=(0, 0, 0, alpha)
            )
        
        # Накладываем градиент на изображение
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        # Загружаем шрифт с кириллицей
        font = get_font(font_size)
        
        # Разбиваем текст на строки
        draw = ImageDraw.Draw(img)
        words = text.split()
        lines = []
        current_line = []
        
        max_width = int(width * 0.85)  # 85% ширины
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
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
        
        # Рисуем текст
        line_spacing = int(font_size * 1.2)
        total_height = len(lines) * line_spacing
        
        # Центрируем по вертикали в области градиента
        text_y = gradient_start + (gradient_height - total_height) // 2
        
        # Цвет текста
        hex_font_color = font_color.lstrip('#')
        rgb_font = tuple(int(hex_font_color[i:i+2], 16) for i in (0, 2, 4))
        
        for i, line in enumerate(lines):
            # Вычисляем позицию по X (центр)
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            y_pos = text_y + i * line_spacing
            
            # Рисуем ТОЛСТУЮ обводку (8 пикселей)
            outline = 8
            for dx in range(-outline, outline + 1):
                for dy in range(-outline, outline + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            (text_x + dx, y_pos + dy),
                            line,
                            font=font,
                            fill=(0, 0, 0)
                        )
            
            # Основной белый текст
            draw.text(
                (text_x, y_pos),
                line,
                font=font,
                fill=rgb_font
            )
        
        # Сохраняем результат
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        print("Image processed successfully")
        return send_file(output, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    # Проверяем доступные шрифты
    fonts_available = []
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        os.path.join(os.path.dirname(__file__), "fonts", "Montserrat-Bold.ttf"),
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            fonts_available.append(fp)
    
    return {
        'status': 'ok',
        'fonts_available': fonts_available,
        'cyrillic_support': len(fonts_available) > 0
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)