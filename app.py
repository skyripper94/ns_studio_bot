from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import base64
import os

app = Flask(__name__)

def get_font(size):
    """Загрузка шрифта с кириллицей"""
    font_paths = [
        os.path.join(os.path.dirname(__file__), "fonts", "Montserrat-Bold.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except:
            continue
    
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
        gradient_percent = config.get('gradientPercent', 40) / 100  # Увеличено до 40%
        font_size = config.get('fontSize', 56)
        
        # Декодируем изображение
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        width, height = img.size
        
        # 1. ЭФФЕКТЫ НА ФОТО
        # Увеличиваем резкость
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Резкость x2
        
        # Легкий сепия-эффект (теплый оттенок)
        img_array = img.convert('RGB')
        pixels = img_array.load()
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                # Сепия формула
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                
                # Ограничиваем значения
                tr = min(255, tr)
                tg = min(255, tg)
                tb = min(255, tb)
                
                # Смешиваем с оригиналом (50% сепия)
                pixels[x, y] = (
                    int(r * 0.5 + tr * 0.5),
                    int(g * 0.5 + tg * 0.5),
                    int(b * 0.5 + tb * 0.5)
                )
        
        img = img_array
        
        # 2. ГРАДИЕНТ (нижние 40%)
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        # Черный градиент (быстрое затемнение)
        for y in range(gradient_start, height):
            progress = (y - gradient_start) / gradient_height
            alpha = int(255 * (progress ** 0.4))
            draw_overlay.rectangle(
                [(0, y), (width, y + 1)],
                fill=(0, 0, 0, alpha)
            )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        # 3. ЛОГОТИП "NEUROSTEP" С ПОЛОСКАМИ
        draw = ImageDraw.Draw(img)
        
        # Позиция логотипа (сверху градиента)
        logo_y = gradient_start + 20
        
        # Шрифт для логотипа (меньше)
        logo_font = get_font(24)
        logo_text = "NEUROSTEP"
        
        # Центрируем логотип
        bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
        logo_width = bbox[2] - bbox[0]
        logo_x = (width - logo_width) // 2
        
        # Рисуем полоски слева и справа от логотипа
        line_y = logo_y + 12  # Вертикальная позиция линий
        line_thickness = 2
        line_margin = 15  # Расстояние от текста до линии
        
        # Левая линия
        left_line_end = logo_x - line_margin
        draw.rectangle(
            [(30, line_y), (left_line_end, line_y + line_thickness)],
            fill=(255, 255, 255)
        )
        
        # Правая линия
        right_line_start = logo_x + logo_width + line_margin
        draw.rectangle(
            [(right_line_start, line_y), (width - 30, line_y + line_thickness)],
            fill=(255, 255, 255)
        )
        
        # Логотип (белый текст)
        draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=(255, 255, 255))
        
        # 4. ОСНОВНОЙ ТЕКСТ (компактно)
        # Разбиваем на строки с минимальным расстоянием
        main_font = get_font(font_size)
        words = text.split()
        lines = []
        current_line = []
        
        max_width = int(width * 0.90)
        
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
        
        # МИНИМАЛЬНОЕ расстояние между строками
        line_spacing = int(font_size * 1.1)  # Было 1.2, теперь 1.1
        
        # Позиция текста (ниже логотипа)
        text_start_y = logo_y + 50
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=main_font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            y_pos = text_start_y + i * line_spacing
            
            # Жирная обводка
            outline = 7
            for dx in range(-outline, outline + 1):
                for dy in range(-outline, outline + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            (text_x + dx, y_pos + dy),
                            line,
                            font=main_font,
                            fill=(0, 0, 0)
                        )
            
            # Белый текст
            draw.text((text_x, y_pos), line, font=main_font, fill=(255, 255, 255))
        
        # 5. СТРЕЛКА (в правом нижнем углу)
        arrow_size = 50
        arrow_x = width - arrow_size - 30
        arrow_y = height - arrow_size - 30
        
        # Рисуем стрелку →
        arrow_points = [
            (arrow_x, arrow_y + arrow_size // 2),
            (arrow_x + arrow_size - 15, arrow_y + arrow_size // 2),
        ]
        
        # Линия стрелки
        draw.line(arrow_points, fill=(255, 255, 255), width=4)
        
        # Наконечник стрелки (треугольник)
        tip_points = [
            (arrow_x + arrow_size, arrow_y + arrow_size // 2),
            (arrow_x + arrow_size - 15, arrow_y + arrow_size // 2 - 10),
            (arrow_x + arrow_size - 15, arrow_y + arrow_size // 2 + 10),
        ]
        draw.polygon(tip_points, fill=(255, 255, 255))
        
        # Сохраняем результат
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
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
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        os.path.join(os.path.dirname(__file__), "fonts", "Montserrat-Bold.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"),
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            fonts_available.append(fp)
    
    return {
        'status': 'ok',
        'fonts_available': fonts_available,
        'cyrillic_support': len(fonts_available) > 0,
        'style': 'NEUROSTEP'
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)