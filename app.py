from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

app = Flask(__name__)

def get_font(size):
    """Загрузка шрифта с поддержкой кириллицы"""
    font_paths = [
        # Путь к вашему шрифту в проекте
        os.path.join(os.path.dirname(__file__), "fonts", "Montserrat-Bold.ttf"),
        # Системные шрифты Railway с кириллицей
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # Fallback на дефолтный шрифт
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
        gradient_color = config.get('gradientColor', '#000000')
        font_size = config.get('fontSize', 64)
        font_color = config.get('fontColor', '#ffffff')
        
        # Декодируем изображение
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        width, height = img.size
        
        # Создаем градиент (нижние X%)
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        # Конвертируем HEX в RGB
        hex_color = gradient_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Рисуем градиент как на примере:
        # Сверху прозрачный → постепенно → внизу полностью черный
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for y in range(gradient_start, height):
            # Прогресс от 0 (вверху градиента) до 1 (внизу)
            progress = (y - gradient_start) / gradient_height
            
            # Плавный переход прозрачности
            # progress^0.5 = более быстрое затемнение сверху
            alpha = int(255 * (progress ** 0.5))
            
            # Рисуем линию
            color_with_alpha = rgb + (alpha,)
            draw.rectangle([(0, y), (width, y+1)], fill=color_with_alpha)
        
        # Загружаем шрифт с кириллицей
        font = get_font(font_size)
        
        # Разбиваем текст на строки (макс 35 символов)
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Проверяем ширину строки
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width > width * 0.9:  # 90% от ширины изображения
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Рисуем текст по центру нижней области
        draw = ImageDraw.Draw(img)
        line_spacing = font_size + 15
        total_text_height = len(lines) * line_spacing
        text_y = gradient_start + (gradient_height - total_text_height) // 2
        
        # Конвертируем цвет текста
        hex_font_color = font_color.lstrip('#')
        rgb_font = tuple(int(hex_font_color[i:i+2], 16) for i in (0, 2, 4))
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            # Жирная обводка для контраста (6 пикселей)
            outline_width = 6
            for offset_x in range(-outline_width, outline_width + 1):
                for offset_y in range(-outline_width, outline_width + 1):
                    if offset_x != 0 or offset_y != 0:
                        draw.text(
                            (text_x + offset_x, text_y + offset_y + i * line_spacing),
                            line, 
                            fill=(0, 0, 0, 255), 
                            font=font
                        )
            
            # Основной белый текст
            draw.text(
                (text_x, text_y + i * line_spacing), 
                line, 
                fill=rgb_font, 
                font=font
            )
        
        # Конвертируем результат в JPEG
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return send_file(output, mimetype='image/jpeg')
    
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok', 'font_available': True}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)