from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        
        # Получаем данные
        image_base64 = data.get('image', '')
        text = data.get('text', 'Заголовок')
        config = data.get('config', {})
        
        # Параметры по умолчанию
        gradient_percent = config.get('gradientPercent', 35) / 100
        gradient_color = config.get('gradientColor', '#1a1a2e')
        font_size = config.get('fontSize', 48)
        font_color = config.get('fontColor', '#ffffff')
        
        # Декодируем изображение
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        # Создаем градиент (нижние X%)
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        # Конвертируем HEX в RGB
        hex_color = gradient_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Рисуем градиент
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for y in range(gradient_start, height):
            # Вычисляем прозрачность (от 0 до 255)
            progress = (y - gradient_start) / gradient_height
            alpha = int(255 * (progress ** 1.5))  # квадратичная функция
            
            # Рисуем линию с нужной прозрачностью
            color_with_alpha = rgb + (alpha,)
            draw.rectangle([(0, y), (width, y+1)], fill=color_with_alpha)
        
        # Добавляем текст
        try:
            # Попытка загрузить шрифт (если есть)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            # Дефолтный шрифт
            font = ImageFont.load_default()
        
        # Разбиваем текст на строки (макс 40 символов)
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 40:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Рисуем текст по центру
        draw = ImageDraw.Draw(img)
        text_y = gradient_start + (gradient_height // 2) - (len(lines) * font_size // 2)
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            # Тень для читаемости
            shadow_offset = 2
            draw.text((text_x + shadow_offset, text_y + shadow_offset + i * font_size), 
                     line, fill=(0, 0, 0, 180), font=font)
            
            # Основной текст
            hex_font_color = font_color.lstrip('#')
            rgb_font = tuple(int(hex_font_color[i:i+2], 16) for i in (0, 2, 4))
            draw.text((text_x, text_y + i * font_size), line, fill=rgb_font, font=font)
        
        # Конвертируем результат
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return send_file(output, mimetype='image/jpeg')
    
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)