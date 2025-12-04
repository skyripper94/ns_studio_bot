from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import base64
import os

app = Flask(__name__)

def get_font(size, weight='bold'):
    """Загрузка шрифта с приоритетом Gotham"""
    if weight == 'bold':
        font_paths = [
            os.path.join(os.path.dirname(__file__), "fonts", "gotham_bold.otf"),
            os.path.join(os.path.dirname(__file__), "fonts", "Rubik[wght].ttf"),
            os.path.join(os.path.dirname(__file__), "fonts", "Exo2-Bold.ttf"),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:  # medium
        font_paths = [
            os.path.join(os.path.dirname(__file__), "fonts", "gotham_medium.otf"),
            os.path.join(os.path.dirname(__file__), "fonts", "Rubik[wght].ttf"),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
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


def calculate_adaptive_gradient(img, has_long_text=False):
    """Определяет оптимальную высоту градиента на основе яркости"""
    width, height = img.size
    
    # Анализируем нижние 50% изображения
    bottom_half = img.crop((0, height // 2, width, height))
    
    # Конвертируем в grayscale и считаем среднюю яркость
    gray = bottom_half.convert('L')
    pixels = list(gray.getdata())
    avg_brightness = sum(pixels) / len(pixels)
    
    # Определяем высоту градиента
    if avg_brightness > 150:  # Светлое фото
        gradient_percent = 0.45
    elif avg_brightness > 100:  # Среднее
        gradient_percent = 0.35
    else:  # Темное
        gradient_percent = 0.28
    
    # ✅ НОВОЕ: Если текст длинный → увеличиваем градиент
    if has_long_text:
        gradient_percent = max(gradient_percent, 0.40)  # Минимум 40%
        print(f"[Adaptive Gradient] Long text detected, increased gradient")
    
    print(f"[Adaptive Gradient] Brightness: {avg_brightness:.0f}, Gradient: {gradient_percent*100:.0f}%")
    return gradient_percent


def remove_old_text(img, bounding_boxes):
    """Закрашивает старый текст чёрным цветом с увеличенным padding"""
    if not bounding_boxes:
        return img
    
    draw = ImageDraw.Draw(img)
    height = img.size[1]
    removed_count = 0
    
    # Обрабатываем только текст из нижней половины
    for box in bounding_boxes:
        vertices = box.get('vertices', [])
        if len(vertices) < 4:
            continue
        
        # Проверяем, что текст в нижней половине
        min_y = min(v.get('y', 0) for v in vertices)
        if min_y < height * 0.5:
            continue
        
        # Преобразуем vertices в координаты для rectangle
        xs = [v.get('x', 0) for v in vertices]
        ys = [v.get('y', 0) for v in vertices]
        
        # ✅ УВЕЛИЧИЛИ padding для полного закрытия артефактов
        padding_x = 15  # Горизонтальный padding
        padding_y = 20  # Вертикальный padding (больше, чтобы закрыть тени)
        
        draw.rectangle(
            [(min(xs) - padding_x, min(ys) - padding_y), 
             (max(xs) + padding_x, max(ys) + padding_y)],
            fill=(0, 0, 0, 255)
        )
        removed_count += 1
    
    print(f"[Text Removal] Removed {removed_count} text blocks with increased padding")
    return img


def wrap_text(text, font, max_width, draw):
    """Разбивает текст на строки по ширине"""
    words = text.split()
    lines = []
    current_line = []
    
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
    
    return lines


def draw_text_with_outline(draw, pos, text, font, color):
    """Рисует текст с чёрной обводкой"""
    x, y = pos
    
    # Обводка (8 направлений)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 200))
    
    # Основной текст
    draw.text((x, y), text, font=font, fill=color)


def draw_title_subtitle(img, draw, title, subtitle, gradient_start, add_logo, width):
    """Рисует заголовок и подзаголовок с правильной иерархией и переносом строк"""
    
    # Цвета
    cyan = (0, 188, 212)  # Бирюзовый
    white = (255, 255, 255)
    
    # ✅ НОВОЕ: Начинаем выше, чтобы текст не вылезал
    current_y = gradient_start + 60  # Было +80
    
    # ═══════════════════════════════════════════════════
    # TITLE (главное)
    # ═══════════════════════════════════════════════════
    if title:
        title = title.upper()
        title_color = cyan  # ✅ ВСЕГДА cyan
        
        # Динамический размер
        title_size = 56
        title_font = get_font(title_size, weight='bold')
        
        # ✅ НОВОЕ: Разбиваем title на строки если не влезает
        max_width = width * 0.88
        title_lines = wrap_text(title, title_font, max_width, draw)
        
        # Если не влезает в 2 строки → уменьшаем шрифт
        while len(title_lines) > 2 and title_size > 36:
            title_size -= 2
            title_font = get_font(title_size, weight='bold')
            title_lines = wrap_text(title, title_font, max_width, draw)
        
        # Рисуем каждую строку
        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            line_x = (width - line_width) // 2
            
            draw_text_with_outline(draw, (line_x, current_y), line, title_font, title_color)
            current_y += line_height + 10  # Межстрочный интервал
        
        print(f"[Title] Text: '{title}', Lines: {len(title_lines)}, Size: {title_size}px, Color: Cyan")
    
    # ═══════════════════════════════════════════════════
    # SUBTITLE (детали)
    # ═══════════════════════════════════════════════════
    if subtitle:
        subtitle_color = white  # ✅ ВСЕГДА white
        subtitle_font = get_font(32, weight='medium')
        
        # Разбиваем на строки (если длинное)
        subtitle_lines = wrap_text(subtitle, subtitle_font, width * 0.88, draw)
        
        # Рисуем каждую строку
        for line in subtitle_lines:
            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            line_x = (width - line_width) // 2
            
            draw_text_with_outline(draw, (line_x, current_y), line, subtitle_font, subtitle_color)
            current_y += line_height + 10  # Межстрочный интервал
        
        print(f"[Subtitle] Text: '{subtitle}', Lines: {len(subtitle_lines)}, Color: White")


@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        
        # Получаем данные
        image_base64 = data.get('image', '')
        title = data.get('title', '')
        subtitle = data.get('subtitle', '')
        bounding_boxes = data.get('boundingBoxes', [])
        add_logo = data.get('addLogo', False)
        
        # Обратная совместимость со старым API
        if not title and not subtitle:
            text = data.get('text', 'ЗАГОЛОВОК')
            title = text
        
        print(f"[Processing] Title: '{title}', Subtitle: '{subtitle}', Logo: {add_logo}")
        
        # Декодируем изображение
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        width, height = img.size
        
        print(f"[Image] Size: {width}x{height}")
        
        # ═══════════════════════════════════════════════════
        # ШАГ 1: УЛУЧШЕНИЕ ФОТО
        # ═══════════════════════════════════════════════════
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(1.9)
        
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.1)
        
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(0.95)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 2: АДАПТИВНЫЙ ГРАДИЕНТ
        # ═══════════════════════════════════════════════════
        # ✅ НОВОЕ: Определяем, длинный ли текст
        has_long_text = len(title) > 30  # Если >30 символов
        gradient_percent = calculate_adaptive_gradient(img, has_long_text=has_long_text)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 3: "УДАЛЕНИЕ" СТАРОГО ТЕКСТА
        # ═══════════════════════════════════════════════════
        if bounding_boxes:
            img = remove_old_text(img, bounding_boxes)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 4: НАЛОЖЕНИЕ ГРАДИЕНТА
        # ═══════════════════════════════════════════════════
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        # Сплошной черный (нижние 65% от градиента)
        solid_portion = 0.65
        solid_black_height = int(gradient_height * solid_portion)
        solid_black_start = height - solid_black_height
        
        draw_overlay.rectangle(
            [(0, solid_black_start), (width, height)],
            fill=(0, 0, 0, 255)
        )
        
        # Плавный градиент (верхние 35% от градиента)
        gradient_zone_height = solid_black_start - gradient_start
        if gradient_zone_height > 0:
            for y in range(gradient_start, solid_black_start):
                progress = (y - gradient_start) / gradient_zone_height
                alpha = int(255 * (progress ** 2))
                draw_overlay.rectangle(
                    [(0, y), (width, y + 1)],
                    fill=(0, 0, 0, alpha)
                )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 5: ЛОГОТИП (если нужен)
        # ═══════════════════════════════════════════════════
        if add_logo:
            logo_text = "@neurostep.media"
            logo_font = get_font(18, weight='bold')

            logo_bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
            logo_width = logo_bbox[2] - logo_bbox[0]
            logo_height = logo_bbox[3] - logo_bbox[1]

            logo_x = (width - logo_width) // 2
            logo_y = max(0, gradient_start + 10)

            # Тень логотипа
            shadow_offset = 1
            draw.text((logo_x + shadow_offset, logo_y + shadow_offset), logo_text, font=logo_font, fill=(0, 0, 0, 150))

            # Рисуем логотип белым
            draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=(255, 255, 255, 255))

            # Линии от логотипа
            line_y = logo_y + logo_height // 2
            line_thickness = 1
            line_color = (0, 188, 212, 255)
            line_length = 185

            # Левая линия
            left_line_end = logo_x - 8
            left_line_start = left_line_end - line_length
            draw.rectangle(
                [(left_line_start, line_y), (left_line_end, line_y + line_thickness)],
                fill=line_color
            )

            # Правая линия
            right_line_start = logo_x + logo_width + 8
            right_line_end = right_line_start + line_length
            draw.rectangle(
                [(right_line_start, line_y), (right_line_end, line_y + line_thickness)],
                fill=line_color
            )

            print(f"✓ Logo rendered at ({logo_x}, {logo_y})")
        
        # ═══════════════════════════════════════════════════
        # ШАГ 6: РИСУЕМ TITLE И SUBTITLE
        # ═══════════════════════════════════════════════════
        draw_title_subtitle(img, draw, title, subtitle, gradient_start, add_logo, width)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 7: БИРЮЗОВАЯ ПОЛОСКА ВНИЗУ
        # ═══════════════════════════════════════════════════
        bar_height = int(height * 0.04)
        bar_color = (0, 150, 170, 255)
        bar_y_start = height - int(bar_height * 0.5)
        bar_y_end = height + int(bar_height * 0.5)
        
        draw.rectangle(
            [(0, bar_y_start), (width, bar_y_end)],
            fill=bar_color
        )
        
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
    return {
        'status': 'ok',
        'version': 'NEUROSTEP_v8_GOTHAM_FIXED',
        'features': [
            'Title/Subtitle separation',
            'Adaptive gradient (with long text detection)',
            'Old text removal (increased padding)',
            'Gotham Bold/Medium fonts',
            'Title: Cyan, Subtitle: White',
            'Multi-line text wrapping',
        ]
    }
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)