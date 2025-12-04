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
    
    # ✅ УМЕНЬШИЛИ градиент на 5% (было 40-50%, теперь 35-45%)
    if avg_brightness > 150:  # Светлое фото
        gradient_percent = 0.45  # Было 0.50
    elif avg_brightness > 100:  # Среднее
        gradient_percent = 0.40  # Было 0.45
    else:  # Темное
        gradient_percent = 0.35  # Было 0.40
    
    # ✅ Если текст длинный → гарантируем минимум 45% (было 50%)
    if has_long_text:
        gradient_percent = max(gradient_percent, 0.45)
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


def wrap_text(text, font, max_width, draw, tracking=-1):
    """Разбивает текст на строки по ширине с учетом tracking"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        
        # ✅ НОВОЕ: Вычисляем ширину с учетом tracking
        if tracking != 0:
            # Считаем ширину с tracking вручную
            text_width = 0
            for char in test_line:
                bbox = draw.textbbox((0, 0), char, font=font)
                char_width = bbox[2] - bbox[0]
                if char == ' ':
                    text_width += char_width
                else:
                    text_width += char_width + tracking
        else:
            # Стандартный расчет
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
    """Рисует текст БЕЗ обводки (для чистого вида)"""
    x, y = pos
    
    # ✅ УБРАЛИ обводку - только основной текст
    draw.text((x, y), text, font=font, fill=color)


def draw_text_with_tracking(draw, pos, text, font, color, tracking=-1):
    """Рисует текст с уменьшенным межбуквенным интервалом (tracking)"""
    x, y = pos
    
    for char in text:
        if char == ' ':
            # Пробел - используем стандартную ширину
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            x += char_width
        else:
            # Буква - рисуем с уменьшенным tracking
            draw.text((x, y), char, font=font, fill=color)
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            x += char_width + tracking  # Уменьшенный интервал
    
    return x  # Возвращаем финальную X позицию


def draw_title_subtitle(img, draw, title, subtitle, start_y, width):
    """Рисует заголовок и подзаголовок с правильной иерархией и переносом строк"""
    
    # Цвета
    cyan = (0, 188, 212)  # Бирюзовый
    white = (255, 255, 255)
    
    # ✅ НОВОЕ: Используем переданный start_y
    current_y = start_y
    
    # ═══════════════════════════════════════════════════
    # TITLE (главное)
    # ═══════════════════════════════════════════════════
    if title:
        title = title.upper()
        title_color = cyan  # ✅ ВСЕГДА cyan
        
        # ✅ Определяем режим по наличию subtitle
        has_logo = (subtitle == '')  # Если subtitle пустой, значит режим с лого
        
        # ✅ НОВОЕ: Увеличен шрифт для режима с лого
        if has_logo:
            # С ЛОГО - крупный шрифт
            if len(title) > 30:
                title_size = 60  # Было 54, увеличили до 60
            else:
                title_size = 72  # Было 64, увеличили до 72 (+8px)
        else:
            # БЕЗ ЛОГО - стандартный
            if len(title) > 30:
                title_size = 42  # Без изменений
            else:
                title_size = 48  # Без изменений
        
        title_font = get_font(title_size, weight='bold')
        
        # ✅ Разбиваем title на строки с tracking
        max_width = width * 0.88
        tracking = -1  # Уменьшенный межбуквенный интервал
        title_lines = wrap_text(title, title_font, max_width, draw, tracking=tracking)
        
        # ✅ Если >3 строк → уменьшаем агрессивнее
        while len(title_lines) > 3 and title_size > 32:
            title_size -= 3
            title_font = get_font(title_size, weight='bold')
            title_lines = wrap_text(title, title_font, max_width, draw, tracking=tracking)
        
        # Рисуем каждую строку с tracking
        for line in title_lines:
            # Вычисляем ширину с tracking для центрирования
            line_width = 0
            for char in line:
                bbox = draw.textbbox((0, 0), char, font=title_font)
                char_width = bbox[2] - bbox[0]
                if char == ' ':
                    line_width += char_width
                else:
                    line_width += char_width + tracking
            
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_height = bbox[3] - bbox[1]
            line_x = (width - line_width) // 2
            
            # ✅ Рисуем с tracking
            draw_text_with_tracking(draw, (line_x, current_y), line, title_font, title_color, tracking=tracking)
            current_y += line_height + 5  # ✅ Межстрочный 5px
        
        # ✅ Если есть subtitle, добавляем gap 15px
        if subtitle:
            current_y += 15
        
        print(f"[Title] Text: '{title}', Lines: {len(title_lines)}, Size: {title_size}px, Mode: {'LOGO' if has_logo else 'NO-LOGO'}")
    
    # ═══════════════════════════════════════════════════
    # SUBTITLE (детали)
    # ═══════════════════════════════════════════════════
    if subtitle:
        subtitle_color = white  # ✅ ВСЕГДА white
        subtitle_font = get_font(32, weight='medium')
        tracking = -1  # ✅ Уменьшенный интервал
        
        # Разбиваем на строки с tracking
        subtitle_lines = wrap_text(subtitle, subtitle_font, width * 0.88, draw, tracking=tracking)
        
        # Рисуем каждую строку с tracking
        for line in subtitle_lines:
            # Вычисляем ширину с tracking
            line_width = 0
            for char in line:
                bbox = draw.textbbox((0, 0), char, font=subtitle_font)
                char_width = bbox[2] - bbox[0]
                if char == ' ':
                    line_width += char_width
                else:
                    line_width += char_width + tracking
            
            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
            line_height = bbox[3] - bbox[1]
            line_x = (width - line_width) // 2
            
            # ✅ Рисуем с tracking
            draw_text_with_tracking(draw, (line_x, current_y), line, subtitle_font, subtitle_color, tracking=tracking)
            current_y += line_height + 5  # ✅ УМЕНЬШИЛИ межстрочный (было 10, теперь 5)
        
        print(f"[Subtitle] Text: '{subtitle}', Lines: {len(subtitle_lines)}")


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
        
        # ✅ НОВОЕ: С логотипом - игнорируем subtitle (только title)
        if add_logo:
            subtitle = ''
            print("[Processing] Logo mode: subtitle disabled, showing only title")
        
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
        img = sharpness.enhance(1.1)
        
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.1)
        
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(0.95)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 2: АДАПТИВНЫЙ ГРАДИЕНТ
        # ═══════════════════════════════════════════════════
        # ✅ ОБНОВЛЕНО: Определяем длинный текст по title И subtitle
        has_long_text = len(title) > 25 or len(subtitle) > 40
        gradient_percent = calculate_adaptive_gradient(img, has_long_text=has_long_text)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 3: "УДАЛЕНИЕ" СТАРОГО ТЕКСТА
        # ═══════════════════════════════════════════════════
        if bounding_boxes:
            img = remove_old_text(img, bounding_boxes)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 4: НАЛОЖЕНИЕ ГРАДИЕНТА (ПЛАВНЫЙ)
        # ═══════════════════════════════════════════════════
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        gradient_height = int(height * gradient_percent)
        gradient_start = height - gradient_height
        
        # ✅ НОВОЕ: 80% на плавный переход
        fade_portion = 0.8
        fade_height = int(gradient_height * fade_portion)
        solid_black_start = gradient_start + fade_height
        
        # Сплошной черный (нижние 60% от градиента)
        draw_overlay.rectangle(
            [(0, solid_black_start), (width, height)],
            fill=(0, 0, 0, 255)
        )
        
        # ✅ НОВОЕ: Плавный градиент с большим количеством шагов
        # Рисуем по 0.5px вместо 1px для устранения артефактов
        steps = fade_height * 2  # В 2 раза больше шагов
        
        for i in range(steps):
            progress = i / steps
            
            # ✅ Cubic ease-in-out для максимально плавного перехода
            if progress < 0.5:
                alpha_progress = 4 * progress ** 3
            else:
                alpha_progress = 1 - pow(-2 * progress + 2, 3) / 2
            
            alpha = int(255 * alpha_progress)
            y_pos = gradient_start + int(i * fade_height / steps)
            
            draw_overlay.rectangle(
                [(0, y_pos), (width, y_pos + 1)],
                fill=(0, 0, 0, alpha)
            )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        print(f"✓ Smooth gradient: {gradient_percent*100:.0f}% height (fade: {fade_height}px [{steps} steps], solid: {gradient_height-fade_height}px)")
        
        draw = ImageDraw.Draw(img)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 5: ЛОГОТИП (если нужен)
        # ═══════════════════════════════════════════════════
        # ✅ НОВОЕ: Поднимаем все конструкции вверх (-40px от v8.7)
        if has_long_text:
            start_y = gradient_start + 100  # Было 180, теперь 140 (-40px)
        else:
            start_y = gradient_start + 150  # Было 230, теперь 190 (-40px)
        
        if add_logo:
            logo_text = "@neurostep.media"
            logo_font = get_font(18, weight='bold')

            logo_bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
            logo_width = logo_bbox[2] - logo_bbox[0]
            logo_height = logo_bbox[3] - logo_bbox[1]

            logo_x = (width - logo_width) // 2
            # ✅ Логотип поднят на -40px (было +190, теперь +150)
            logo_y = max(0, gradient_start + 130)

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
            
            # ✅ Title начинается на 1px от логотипа
            start_y = logo_y + logo_height + 1
        
        # ═══════════════════════════════════════════════════
        # ШАГ 6: РИСУЕМ TITLE И SUBTITLE
        # ═══════════════════════════════════════════════════
        draw_title_subtitle(img, draw, title, subtitle, start_y, width)
        
        # ═══════════════════════════════════════════════════
        # ШАГ 7: БИРЮЗОВАЯ ПОЛОСКА ВНИЗУ
        # ═══════════════════════════════════════════════════
        bar_height = int(height * 0.02)  # ✅ УМЕНЬШИЛИ: было 0.04, теперь 0.02 (в 2 раза короче)
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
        'version': 'NEUROSTEP_v8.8_FINAL',
        'features': [
            'Logo mode: fullText as title, LARGE font (72px)',
            'No-logo mode: title+subtitle, standard font (48px)',
            'Adaptive gradient (35-45%, reduced from 40-50%)',
            'Smooth gradient: 2x steps, cubic easing (no artifacts)',
            'Gradient fade: 40%',
            'Old text removal',
            'Gotham Bold/Medium fonts',
            'Title: Cyan, Subtitle: White',
            'Letter spacing: -1px',
            'Line spacing: 5px',
            'Gap between title/subtitle: 15px',
            'Logo to title: 1px gap',
            'All lifted up: logo +150px, no-logo +140/190px',
            'Multi-line text wrapping',
            'No text outline',
            'Bottom bar: 2% height',
        ]
    }
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)