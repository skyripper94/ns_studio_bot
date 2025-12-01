from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import base64
import os
import random
import re

app = Flask(__name__)

def get_font(size, bold=True):
    """Загрузка шрифта с приоритетом Schist Black (основной текст)"""
    if bold:
        font_paths = [
            # Rubik variable font (from Google Fonts)
            os.path.join(os.path.dirname(__file__), "fonts", "Rubik[wght].ttf"),
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
            # Rubik variable font (from Google Fonts)
            os.path.join(os.path.dirname(__file__), "fonts", "Rubik[wght].ttf"),
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
        
        # ===== 3. ЛОГОТИП ВВЕРХУ =====
        logo_text = "NEUROSTEP"
        logo_font_size = 20
        logo_font = get_font(logo_font_size, bold=True)
        
        logo_bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
        logo_width = logo_bbox[2] - logo_bbox[0]
        logo_height = logo_bbox[3] - logo_bbox[1]
        
        # Центрируем по горизонтали, отступ 20px сверху
        logo_x = (width - logo_width) // 2
        logo_y = 20
        
        # Тень логотипа (чёрный текст со смещением)
        shadow_offset = 2
        draw.text((logo_x + shadow_offset, logo_y + shadow_offset), logo_text, font=logo_font, fill=(0, 0, 0, 150))
        
        # Рисуем логотип белым поверх тени
        draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=(255, 255, 255, 255))
        
        print(f"✓ Logo rendered: {logo_text} at ({logo_x}, {logo_y}) with shadow")
        
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
        line_spacing = int(font_size * 1.10)
        
        # Начало текста
        text_start_y = gradient_start + 60
        
        # Выбираем случайные слова для бирюзового цвета (1-2 слова)
        # Правило: окрашиваем ТОЛЬКО слова, которые:
        #  - содержат буквы (кириллица/латиница),
        #  - не содержат цифр,
        #  - после удаления знаков препинания длина > 3 символов.
        all_words_in_lines = []
        for line in lines:
            all_words_in_lines.extend(line.split())

        # Фильтруем кандидатов по правилу
        candidate_words = []
        for w in all_words_in_lines:
            # Оставляем только буквы (русские/латинские), убирая пунктуацию и прочие символы
            cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", w)
            if len(cleaned) > 3 and any(ch.isalpha() for ch in cleaned):
                # также исключаем слова, содержащие цифры (уже удалены из cleaned)
                if not any(ch.isdigit() for ch in w):
                    candidate_words.append(w)

        num_cyan_words = random.randint(1, 2) if len(all_words_in_lines) >= 2 else 1
        if candidate_words:
            cyan_words = set(random.sample(candidate_words, min(num_cyan_words, len(candidate_words))))
        else:
            # Если подходящих слов нет — не красим ничего (пустой набор)
            cyan_words = set()

        print(f"[DEBUG] Candidate words for cyan: {candidate_words}")
        print(f"[DEBUG] Cyan words selected: {cyan_words}")
        
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
            
            # Определяем цвет для каждого слова в этой линии
            words_in_line = line.split()
            word_colors = {}
            for word in words_in_line:
                if word in cyan_words:
                    word_colors[word] = (0, 188, 212)  # Бирюзовый (cyan)
                else:
                    word_colors[word] = (255, 255, 255)  # Белый
            
            # Рисуем каждое слово с чёрной обводкой
            current_x = padding
            for word in words_in_line:
                word_bbox = temp_draw.textbbox((current_x, padding), word, font=main_font)
                word_width = word_bbox[2] - word_bbox[0]
                word_color = word_colors[word]
                
                # Чёрная обводка (рисуем текст 8 раз со смещением на 1px)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            temp_draw.text((current_x + dx, padding + dy), word, font=main_font, fill=(0, 0, 0, 200))
                
                # Основной цвет (белый или бирюзовый)
                temp_draw.text((current_x, padding), word, font=main_font, fill=word_color)
                
                current_x += word_width + temp_draw.textbbox((0, 0), " ", font=main_font)[2]
            
            # ╔══════════════════════════════════════════════════╗
            # ║  ВЫТЯГИВАНИЕ БУКВ ПО ВЕРТИКАЛИ (+30%)           ║
            # ╚══════════════════════════════════════════════════╝
            
            original_width = text_width + padding*2
            original_height = text_height + padding*2
            
            # УВЕЛИЧИВАЕМ ВЫСОТУ НА 30%
            stretched_height = int(original_height * 1.30)
            
            print(f"[DEBUG] Stretching: {original_height}px -> {stretched_height}px (+30%)")
            
            stretched = temp_img.resize(
                (original_width, stretched_height), 
                Image.Resampling.LANCZOS
            )
            
            # Накладываем на основное изображение
            img.paste(stretched, (text_x - padding, y_pos - padding), stretched)
            
            print(f"[DEBUG] Line '{line}' rendered with cyan accents and black outline")
        
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
    # also list local fonts directory contents for easier debugging
    local_fonts = []
    fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
    if os.path.isdir(fonts_dir):
        for fn in sorted(os.listdir(fonts_dir)):
            local_fonts.append(fn)

    return {
        'status': 'ok',
        'version': 'NEUROSTEP_v6',
        'features': [
            'NEUROSTEP logo (white, top-centered)',
            'Main text: white with black outline',
            'Random 1-2 words: cyan accent color',
            '30% vertical text stretch',
            'No emoji',
            'Schist Black (main text)',
        ],
        'fonts': fonts_available,
        'local_fonts': local_fonts,
    }
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)