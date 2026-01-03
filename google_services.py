import os
import json
import base64
import logging
import io
import time
from typing import List, Dict, Optional

from google.cloud import aiplatform
from google.oauth2 import service_account
from google.api_core.exceptions import ResourceExhausted
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

CATEGORIES = {
    "news": {
        "name": "🔥 Новости",
        "prompt": """Придумай 5 тем про АКТУАЛЬНЫЕ события:
- Релизы (GTA 6, iPhone, фильмы)
- Сделки компаний
- Анонсы технологий
Короткие хуки на русском, макс 8 слов."""
    },
    "compare": {
        "name": "📊 Сравнения",
        "prompt": """Придумай 5 тем для СРАВНЕНИЙ с цифрами:
- MrBeast vs страны по населению
- Доходы актёров/спортсменов
- Компании vs ВВП стран
Короткие хуки на русском, макс 8 слов."""
    },
    "facts": {
        "name": "🧠 Факты",
        "prompt": """Придумай 5 тем "А ты знал?":
- Исторические факты
- Научные открытия
- Необычные законы
Короткие хуки на русском, макс 8 слов."""
    },
    "popculture": {
        "name": "🎬 Кино/Игры",
        "prompt": """Придумай 5 тем про кино/игры/сериалы:
- Эволюция персонажей
- Behind the scenes
- Актёры тогда и сейчас
Короткие хуки на русском, макс 8 слов."""
    },
    "money": {
        "name": "💰 Деньги",
        "prompt": """Придумай 5 тем про богатство:
- Состояния миллиардеров
- Лимитированные авто
- Самые дорогие вещи
Короткие хуки на русском, макс 8 слов."""
    },
    "world": {
        "name": "🌍 Мир",
        "prompt": """Придумай 5 тем про страны:
- Необычные законы
- Тюрьмы разных стран
- Города будущего
Короткие хуки на русском, макс 8 слов."""
    }
}

BASE_IMAGE_STYLE = """Style: Premium magazine cover, editorial design.
Visual elements: forest green accent arrows, forest green circular frames, forest green outlines and highlights.
Composition: Dynamic collage layout, multiple focal points.
Quality: Cinematic lighting, photorealistic, 8K detail, professional photography.
Color accent: Forest green (#228B22) for all graphic elements.
Format: Vertical 3:4 aspect ratio.
IMPORTANT: NO TEXT ON IMAGE."""

COLLAGE_STYLE = """Style: Magazine cover collage combining multiple subjects.
Visual elements: Forest green arrows connecting elements, forest green circular frames, forest green outlines.
Layout: Dynamic composition with overlapping elements.
Quality: Cinematic, photorealistic, premium editorial look.
Color accent: Forest green (#228B22) for all graphic elements.
Format: Vertical 3:4.
IMPORTANT: NO TEXT ON IMAGE."""


class GoogleBrain:
    def __init__(self):
        project_id = os.getenv("GOOGLE_PROJECT_ID", "tough-shard-479214-t2")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        try:
            key_base64 = os.getenv("GOOGLE_KEY_BASE64")
            if key_base64:
                key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
                creds_json = base64.b64decode(key_clean).decode('utf-8')
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                aiplatform.init(project=project_id, location=location, credentials=credentials)
            else:
                aiplatform.init(project=project_id, location=location)
        except Exception as e:
            logger.error(f"Auth Error: {e}")

        try:
            self.text_model = GenerativeModel("gemini-2.0-flash-001")
            self.image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            logger.info("✅ Brain: Gemini 2.0 + Imagen 3")
        except Exception as e:
            logger.error(f"Model Error: {e}")
            self.text_model = None
            self.image_model = None

    def _extract_json(self, text: str) -> List[Dict]:
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start == -1 or end == 0:
                clean = text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean)
            return json.loads(text[start:end])
        except:
            return []

    def _extract_lines(self, text: str) -> List[str]:
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = line.lstrip('0123456789.-•*) ').strip()
            if line and len(line) > 3:
                lines.append(line)
        return lines[:6]

    def generate_topics_by_category(self, category: str) -> List[str]:
        if not self.text_model:
            return ["Ошибка API"]
        
        cat_data = CATEGORIES.get(category, CATEGORIES["facts"])
        prompt = cat_data["prompt"] + "\nОтветь только списком тем, по одной на строку."
        
        try:
            config = GenerationConfig(temperature=0.8)
            response = self.text_model.generate_content(prompt, generation_config=config)
            return self._extract_lines(response.text)[:5]
        except Exception as e:
            logger.error(f"Topics Error: {e}")
            return ["Ошибка генерации тем"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        if not self.text_model:
            return []
        
        if slide_count == 1:
            return [{
                "slide_number": 1,
                "ru_caption": topic,
                "image_prompt": f"Magazine cover collage about: {topic}",
                "is_cover": True
            }]
        
        prompt = f"""Тема: "{topic}" | Слайдов: {slide_count}

СТРУКТУРА:
- Слайд 1: обложка-коллаж
- Слайды 2-{slide_count-1}: контент
- Слайд {slide_count}: финальный коллаж

ТЕКСТ (ru_caption): макс 7 слов, русский, факты с цифрами
КАРТИНКА (image_prompt): описание сцены на английском, БЕЗ стиля

JSON:
[{{"slide_number": 1, "ru_caption": "...", "image_prompt": "...", "is_cover": true}}]"""

        try:
            config = GenerationConfig(temperature=0.7)
            response = self.text_model.generate_content(prompt, generation_config=config)
            plan = self._extract_json(response.text)
            if plan:
                plan[0]["is_cover"] = True
                if len(plan) > 1:
                    plan[-1]["is_cover"] = True
            return plan
        except Exception as e:
            logger.error(f"Plan Error: {e}")
            return []

    def generate_image(self, scene_prompt: str, is_cover: bool = False) -> Optional[bytes]:
        if not self.image_model:
            return None
        
        style = COLLAGE_STYLE if is_cover else BASE_IMAGE_STYLE
        full_prompt = f"{style}\n\nScene: {scene_prompt}"
        
        for attempt in range(2):
            try:
                images = self.image_model.generate_images(
                    prompt=full_prompt,
                    number_of_images=1,
                    aspect_ratio="3:4",
                    add_watermark=False
                )
                if not images:
                    return None
                
                output = io.BytesIO()
                images[0].save(output, format="PNG")
                return output.getvalue()
            
            except ResourceExhausted:
                time.sleep(5)
            except Exception as e:
                logger.error(f"Imagen Error: {e}")
                time.sleep(2)
        return None

    def regenerate_with_feedback(self, original_prompt: str, feedback: str, is_cover: bool = False) -> tuple:
        if not self.text_model:
            return original_prompt, self.generate_image(original_prompt, is_cover)
        
        edit_prompt = f"""Оригинал: "{original_prompt}"
Изменить: "{feedback}"
Напиши НОВОЕ описание сцены на английском (1-2 предложения). Только описание."""

        try:
            response = self.text_model.generate_content(edit_prompt)
            new_scene = response.text.strip()
            return new_scene, self.generate_image(new_scene, is_cover)
        except:
            return original_prompt, self.generate_image(original_prompt, is_cover)

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img.size
            draw = ImageDraw.Draw(pil_img)
            draw.rectangle([(0, int(h * 0.75)), (w, h)], fill=(255, 255, 255))
            output = io.BytesIO()
            pil_img.save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Remove Error: {e}")
            return None
