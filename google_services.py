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
        "prompt": "5 тем про АКТУАЛЬНЫЕ события (GTA 6, iPhone, IT). Коротко на русском."
    },
    "compare": {
        "name": "📊 Сравнения",
        "prompt": "5 тем для сравнений (MrBeast vs страны, доходы и т.д.). Коротко на русском."
    },
    "facts": {
        "name": "🧠 Факты",
        "prompt": "5 необычных фактов 'А ты знал?'. Коротко на русском."
    }
}

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
            # Используем Gemini 2.0 Flash для лучших промптов
            self.text_model = GenerativeModel("gemini-2.0-flash-001")
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Brain Online: Gemini 2.0 (Premium Nano Style)")
        except Exception:
            self.text_model = None
            self.image_model = None

    def _extract_json(self, text: str) -> List[Dict]:
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            return json.loads(text[start:end])
        except: return []

    def generate_topics(self, category_key: str) -> List[str]:
        cat = CATEGORIES.get(category_key, CATEGORIES["news"])
        prompt = cat["prompt"]
        try:
            res = self.text_model.generate_content(prompt)
            return [l.strip().replace("*", "").replace("-", "") for l in res.text.split('\n') if len(l) > 5][:5]
        except: return ["Ошибка генерации тем"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        prompt = f"""
        Тема: "{topic}" ({slide_count} слайдов).
        Задача: План для Instagram. Язык: РУССКИЙ.
        
        ИНСТРУКЦИЯ ПО ВИЗУАЛУ (image_prompt):
        Для каждого слайда пиши промпт на английском строго по формуле:
        "A vertical 3:4 aspect ratio photograph. [PHOTO OF REAL SUBJECTS/SCENE]. Cinematic lighting, photorealistic, 8k. In the top right corner, there is a clean circular inset picture with a THICK FOREST GREEN BORDER showing a close-up of [DETAIL]. A small, styled FOREST GREEN ARROW points from the main scene towards this circular inset. Full bleed image, completely frameless, edge-to-edge."

        ВАЖНО:
        - Если в теме MrBeast — пиши "High-quality photo of real Jimmy Donaldson (MrBeast)".
        - Все графические элементы (стрелка, обводка) должны быть FOREST GREEN.
        
        Верни JSON:
        [
          {{"slide_number": 1, "ru_caption": "Текст...", "image_prompt": "..."}}
        ]
        """
        try:
            res = self.text_model.generate_content(prompt, generation_config=GenerationConfig(temperature=0.3))
            return self._extract_json(res.text)
        except: return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        # Негативный промпт убирает 'зеленку' с лиц и фона
        negative = "green skin, green face, green atmosphere, neon green, lime green, cartoon, anime, illustration, text, watermark, white frame, border"
        
        for attempt in range(2):
            try:
                images = self.image_model.generate_images(
                    prompt=prompt,
                    negative_prompt=negative,
                    number_of_images=1,
                    aspect_ratio="3:4",
                    safety_filter_level="block_some",
                    person_generation="allow_adult"
                )
                if images:
                    buf = io.BytesIO()
                    images[0].save(buf, format="PNG")
                    return buf.getvalue()
            except ResourceExhausted: time.sleep(5)
            except Exception as e:
                logger.error(f"Imagen Error: {e}")
                time.sleep(1)
        return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img.size
            draw = ImageDraw.Draw(pil_img)
            draw.rectangle([(0, int(h * 0.75)), (w, h)], fill=(255, 255, 255))
            out = io.BytesIO()
            pil_img.save(out, format="PNG")
            return out.getvalue()
        except: return None
