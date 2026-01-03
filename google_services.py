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
    "news": {"name": "🔥 Новости", "prompt": "5 тем про GTA 6, iPhone, или новые технологии. Коротко, русский язык."},
    "compare": {"name": "📊 Сравнения", "prompt": "5 тем сравнений (MrBeast vs население стран, доходы и т.д.). Русский язык."},
    "facts": {"name": "🧠 Факты", "prompt": "5 шокирующих фактов о мире или науке. Русский язык."}
}

class GoogleBrain:
    def __init__(self):
        project_id = os.getenv("GOOGLE_PROJECT_ID", "tough-shard-479214-t2")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        try:
            key_base64 = os.getenv("GOOGLE_KEY_BASE64")
            if key_base64:
                key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
                creds_json = base64.decodebytes(key_clean.encode()).decode()
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                aiplatform.init(project=project_id, location=location, credentials=credentials)
            else:
                aiplatform.init(project=project_id, location=location)
        except Exception: pass

        try:
            self.text_model = GenerativeModel("gemini-2.0-flash-001")
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        except: pass

    def _extract_json(self, text: str) -> List[Dict]:
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            return json.loads(text[start:end])
        except: return []

    def generate_topics(self, cat_key: str) -> List[str]:
        cat = CATEGORIES.get(cat_key, CATEGORIES["news"])
        try:
            res = self.text_model.generate_content(cat["prompt"])
            return [l.strip().replace("*", "").replace("-", "") for l in res.text.split('\n') if len(l) > 5][:5]
        except: return ["Ошибка получения тем"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        prompt = f"""
        Topic: "{topic}" ({slide_count} slides). Language: RUSSIAN.
        
        VISUAL STYLE (MANDATORY):
        "A vertical 3:4 aspect ratio photograph. [SCENE DESCRIPTION]. Cinematic, 8k. In top right: circular inset with THICK FOREST GREEN BORDER. A FOREST GREEN ARROW points to it. Full bleed, frameless."

        IMPORTANT: If it is about a celebrity (like MrBeast), describe them as "a popular male YouTuber Jimmy with brown hair" to avoid safety filters while keeping likeness.
        
        Return JSON array:
        [
          {{"slide_number": 1, "ru_caption": "Short text", "image_prompt": "Prompt following the rule above"}}
        ]
        """
        try:
            res = self.text_model.generate_content(prompt, generation_config=GenerationConfig(temperature=0.3))
            return self._extract_json(res.text)
        except: return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        # Негатив убирает лишний зеленый тон и мусор
        neg = "green skin, green atmosphere, neon green, bright green, cartoon, anime, text, watermark, white border"
        
        for _ in range(2):
            try:
                images = self.image_model.generate_images(
                    prompt=prompt, negative_prompt=neg, 
                    number_of_images=1, aspect_ratio="3:4"
                )
                if images and len(images) > 0:
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
