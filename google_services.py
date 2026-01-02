import os
import json
import base64
import logging
import io
import re
from typing import List, Dict, Optional

import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

class GoogleBrain:
    def __init__(self):
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        if not project_id:
            raise ValueError("GOOGLE_PROJECT_ID not set")

        key_base64 = os.getenv("GOOGLE_KEY_BASE64")
        if key_base64:
            try:
                key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
                creds_json = base64.b64decode(key_clean).decode('utf-8')
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                vertexai.init(project=project_id, location=location, credentials=credentials)
                logger.info("✅ Auth OK via Service Account")
            except Exception as e:
                logger.error(f"❌ Auth Error: {e}")
                raise
        else:
            vertexai.init(project=project_id, location=location)
            logger.warning("⚠️ No explicit credentials")

        try:
            self.text_model = GenerativeModel("gemini-2.5-flash")
            logger.info("✅ Gemini 2.5 Flash connected")
        except Exception as e:
            logger.error(f"❌ Gemini Error: {e}")
            self.text_model = None

        try:
            self.image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            logger.info("✅ Imagen 3 connected")
        except Exception as e:
            logger.error(f"❌ Imagen Error: {e}")
            self.image_model = None

    def _parse_json(self, text: str):
        text = text.strip()
        json_match = re.search(r'\[[\s\S]*?\]', text)
        if json_match:
            return json.loads(json_match.group())
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text.strip())

    def generate_topics(self) -> List[str]:
        if not self.text_model:
            return ["API Error", "Check billing"]
        
        prompt = """Придумай 5 вирусных тем для Instagram-карусели про технологии и бизнес.
        Ответь ТОЛЬКО JSON массивом строк: ["тема1", "тема2", ...]"""
        try:
            response = self.text_model.generate_content(prompt)
            return self._parse_json(response.text)[:5]
        except Exception as e:
            logger.error(f"Topics Error: {e}")
            return ["ИИ в бизнесе", "Автоматизация", "Нейросети", "Стартапы 2025", "Пассивный доход"]

    def generate_carousel_plan(self, topic: str) -> Optional[List[Dict]]:
        if not self.text_model:
            return None
        
        prompt = f"""Тема: "{topic}"
        Создай план карусели из 5 слайдов.
        Ответь ТОЛЬКО JSON без markdown:
        [
          {{"slide_number": 1, "ru_caption": "Текст подписи на русском", "image_prompt": "English prompt for image generation, vertical 4:5 ratio"}}
        ]"""
        try:
            response = self.text_model.generate_content(prompt)
            return self._parse_json(response.text)
        except Exception as e:
            logger.error(f"Plan Error: {e}")
            return None

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model:
            return None
        try:
            images = self.image_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="4:5",
                add_watermark=False
            )
            output = io.BytesIO()
            images[0].save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Imagen Error: {e}")
            return None

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
            logger.error(f"Edit Error: {e}")
            return None
