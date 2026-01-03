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

    def generate_topics(self) -> List[str]:
        if not self.text_model: return ["Ошибка API"]
        
        prompt = "Придумай 6 тем для блога (Tech, Business). Ответь списком строк на русском."
        try:
            config = GenerationConfig(temperature=0.6)
            response = self.text_model.generate_content(prompt, generation_config=config)
            lines = [l.strip().replace("*", "").replace("-", "").strip() for l in response.text.split('\n') if l.strip()]
            return lines[:6]
        except Exception:
            return ["ИИ в Бизнесе", "История Ferrari", "Будущее Энергетики", "Рынок Люкса"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        prompt = f"""
        Тема: "{topic}" ({slide_count} слайдов).
        
        ru_caption: Факты, макс 7 слов, русский.
        image_prompt: Cinematic, Photorealistic, 4:5 Vertical.
        
        JSON:
        [{{"slide_number": 1, "ru_caption": "...", "image_prompt": "..."}}]
        """
        try:
            config = GenerationConfig(temperature=0.6)
            response = self.text_model.generate_content(prompt, generation_config=config)
            return self._extract_json(response.text)
        except Exception:
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        
        for attempt in range(2):
            try:
                images = self.image_model.generate_images(
                    prompt=prompt, 
                    number_of_images=1, 
                    aspect_ratio="4:5",
                    add_watermark=False
                )
                if not images: return None
                
                output = io.BytesIO()
                images[0].save(output, format="PNG")
                return output.getvalue()

            except ResourceExhausted:
                time.sleep(3)
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
            output = io.BytesIO()
            pil_img.save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Edit Error: {e}")
            return None
