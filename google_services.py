import os
import json
import base64
import logging
import io
import traceback
import time
from typing import List, Dict, Optional

from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
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

        # МОДЕЛИ
        try:
            self.text_model = GenerativeModel("gemini-1.5-flash")
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Models Connected: Gemini Flash + Imagen 3")
        except Exception:
            self.text_model = None
            self.image_model = None

    def generate_topics(self) -> List[str]:
        if not self.text_model: return ["Ошибка API"]
        
        # НОВЫЙ ПРОМПТ: Тренды, Сравнения, Новости
        prompt = """
        Act as a viral content researcher. Generate 5 topics for Instagram Carousels.
        Focus on: 
        1. "Visual Comparisons" (e.g., iPhone 1 vs iPhone 16).
        2. "Mind-blowing Facts" (Science/Tech/History).
        3. "News/Trends" (AI, Crypto, Space).
        
        Style: Short, Punchy, Clickbait.
        Output: A simple list of 5 strings.
        """
        try:
            response = self.text_model.generate_content(prompt)
            lines = [line.strip().replace("*", "").replace("-", "").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:5]
        except Exception:
            return ["Neuralink vs Human Brain", "Cost of Living: 1950 vs 2024", "AI Revolution Stats", "Mars Colonization Plans"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        # НОВЫЙ ПРОМПТ: МИНИМУМ СЛОВ
        prompt = f"""
        Topic: "{topic}"
        Create a 4-slide plan.
        
        RULES:
        1. TEXT: Absolute minimum. Max 10-15 words per slide. Punchy facts.
        2. STYLE: "Did you know?", "X vs Y", "then vs now".
        3. IMAGE PROMPT: Photorealistic, cinematic lighting, 8k. 
           Must include: "Vertical 3:4 aspect ratio".
           Composition: Clean center subject.
        
        Output JSON list:
        [
          {{"slide_number": 1, "ru_caption": "Super short text (Russian)", "image_prompt": "English prompt..."}}
        ]
        """
        try:
            response = self.text_model.generate_content(prompt)
            clean = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except Exception as e:
            logger.error(f"Plan Error: {e}")
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        
        try:
            # ИСПРАВЛЕНО: aspect_ratio="3:4" (4:5 не поддерживается)
            images = self.image_model.generate_images(
                prompt=prompt, 
                number_of_images=1, 
                aspect_ratio="3:4", 
                safety_filter_level="block_some", 
                person_generation="allow_adult"
            )
            output = io.BytesIO()
            images[0].save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Imagen Error: {e}")
            return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        if not self.image_model: return None
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            w, h = pil_img.size
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, int(h * 0.70)), (w, h)], fill=255)
            mask_buf = io.BytesIO()
            mask.save(mask_buf, format="PNG")
            
            v_img = VertexImage(image_bytes=img_bytes)
            v_mask = VertexImage(image_bytes=mask_buf.getvalue())
            
            edited = self.image_model.edit_images(base_image=v_img, mask=v_mask, prompt="clean background", number_of_images=1)
            output = io.BytesIO()
            edited[0].save(output, format="PNG")
            return output.getvalue()
        except Exception:
            return None
