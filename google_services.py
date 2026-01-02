import os
import json
import base64
import logging
import io
import traceback
import asyncio
from typing import List, Dict, Optional

from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
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

        # ПОДКЛЮЧЕНИЕ МОДЕЛЕЙ
        try:
            self.text_model = GenerativeModel("gemini-1.5-flash") # Быстрая и умная
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006") # Imagen 3
            logger.info("✅ Brain Connected: Gemini Flash + Imagen 3")
        except Exception:
            self.text_model = None
            self.image_model = None

    def generate_topics(self) -> List[str]:
        if not self.text_model: return ["Ошибка API"]
        
        # ПРОМПТ: ТОЛЬКО ХАРДКОРНЫЕ ФАКТЫ И СРАВНЕНИЯ
        prompt = """
        Ты — редактор топового Instagram-паблика в нише Wealth/Facts/Tech.
        Придумай 5 тем для каруселей.
        
        КРИТЕРИИ ТЕМ:
        1. СРАВНЕНИЯ (Then vs Now, Price vs Quality, Rich vs Poor).
        2. ШОК-ФАКТЫ (Цифры, которые взрывают мозг).
        3. ТЕХНОЛОГИИ (AI, Space, Crypto).
        
        ПРИМЕРЫ (КАК НАДО):
        - iPhone 1 (2007) vs iPhone 16 (2025)
        - $100 в 1990 vs $100 сегодня
        - Зарплата Илона Маска в секунду
        - Скорость AI vs Скорость Мозга
        
        Верни только список из 5 строк. Без нумерации. На русском языке.
        """
        try:
            response = self.text_model.generate_content(prompt)
            lines = [line.strip().replace("*", "").replace("-", "").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:5]
        except Exception:
            return ["Биткоин: 2010 vs 2026", "Скорость света vs Скорость мысли", "Цена золота за 100 лет", "AI заменит врачей?"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        # ПРОМПТ: СЛАЙДЫ С МИНИМУМОМ СЛОВ
        prompt = f"""
        Topic: "{topic}"
        Create a 4-slide plan for Instagram.
        
        STRICT RULES FOR TEXT:
        1. MAX 7 WORDS per slide. Absolute minimum.
        2. NO sentences. Only "Data + Label" or "Object A vs Object B".
        3. Font style implied: Big, Bold, Impactful.
        
        STRICT RULES FOR IMAGE PROMPT:
        1. Aspect Ratio: Vertical 3:4.
        2. Style: Photorealistic, 8k, Editorial Photography.
        3. Composition: Split screen for comparisons OR Central hero object for facts.
        
        Output JSON list:
        [
          {{"slide_number": 1, "ru_caption": "10 МБ в 1990 = $5000", "image_prompt": "Vertical 3:4, vintage hard drive photo comparison..."}},
          {{"slide_number": 2, "ru_caption": "10 МБ сегодня = Бесплатно", "image_prompt": "Vertical 3:4, modern cloud server abstract..."}}
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
            # Формат 3:4 (идеально для Инсты)
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
            # Чистим нижние 35% картинки
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, int(h * 0.65)), (w, h)], fill=255)
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
