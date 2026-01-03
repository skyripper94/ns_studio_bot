import os
import json
import base64
import logging
import io
import asyncio
import random
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
        
        # Авторизация
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

        # Модели
        try:
            self.text_model = GenerativeModel("gemini-2.0-flash-001")
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Brain Online: Gemini 2.0 + Imagen 3 (Product Mode)")
        except Exception:
            self.text_model = None
            self.image_model = None

    def _extract_json(self, text: str) -> List[Dict]:
        """Умный парсер JSON"""
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
        """Генерирует микс из Новостей, Сравнений и Фактов"""
        if not self.text_model: return ["Ошибка API"]
        
        # Жесткий промпт на актуальность и хайп
        prompt = """
        Role: Senior Content Editor for a Wealth/Tech/Hype channel.
        Task: Generate 6 viral topics for Instagram Carousels (Mix of categories).
        
        Categories to include:
        1. VERSUS (e.g., "iPhone 1 vs iPhone 16", "Dubai vs NYC").
        2. BREAKING NEWS/HYPE (e.g., "GTA 6 Release Date", "Ferrari New Hypercar").
        3. MONEY FACTS (e.g., "MrBeast Net Worth", "Rothschild Family").
        
        Style: Short, Punchy, Clickbait. NO "Introduction to...".
        Language: RUSSIAN (Strictly).
        Output: List of 6 strings. No bullets.
        """
        
        try:
            config = GenerationConfig(temperature=0.9)
            response = self.text_model.generate_content(prompt, generation_config=config)
            lines = [l.strip().replace("*", "").replace("-", "").strip() for l in response.text.split('\n') if l.strip()]
            return lines[:6]
        except Exception:
            return ["GTA 6: Что известно сейчас", "Доходы: Роналду vs Месси", "Секреты Apple", "Дубай: Мифы и Реальность"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        """Маршрутизатор: создает план в зависимости от типа темы"""
        if not self.text_model: return []
        
        # Инструкция для Gemini: Как работать с картинками и текстом
        prompt = f"""
        Act as a Professional Instagram Producer.
        Topic: "{topic}"
        Format: Carousel of {slide_count} slides.
        Target Audience: CIS/Russia (Russian Language ONLY).
        
        VISUAL STRUCTURE RULES (Important):
        - Slide 1 (Cover): Must be a HIGH-IMPACT COLLAGE. Combine main elements (e.g., Split screen X vs Y, or Hero Object + Background).
        - Middle Slides: Photorealistic, Cinematic, 8k, Vertical 3:4.
        - Last Slide: Variation of the Cover (Collage) but with different lighting/angle.
        
        TEXT RULES:
        - Language: RUSSIAN ONLY (No English text in 'ru_caption').
        - Length: Super short. Max 5-8 words per slide.
        - Style: Facts, Numbers, Dates. No "water".
        
        JSON Output Format:
        [
          {{
            "slide_number": 1, 
            "ru_caption": "GTA 6: Дата выхода подтверждена?", 
            "image_prompt": "Vertical 3:4, split screen collage, left side Grand Theft Auto Vice City graphics, right side hyper-realistic GTA 6 graphics, neon lighting"
          }},
          ...
        ]
        """
        
        try:
            config = GenerationConfig(temperature=0.7) # Чуть строже для соблюдения JSON
            response = self.text_model.generate_content(prompt, generation_config=config)
            data = self._extract_json(response.text)
            
            # Валидация: если Gemini вдруг сгенерировала меньше/больше, чем просили
            if len(data) != slide_count:
                logger.warning(f"Gemini slide count mismatch. Asked {slide_count}, got {len(data)}")
                # Можно обрезать или дополнить, но пока оставим как есть
            
            return data
        except Exception as e:
            logger.error(f"Plan Gen Error: {e}")
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        
        for attempt in range(2):
            try:
                images = self.image_model.generate_images(
                    prompt=prompt, number_of_images=1, aspect_ratio="3:4",
                    safety_filter_level="block_some", person_generation="allow_adult"
                )
                output = io.BytesIO()
                images[0].save(output, format="PNG")
                return output.getvalue()
            except ResourceExhausted:
                time.sleep(4)
                continue
            except Exception as e:
                logger.error(f"Imagen Error: {e}")
                if attempt == 1: return None
                time.sleep(1)
        return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        # Стандартная очистка
        if not self.image_model: return None
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.width > 2000 or pil_img.height > 2000:
                pil_img.thumbnail((1500, 1500))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

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
