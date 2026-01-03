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
            # SYSTEM INSTRUCTION: ЗАПРЕТ НА АНГЛИЙСКИЙ
            system_instruction = """
            Ты — русскоязычный контент-мейкер. 
            Твоя задача — создавать вирусные заголовки и тексты для Instagram.
            ГЛАВНОЕ ПРАВИЛО: ВЕСЬ ВЫВОД ДОЛЖЕН БЫТЬ СТРОГО НА РУССКОМ ЯЗЫКЕ.
            Никакого английского в заголовках или описаниях.
            """
            self.text_model = GenerativeModel(
                "gemini-2.0-flash-001",
                system_instruction=[system_instruction]
            )
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Brain Online: Gemini 2.0 (Russian Force) + Imagen 3")
        except Exception:
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
        
        prompt = """
        Придумай 6 вирусных тем (хуков) для Instagram каруселей.
        Смешай категории:
        1. СРАВНЕНИЯ (Versus): Кто богаче, Что лучше (iPhone vs Samsung).
        2. НОВОСТИ (News): GTA 6, Илон Маск, Netflix, Ferrari.
        3. ФАКТЫ (Facts): Лавкрафт, Терракотовая армия, Деньги.
        
        Стиль: Кликбейт, Коротко, Хайп.
        Язык: ТОЛЬКО РУССКИЙ.
        Формат ответа: Простой список из 6 строк.
        """
        
        try:
            config = GenerationConfig(temperature=0.9)
            response = self.text_model.generate_content(prompt, generation_config=config)
            lines = [l.strip().replace("*", "").replace("-", "").strip() for l in response.text.split('\n') if l.strip()]
            return lines[:6]
        except Exception:
            return ["GTA 6: Дата выхода", "Феррари: Новая модель", "Мистер Бист: Доходы", "Apple vs Android"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        # Инструкция для Gemini под твою структуру
        prompt = f"""
        Тема: "{topic}"
        Количество слайдов: {slide_count}.
        
        ЗАДАЧА: Написать сценарий для Instagram карусели.
        ЯЗЫК: РУССКИЙ (ВСЕ ТЕКСТЫ В ru_caption ТОЛЬКО НА РУССКОМ).
        
        СТРУКТУРА:
        1. Слайд 1 (Обложка): 
           - Текст: Хук/Заголовок (коротко).
           - Картинка: КОЛЛАЖ (Split screen collage, high contrast). Смесь главных объектов темы.
        2. Средние слайды:
           - Текст: Факты, цифры, сравнения. Максимум 7 слов на слайд. Без воды.
           - Картинка: Кинематографичная, вертикальная 3:4, фотореализм.
        3. Последний слайд:
           - Картинка: Вариация коллажа с первого слайда, но другой ракурс.
        
        Верни JSON список:
        [
          {{
            "slide_number": 1, 
            "ru_caption": "Текст на русском...", 
            "image_prompt": "Vertical 3:4, split screen collage..."
          }}
        ]
        """
        
        try:
            config = GenerationConfig(temperature=0.7)
            response = self.text_model.generate_content(prompt, generation_config=config)
            data = self._extract_json(response.text)
            
            # Если AI вернул меньше слайдов, чем просили — не страшно, главное чтобы не пусто
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
                time.sleep(5) # Ждем дольше при лимитах
                continue
            except Exception as e:
                logger.error(f"Imagen Error: {e}")
                if attempt == 1: return None
                time.sleep(2)
        return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
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
