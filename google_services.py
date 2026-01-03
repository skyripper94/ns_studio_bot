import os
import json
import base64
import logging
import io
import asyncio
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
            # SYSTEM INSTRUCTION: ПРЕМИАЛЬНЫЙ СТИЛЬ ТЕКСТА
            system_instruction = """
            Ты — редактор премиального делового медиа.
            Тон: Спокойный, Интеллектуальный, Фактический, Дорогой.
            ЗАПРЕЩЕНО: Кликбейт, слова "Шок", "Срочно", Caps Lock, эмодзи в тексте слайдов.
            Язык: Строго РУССКИЙ.
            """
            self.text_model = GenerativeModel(
                "gemini-2.0-flash-001",
                system_instruction=[system_instruction]
            )
            # Используем лучшую доступную модель Imagen 3
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Brain Online: Gemini 2.0 (Nano Style) + Imagen 3")
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
        Придумай 6 тем для премиального блога (Tech, Business, History, Luxury).
        Стиль заголовков: Спокойный, аналитический, вызывающий интерес фактом.
        Примеры: "Экономика MrBeast", "Как Netflix меняет кино", "Феномен Ferrari".
        Верни список строк на русском.
        """
        try:
            config = GenerationConfig(temperature=0.7)
            response = self.text_model.generate_content(prompt, generation_config=config)
            lines = [l.strip().replace("*", "").replace("-", "").strip() for l in response.text.split('\n') if l.strip()]
            return lines[:6]
        except Exception:
            return ["История Ferrari", "Экономика Дубая", "Будущее ИИ", "Рынок Люкса"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        # --- ГЛАВНЫЙ ПРОМПТ С ФИРМЕННЫМ СТИЛЕМ ---
        prompt = f"""
        Тема: "{topic}" ({slide_count} слайдов).
        Задача: Сценарий для премиальной карусели.
        
        ЧАСТЬ 1: ТЕКСТ (ru_caption)
        - Сухие факты, цифры. Максимум 7 слов.
        - Строго на русском.
        
        ЧАСТЬ 2: ВИЗУАЛ (image_prompt) - САМОЕ ВАЖНОЕ!
        Ты обязан генерировать промпты строго в определенном стиле.
        
        ШАБЛОН ПРОМПТА (Используй его для каждого слайда!):
        "A vertical 3:4 aspect ratio photograph. [ОПИСАНИЕ ГЛАВНОЙ СЦЕНЫ РЕАЛИСТИЧНО]. Photorealistic, 8k, cinematic lighting. In the top right corner, there is a clean circular inset picture with a THICK FOREST GREEN BORDER showing a close-up of [ДЕТАЛЬ]. A small, styled FOREST GREEN ARROW points from the main scene towards this circular inset. Full bleed image, completely frameless."
        
        Пример для темы про MrBeast:
        "A vertical 3:4 aspect ratio photograph. Wide shot of the real Jimmy Donaldson (MrBeast) standing in a massive studio filled with money and cameras. Photorealistic, 8k, cinematic lighting. In the top right corner, a clean circular inset picture with a thick forest green border showing a close-up of his logo on a shirt. A small styled forest green arrow points from him to the circle. Full bleed."
        
        Опиши каждый слайд уникально, но строго следуя этому шаблону с зелеными элементами.
        
        JSON Output:
        [
          {{
            "slide_number": 1, 
            "ru_caption": "Заголовок...", 
            "image_prompt": "A vertical 3:4 aspect ratio photograph..."
          }}
        ]
        """
        try:
            config = GenerationConfig(temperature=0.7)
            response = self.text_model.generate_content(prompt, generation_config=config)
            data = self._extract_json(response.text)
            return data
        except Exception:
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        
        # Добавляем негативный промпт, чтобы убрать мультяшность и текст
        negative_prompt = "cartoon, anime, illustration, painting, text, watermark, signature, ugly, deformed, blurry, low quality, borders, frames"
        
        for attempt in range(2):
            try:
                # Imagen 3 поддерживает negative_prompt в последних версиях SDK
                # Если вдруг нет - он его просто проигнорирует
                try:
                    images = self.image_model.generate_images(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        number_of_images=1, 
                        aspect_ratio="3:4",
                        safety_filter_level="block_some", 
                        person_generation="allow_adult"
                    )
                except TypeError:
                     # Фоллбэк для старых версий SDK без negative_prompt
                     images = self.image_model.generate_images(
                        prompt=prompt,
                        number_of_images=1, 
                        aspect_ratio="3:4",
                        safety_filter_level="block_some", 
                        person_generation="allow_adult"
                    )

                if not images: return None
                
                output = io.BytesIO()
                # Универсальное сохранение
                try:
                    images[0].save(output, format="PNG")
                except TypeError:
                     images[0].save(output)
                
                return output.getvalue()

            except ResourceExhausted:
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Imagen Error: {e}")
                time.sleep(1)
        return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        if not self.image_model: return None
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.width > 2000:
                pil_img.thumbnail((1500, 1500))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

            v_img = Image(image_bytes=img_bytes)
            w, h = pil_img.size
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, int(h * 0.70)), (w, h)], fill=255)
            mask_buf = io.BytesIO()
            mask.save(mask_buf, format="PNG")
            
            from vertexai.preview.vision_models import Image as VertexImage
            v_img = VertexImage(image_bytes=img_bytes)
            v_mask = VertexImage(image_bytes=mask_buf.getvalue())
            
            edited = self.image_model.edit_images(base_image=v_img, mask=v_mask, prompt="clean background", number_of_images=1)
            
            out = io.BytesIO()
            try:
                edited[0].save(out, format="PNG")
            except:
                edited[0].save(out)
            return out.getvalue()
        except Exception:
            return None
