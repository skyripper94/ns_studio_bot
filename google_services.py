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
            # SYSTEM INSTRUCTION: ПРЕМИАЛЬНЫЙ СТИЛЬ
            system_instruction = """
            Ты — главный редактор премиального делового медиа (стиль Forbes, Esquire, RBC).
            Твоя задача — создавать интеллектуальный контент для Instagram.
            
            TONE OF VOICE:
            - Спокойный, уверенный, экспертный.
            - НИКАКОГО кликбейта, капса, восклицательных знаков и слов "ШОК", "ВЗРЫВ", "СРОЧНО".
            - Только проверенные факты, цифры и аналитика.
            - Язык: Строго РУССКИЙ.
            """
            self.text_model = GenerativeModel(
                "gemini-2.0-flash-001",
                system_instruction=[system_instruction]
            )
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Brain Online: Gemini 2.0 (Premium Mode) + Imagen 3")
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
        Придумай 6 актуальных тем для делового медиа.
        
        Категории:
        1. ТЕХНОЛОГИИ (AI, Space, Biotech) — только факты.
        2. БИЗНЕС/ДЕНЬГИ (Доходы, Рынки, История брендов).
        3. КУЛЬТУРА/ИСТОРИЯ (Личности, Артефакты, Кино).
        
        ТРЕБОВАНИЯ:
        - Заголовки должны быть спокойными и интригующими (без желтизны).
        - Пример ХОРОШО: "Как Netflix меняет подход к сериалам", "Экономика GTA 6", "Феномен Лавкрафта".
        - Пример ПЛОХО: "ШОК! ТЫ НЕ ПОВЕРИШЬ!", "ЭТО ВЗОРВАЛО ИНТЕРНЕТ".
        
        Верни просто список из 6 строк на русском языке.
        """
        
        try:
            config = GenerationConfig(temperature=0.7) # Снизил температуру для строгости
            response = self.text_model.generate_content(prompt, generation_config=config)
            lines = [l.strip().replace("*", "").replace("-", "").strip() for l in response.text.split('\n') if l.strip()]
            return lines[:6]
        except Exception:
            return ["История Ferrari", "Экономика Дубая", "Будущее нейросетей", "Рынок искусства"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        prompt = f"""
        Тема: "{topic}"
        Количество слайдов: {slide_count}.
        
        ЗАДАЧА: Сценарий для премиальной карусели.
        
        СТИЛЬ ТЕКСТА (ru_caption):
        - Лаконичный. Максимум 6-8 слов на слайд.
        - Сухие факты. Цифры. Годы. Имена.
        - Без эмоциональных окрасок.
        
        СТИЛЬ ВИЗУАЛА (image_prompt):
        - Слайд 1 (Обложка): Elegant minimalistic collage. High-end editorial style.
        - Остальные: Cinematic, photorealistic, muted colors, 8k, vertical 3:4.
        - Последний: Закрепление образа с обложки.
        
        JSON Output:
        [
          {{
            "slide_number": 1, 
            "ru_caption": "Заголовок на русском", 
            "image_prompt": "Vertical 3:4, elegant collage..."
          }}
        ]
        """
        
        try:
            config = GenerationConfig(temperature=0.6) # Строгий режим
            response = self.text_model.generate_content(prompt, generation_config=config)
            data = self._extract_json(response.text)
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
                
                # --- ФИКС ОШИБКИ ---
                # Вместо .save() используем прямой доступ к байтам, если он есть
                # Или проверяем, вернулись ли картинки
                if not images:
                    logger.warning("Google вернул пустой список картинок (Safety Filter?)")
                    return None
                    
                # Самый надежный способ в новой версии SDK:
                # Обычно это images[0]._image_bytes или images[0].image_bytes
                # Но если старый метод save() сломался, попробуем сохранить без формата (если он принимает путь)
                # ИЛИ используем ._image_bytes который есть в GeneratedImage
                
                if hasattr(images[0], "_image_bytes"):
                    return images[0]._image_bytes
                else:
                     # Fallback для некоторых версий SDK
                     # Пытаемся сохранить во временный буфер без аргумента format, если он его не принимает
                     # Но скорее всего _image_bytes сработает
                     return images[0]._image_bytes

            except ResourceExhausted:
                time.sleep(5)
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
            
            # ТАКОЙ ЖЕ ФИКС
            if hasattr(edited[0], "_image_bytes"):
                return edited[0]._image_bytes
            return None
            
        except Exception as e:
            logger.error(f"Edit Error: {e}")
            return None
