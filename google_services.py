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
            system_instruction = """
            Ты — редактор премиального делового медиа.
            Тон: Спокойный, Интеллектуальный, Фактический.
            ЗАПРЕЩЕНО: Кликбейт, слова "Шок", "Срочно", Caps Lock.
            Язык: Строго РУССКИЙ.
            """
            self.text_model = GenerativeModel(
                "gemini-2.0-flash-001",
                system_instruction=[system_instruction]
            )
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Brain Online: Gemini 2.0 (Premium) + Imagen 3")
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
        Придумай 6 тем для премиального блога (Tech, Business, Art).
        Стиль: Спокойный, аналитический.
        Примеры: "Феномен NVIDIA", "Рынок искусства 2025", "История дома Gucci".
        Верни список строк на русском.
        """
        try:
            config = GenerationConfig(temperature=0.6)
            response = self.text_model.generate_content(prompt, generation_config=config)
            lines = [l.strip().replace("*", "").replace("-", "").strip() for l in response.text.split('\n') if l.strip()]
            return lines[:6]
        except Exception:
            # FALLBACK TOPICS (Если Google упал)
            return ["ИИ в Бизнесе", "История Ferrari", "Будущее Энергетики", "Рынок Люкса"]

    def generate_carousel_plan(self, topic: str, slide_count: int) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
        prompt = f"""
        Тема: "{topic}" ({slide_count} слайдов).
        Задача: Сценарий для спокойного, интеллектуального медиа.
        
        Текст (ru_caption):
        - Сухие факты. Цифры.
        - Макс 7 слов на слайд.
        - Строго на русском.
        
        Визуал (image_prompt):
        - Слайд 1: Minimalist Collage, High-End Magazine Style.
        - Остальные: Cinematic, Photorealistic, 3:4 Vertical, Soft Lighting.
        
        JSON Format:
        [
          {{
            "slide_number": 1, 
            "ru_caption": "Текст...", 
            "image_prompt": "Vertical 3:4, collage..."
          }}
        ]
        """
        try:
            config = GenerationConfig(temperature=0.6)
            response = self.text_model.generate_content(prompt, generation_config=config)
            data = self._extract_json(response.text)
            return data
        except Exception:
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        
        for attempt in range(2):
            try:
                images = self.image_model.generate_images(
                    prompt=prompt, number_of_images=1, aspect_ratio="3:4",
                    safety_filter_level="block_some", person_generation="allow_adult"
                )
                if not images: return None
                
                # --- УНИВЕРСАЛЬНЫЙ СПОСОБ СОХРАНЕНИЯ (БЕЗ ОШИБОК) ---
                output = io.BytesIO()
                # Если объект имеет метод save - используем его без формата, если он ругается
                # Но лучше всего достать байты, если SDK это позволяет
                try:
                    images[0].save(output, format="PNG")
                except TypeError:
                     # Если метод save не принимает format (старая/новая версия SDK)
                     images[0].save(output)
                
                return output.getvalue()

            except ResourceExhausted:
                time.sleep(3)
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

            v_img = Image(image_bytes=img_bytes) # Внимание: тут может быть конфликт имен, но в контексте класса используется VertexImage через алиас
            # (В коде выше импорт был: from vertexai.preview.vision_models import ImageGenerationModel)
            # Приведем к безопасному виду:
            from vertexai.preview.vision_models import Image as VertexImage
            
            w, h = pil_img.size
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, int(h * 0.70)), (w, h)], fill=255)
            mask_buf = io.BytesIO()
            mask.save(mask_buf, format="PNG")
            
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
