import os
import json
import base64
import logging
import io
import traceback
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
        
        # --- 1. АВТОРИЗАЦИЯ ---
        try:
            # Сначала пробуем ключ из Railway Variables
            key_base64 = os.getenv("GOOGLE_KEY_BASE64")
            if key_base64:
                # Чистим строку от возможных пробелов при копировании
                key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
                creds_json = base64.b64decode(key_clean).decode('utf-8')
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                aiplatform.init(project=project_id, location=location, credentials=credentials)
                logger.info("✅ Google Auth: Ключ из Railway принят.")
            
            # Если ключа нет в переменных, ищем файл (локальный тест)
            elif os.path.exists("google_key.json"):
                 credentials = service_account.Credentials.from_service_account_file("google_key.json")
                 aiplatform.init(project=project_id, location=location, credentials=credentials)
                 logger.info("✅ Google Auth: Локальный файл ключа принят.")
            
            else:
                logger.warning("⚠️ ВНИМАНИЕ: Ключи не найдены. Бот попытается работать, но скорее всего упадет.")
                aiplatform.init(project=project_id, location=location)

        except Exception as e:
            logger.error(f"❌ Критическая ошибка авторизации (проверьте GOOGLE_KEY_BASE64): {e}")

        # --- 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ (УНИВЕРСАЛЬНЫЕ ИМЕНА) ---
        
        # TEXT: Используем алиас "gemini-1.5-flash". Google сам направит на 001 или 002.
        try:
            self.text_model = GenerativeModel("gemini-1.5-flash")
            logger.info("✅ Текстовая модель Gemini подключена")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения Gemini: {e}")

        # IMAGE: Используем технический ID для Imagen 3
        try:
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Генератор картинок Imagen 3 подключен")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения Imagen: {e}")


    def generate_topics(self) -> List[str]:
        prompt = "Придумай 5 вирусных тем для Instagram-карусели про деньги, историю или будущее. Верни просто список."
        try:
            response = self.text_model.generate_content(prompt)
            lines = [line.strip().replace("*", "").replace("-", "").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:5]
        except Exception as e:
            logger.error(f"Ошибка Gemini Topics: {e}")
            logger.error(traceback.format_exc()) # Покажем детали ошибки в логах
            return ["Ошибка доступа к Google", "Проверьте права Vertex AI Administrator", "Включите API в консоли"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        prompt = f"""
        Topic: "{topic}"
        Create a plan for 4 slides.
        Output purely JSON list:
        [
          {{"slide_number": 1, "ru_caption": "Text in Russian...", "image_prompt": "Vertical 4:5 photo, [scene description], photorealistic, green circular inset detail"}}
        ]
        Do not use markdown blocks.
        """
        try:
            response = self.text_model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Ошибка Gemini Plan: {e}")
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        try:
            images = self.image_model.generate_images(
                prompt=prompt, number_of_images=1, aspect_ratio="4:5",
                safety_filter_level="block_some", person_generation="allow_adult"
            )
            output = io.BytesIO()
            images[0].save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Ошибка Imagen: {e}")
            return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        try:
            # Открываем изображение правильно
            pil_img = Image.open(io.BytesIO(img_bytes))
            w, h = pil_img.size
            
            # Маска на нижние 30%
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, int(h * 0.70)), (w, h)], fill=255)
            
            mask_buf = io.BytesIO()
            mask.save(mask_buf, format="PNG")
            
            v_img = VertexImage(image_bytes=img_bytes)
            v_mask = VertexImage(image_bytes=mask_buf.getvalue())
            
            edited = self.image_model.edit_images(
                base_image=v_img, mask=v_mask, 
                prompt="clean background, remove text", number_of_images=1
            )
            output = io.BytesIO()
            edited[0].save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Ошибка Edit: {e}")
            return None
