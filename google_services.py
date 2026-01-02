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
            key_base64 = os.getenv("GOOGLE_KEY_BASE64")
            if key_base64:
                # Чистим ключ от мусора
                key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
                creds_json = base64.b64decode(key_clean).decode('utf-8')
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                aiplatform.init(project=project_id, location=location, credentials=credentials)
                logger.info("✅ Auth: Ключ из Railway принят.")
            elif os.path.exists("google_key.json"):
                 credentials = service_account.Credentials.from_service_account_file("google_key.json")
                 aiplatform.init(project=project_id, location=location, credentials=credentials)
                 logger.info("✅ Auth: Локальный файл принят.")
            else:
                aiplatform.init(project=project_id, location=location)

        except Exception as e:
            logger.error(f"❌ Auth Error: {e}")

        # --- 2. ПОИСК РАБОЧЕЙ МОДЕЛИ (GEMINI) ---
        self.text_model = None
        
        # Список всех возможных названий моделей на 2026 год
        candidates = [
            "gemini-2.0-flash-exp", # Экспериментальная (самая новая)
            "gemini-1.5-flash-002", # Обновленная Flash
            "gemini-1.5-flash-001", # Стандартная Flash
            "gemini-1.5-flash",     # Алиас
            "gemini-1.5-pro-002",   # Обновленная Pro
            "gemini-1.5-pro-001",   # Стандартная Pro
            "gemini-1.5-pro",       # Алиас Pro
            "gemini-1.0-pro",       # Старая надежная
            "gemini-pro"            # Самая старая
        ]

        logger.info("🔍 Начинаю поиск рабочей модели...")
        
        for model_name in candidates:
            try:
                # Пытаемся инициализировать
                model = GenerativeModel(model_name)
                # Пытаемся сделать холостой запрос (Ping), чтобы проверить доступ
                # Это займет секунду, но гарантирует, что модель жива
                model.generate_content("test") 
                
                self.text_model = model
                logger.info(f"🎉 УРА! Найдена рабочая модель: {model_name}")
                break # Выходим из цикла, победа
            except Exception as e:
                logger.warning(f"⚠️ {model_name} недоступна ({str(e)[:50]}...)")
                continue
        
        if not self.text_model:
            logger.critical("⛔️ НИ ОДНА МОДЕЛЬ НЕ ОТВЕТИЛА. Проверьте 'Generative AI API' в консоли.")
            # Ставим любую, чтобы не крашить инициализацию, ошибка вылетит при генерации
            self.text_model = GenerativeModel("gemini-1.5-flash")

        # --- 3. IMAGEN (Картинки) ---
        # Тут вариантов меньше, пробуем основной
        try:
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Imagen 3 подключен")
        except:
            try:
                # План Б для картинок
                self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
                logger.info("✅ Imagen 2 (Fallback) подключен")
            except Exception as e:
                logger.error(f"❌ Ошибка Imagen: {e}")


    def generate_topics(self) -> List[str]:
        prompt = "Придумай 5 вирусных тем для Instagram-карусели. Верни список."
        try:
            response = self.text_model.generate_content(prompt)
            lines = [line.strip().replace("*", "").replace("-", "").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:5]
        except Exception as e:
            logger.error(f"Ошибка генерации тем: {e}")
            logger.error(traceback.format_exc())
            return ["Как работает ИИ", "Секреты богатства", "История брендов", "Тренды 2026"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        prompt = f"""
        Topic: "{topic}"
        Create a 4-slide plan. JSON format list:
        [
          {{"slide_number": 1, "ru_caption": "Text...", "image_prompt": "Vertical 4:5 photo, [desc], green circle inset"}}
        ]
        No markdown.
        """
        try:
            response = self.text_model.generate_content(prompt)
            clean = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except Exception as e:
            logger.error(f"Ошибка плана: {e}")
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        try:
            images = self.image_model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="4:5")
            output = io.BytesIO()
            images[0].save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            logger.error(f"Ошибка картинки: {e}")
            return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
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
        except Exception as e:
            logger.error(f"Ошибка Edit: {e}")
            return None
