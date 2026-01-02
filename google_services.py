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
        
        # --- АВТОРИЗАЦИЯ ---
        try:
            key_base64 = os.getenv("GOOGLE_KEY_BASE64")
            if key_base64:
                # Декодируем ключ
                key_clean = key_base64.strip().replace('\n', '').replace(' ', '')
                creds_json = base64.b64decode(key_clean).decode('utf-8')
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                aiplatform.init(project=project_id, location=location, credentials=credentials)
                logger.info("✅ Auth: Успешный вход через Service Account.")
            else:
                # Если ключа нет, пробуем авто-авторизацию (работает только локально с gcloud auth)
                aiplatform.init(project=project_id, location=location)
                logger.warning("⚠️ Auth: Работаю без явного ключа (Environment).")

        except Exception as e:
            logger.error(f"❌ Auth Error: {e}")

        # --- МОДЕЛИ ПО ДОКУМЕНТАЦИИ (Stable) ---
        
        # 1. GEMINI (Текст)
        # Используем "gemini-1.5-flash". 
        # Это alias, который Google сам обновляет на самую свежую стабильную версию (001, 002 и т.д.)
        try:
            self.text_model = GenerativeModel("gemini-1.5-flash")
            logger.info("✅ Gemini 1.5 Flash (Stable) подключена.")
        except Exception as e:
            logger.error(f"❌ Ошибка Gemini: {e}")
            # Заглушка, чтобы бот не падал при старте, если биллинг отключен
            self.text_model = None

        # 2. IMAGEN (Картинки)
        # "imagegeneration@006" — это официальный ID для Imagen 3 в Vertex AI
        try:
            self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("✅ Imagen 3 (v6) подключен.")
        except Exception as e:
            logger.error(f"❌ Ошибка Imagen: {e}")
            self.image_model = None


    def generate_topics(self) -> List[str]:
        if not self.text_model: return ["Ошибка биллинга Google Cloud", "Проверьте оплату проекта"]
        
        prompt = "Придумай 5 вирусных тем для Instagram-карусели. Верни список."
        try:
            response = self.text_model.generate_content(prompt)
            lines = [line.strip().replace("*", "").replace("-", "").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:5]
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            # Выводим в лог реальную причину (например, 401 Account Invalid)
            logger.error(traceback.format_exc())
            return ["Ошибка API Google", "Проверьте Billing Account", "Включите Vertex AI API"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        if not self.text_model: return []
        
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
            logger.error(f"Plan Error: {e}")
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        if not self.image_model: return None
        
        try:
            images = self.image_model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="4:5")
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
        except Exception as e:
            logger.error(f"Edit Error: {e}")
            return None
