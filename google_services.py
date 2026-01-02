import os
import json
import base64
import logging
import io
import traceback
from typing import List, Dict, Optional

from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, SafetySetting
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

class GoogleBrain:
    def __init__(self):
        project_id = os.getenv("GOOGLE_PROJECT_ID", "tough-shard-479214-t2")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        # Аутентификация
        try:
            key_base64 = os.getenv("GOOGLE_KEY_BASE64")
            if key_base64:
                creds_json = base64.b64decode(key_base64).decode('utf-8')
                creds_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                aiplatform.init(project=project_id, location=location, credentials=credentials)
                logger.info("✅ Google Auth: Ключ из переменных окружения применен.")
            else:
                # Пытаемся найти локальный файл (для тестов)
                if os.path.exists("google_key.json"):
                     credentials = service_account.Credentials.from_service_account_file("google_key.json")
                     aiplatform.init(project=project_id, location=location, credentials=credentials)
                     logger.info("✅ Google Auth: Локальный файл ключа применен.")
                else:
                    aiplatform.init(project=project_id, location=location)
                    logger.warning("⚠️ Google Auth: Запуск без явных ключей (может не работать на Railway).")
        except Exception as e:
            logger.error(f"❌ Ошибка авторизации Google: {e}")

        # Инициализация моделей (СТАБИЛЬНЫЕ ВЕРСИИ)
        # gemini-1.5-flash-001 - самая стабильная версия сейчас
        self.text_model = GenerativeModel("gemini-1.5-flash-001")
        # imagegeneration@006 - это стабильный Imagen 3, который работает у всех
        self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")

    def generate_topics(self) -> List[str]:
        prompt = "Предложи 5 трендовых тем для Instagram-карусели про технологии, историю или деньги. Верни просто список."
        try:
            response = self.text_model.generate_content(prompt)
            # Чистим ответ от лишнего форматирования
            lines = [line.strip().replace("*", "").replace("-", "").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:5]
        except Exception as e:
            logger.error(f"Ошибка тем: {e}")
            return ["История денег", "Секреты успеха", "Будущее ИИ", "Загадки истории"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        prompt = f"""
        Создай план карусели (3-6 слайдов) на тему: "{topic}".
        Формат ответа JSON список:
        [
          {{"slide_number": 1, "ru_caption": "Текст слайда (рус)", "image_prompt": "Prompt for Image generation (English), vertical 4:5 aspect ratio, photorealistic, with green circle detail inset"}}
        ]
        Не используй markdown ```json. Только чистый JSON.
        """
        try:
            response = self.text_model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Ошибка плана: {e}")
            return []

    def generate_image(self, prompt: str) -> Optional[bytes]:
        try:
            # Генерация
            images = self.image_model.generate_images(
                prompt=prompt, 
                number_of_images=1, 
                aspect_ratio="4:5",
                safety_filter_level="block_some",
                person_generation="allow_adult"
            )
            
            # Правильное сохранение байтов
            output = io.BytesIO()
            images[0].save(output, format="PNG")
            output.seek(0)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Ошибка генерации фото: {e}")
            # logger.error(traceback.format_exc()) # Раскомментируй для дебага
            return None

    def remove_text_from_image(self, img_bytes: bytes) -> Optional[bytes]:
        try:
            # 1. Открываем исходник (ПРАВИЛЬНЫЙ МЕТОД PIL)
            pil_img = Image.open(io.BytesIO(img_bytes))
            w, h = pil_img.size
            
            # 2. Создаем маску (нижняя треть)
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            # Закрашиваем низ белым (это зона редактирования)
            draw.rectangle([(0, int(h * 0.70)), (w, h)], fill=255)
            
            # 3. Сохраняем маску в байты
            mask_buf = io.BytesIO()
            mask.save(mask_buf, format="PNG")
            
            # 4. Готовим объекты для Vertex AI
            v_img = VertexImage(image_bytes=img_bytes)
            v_mask = VertexImage(image_bytes=mask_buf.getvalue())
            
            # 5. Отправляем на редактирование
            edited_images = self.image_model.edit_images(
                base_image=v_img,
                mask=v_mask,
                prompt="clean background, remove text, high quality texture fill",
                number_of_images=1
            )
            
            # 6. Возвращаем результат
            output = io.BytesIO()
            edited_images[0].save(output, format="PNG")
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Ошибка очистки фото: {e}")
            logger.error(traceback.format_exc())
            return None
