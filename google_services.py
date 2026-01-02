import os
import json
import logging
import base64
import io
from typing import List, Dict, Optional
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ ---
PROJECT_ID = "tough-shard-479214-t2"  # Ваш ID проекта
LOCATION = "us-central1"
KEY_FILE = "google_key.json"

class GoogleBrain:
    def __init__(self):
        self._setup_credentials()
        
        # Инициализация моделей
        self.text_model = GenerativeModel("gemini-1.5-flash") # Быстрее и стабильнее
        self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006") # Imagen 3

    def _setup_credentials(self):
        """Восстанавливает файл ключа из переменной окружения Railway"""
        # Если файла нет, но есть переменная в Railway -> создаем файл
        if not os.path.exists(KEY_FILE) and os.getenv("GOOGLE_KEY_BASE64"):
            try:
                decoded_key = base64.b64decode(os.getenv("GOOGLE_KEY_BASE64")).decode()
                with open(KEY_FILE, "w") as f:
                    f.write(decoded_key)
                logger.info("🔑 Файл ключа восстановлен из Environment Variables")
            except Exception as e:
                logger.error(f"❌ Ошибка декодирования ключа: {e}")

        # Подключение
        if os.path.exists(KEY_FILE):
            credentials = service_account.Credentials.from_service_account_file(KEY_FILE)
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            logger.info("✅ Google Vertex AI успешно подключен")
        else:
            raise FileNotFoundError("Нет файла google_key.json и переменной GOOGLE_KEY_BASE64")

    def generate_topics(self) -> List[str]:
        prompt = """
        Ты контент-мейкер для Instagram в стиле 'Old Money' / 'Wealth'.
        Придумай 4 вирусные темы для карусели (факты, история, бизнес, тайны).
        Верни только список, каждую тему с новой строки. Без нумерации.
        """
        try:
            response = self.text_model.generate_content(prompt)
            return [line.strip() for line in response.text.strip().split('\n') if line.strip()][:4]
        except Exception as e:
            logger.error(f"Ошибка тем: {e}")
            return ["Ошибка генерации тем"]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        # Наш жесткий шаблон стиля
        style_prompt = """
        TECHNICAL IMAGE PROMPT RULES:
        1. "Vertical 4:5 aspect ratio photograph."
        2. [Scene Description].
        3. "COMPOSITION: Top right corner circular inset with thick forest green border. Inside: [Detail]. Small forest green arrow points to it."
        4. "STYLE: Photorealistic, National Geographic, 8k."
        5. "CRITICAL: Full bleed, no white borders, frameless."
        """

        prompt = f"""
        Topic: "{topic}"
        Create a plan for an Instagram carousel (3 to 10 slides).
        {style_prompt}
        
        Output valid JSON list:
        [
            {{
                "slide_number": 1,
                "ru_caption": "Russian text for post...",
                "image_prompt": "English prompt following RULES..."
            }}
        ]
        Do not use markdown blocks. Just JSON.
        """
        try:
            response = self.text_model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"Ошибка плана: {e}")
            return []

    def generate_image(self, prompt: str) -> bytes:
        try:
            images = self.image_model.generate_images(
                prompt=prompt, number_of_images=1, aspect_ratio="4:5"
            )
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()
        except Exception as e:
            logger.error(f"Ошибка Imagen: {e}")
            return None

    def remove_text_from_image(self, image_bytes: bytes) -> bytes:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w, h = pil_img.size
            # Маска на нижние 35% картинки
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, int(h * 0.65)), (w, h)], fill=255)
            
            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format="PNG")
            
            vertex_img = VertexImage(image_bytes=image_bytes)
            vertex_mask = VertexImage(image_bytes=mask_byte_arr.getvalue())

            edited = self.image_model.edit_images(
                base_image=vertex_img,
                mask=vertex_mask,
                prompt="clean background, remove text, seamless texture fill",
                number_of_images=1
            )
            
            out = io.BytesIO()
            edited[0].save(out, format="PNG")
            out.seek(0)
            return out.getvalue()
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
            return None
