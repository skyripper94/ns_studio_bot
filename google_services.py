import os
import json
import logging
from typing import List, Dict, Optional
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
from PIL import Image
import io

logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ ---
PROJECT_ID = "tough-shard-479214-t2"  # Ваш ID из JSON
LOCATION = "us-central1"  # Imagen 3 доступен здесь
KEY_FILE = "google_key.json"

class GoogleBrain:
    def __init__(self):
        # Аутентификация через JSON файл
        if os.path.exists(KEY_FILE):
            credentials = service_account.Credentials.from_service_account_file(KEY_FILE)
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            logger.info("✅ Google Vertex AI успешно инициализирован")
        else:
            logger.error(f"❌ Не найден файл ключа {KEY_FILE}!")
            raise FileNotFoundError("Положите google_key.json в папку с ботом")

        # Модели
        self.text_model = GenerativeModel("gemini-1.5-pro-preview-0409")
        self.image_model = ImageGenerationModel.from_pretrained("imagegeneration@006") # Imagen 3

    def generate_topics(self) -> List[str]:
        """Генерирует 4 хайповые темы для каруселей"""
        prompt = """
        Ты опытный контент-мейкер для Instagram аккаунта в стиле @Wealth.
        Придумай 4 вирусные, интригующие темы для познавательных каруселей. 
        Темы могут быть про: Историю, Технологии, Загадки, Деньги, Психологию.
        Верни только список тем, каждую с новой строки. Без нумерации.
        Пример:
        Гробница первого императора Китая
        Почему Ролекс стоят так дорого
        Секрет успеха Илона Маска
        Парадокс Ферми
        """
        response = self.text_model.generate_content(prompt)
        topics = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        return topics[:4]

    def generate_carousel_plan(self, topic: str) -> List[Dict[str, str]]:
        """Создает план карусели (3-12 слайдов) с текстом и промптами"""
        
        # Наш "Скелет" промпта для стабильности стиля
        style_instruction = """
        TECHNICAL INSTRUCTIONS FOR IMAGE PROMPTS:
        Every image prompt MUST follow this strict structure:
        1. "Vertical 4:5 aspect ratio photograph."
        2. [Description of the main scene based on slide content].
        3. "COMPOSITION RULE: In the top right (or logical) corner, place a clean circular inset picture with a thick forest green border. Inside this green circle is [Close-up detail related to the slide]. A small, styled forest green arrow points from the main scene towards this circular inset."
        4. "STYLE: Photorealistic, National Geographic documentary style, cinematic lighting, 8k resolution, highly detailed."
        5. "CRITICAL: Full bleed image, completely frameless, no white border around the edge, edge-to-edge composition. No text on image."
        """

        prompt = f"""
        Topic: "{topic}"
        
        Create a viral Instagram carousel plan about this topic.
        Determine the optimal number of slides (between 3 and 12) to tell the story fully.
        
        {style_instruction}

        Output format MUST be a valid JSON list of objects:
        [
            {{
                "slide_number": 1,
                "ru_caption": "Заголовок и текст для этого слайда на Русском языке. Интригующий стиль.",
                "image_prompt": "Full English prompt following the Technical Instructions above"
            }},
            ...
        ]
        Do not add markdown formatting like ```json. Just raw JSON.
        """
        
        try:
            response = self.text_model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            plan = json.loads(clean_json)
            return plan
        except Exception as e:
            logger.error(f"Ошибка генерации плана: {e}")
            return []

    def generate_image(self, prompt: str) -> bytes:
        """Генерирует картинку через Imagen 3"""
        try:
            logger.info(f"🎨 Генерирую: {prompt[:50]}...")
            images = self.image_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="4:5",
                safety_filter_level="block_some",
                person_generation="allow_adult"
            )
            
            # Конвертируем в байты
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()
        except Exception as e:
            logger.error(f"Ошибка Imagen: {e}")
            return None

    def remove_text_from_image(self, image_bytes: bytes) -> bytes:
        """Удаляет текст (Inpainting) через Imagen Edit"""
        try:
            # Создаем маску (нижняя треть изображения, где обычно текст)
            pil_img = Image.open(io.BytesIO(image_bytes))
            w, h = pil_img.size
            mask = Image.new("L", (w, h), 0) # Черная
            draw = ImageDraw.Draw(mask)
            # Закрашиваем белым нижние 35% (зона очистки)
            draw.rectangle([(0, int(h * 0.65)), (w, h)], fill=255)
            
            # Google требует маску в байтах
            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format="PNG")
            
            vertex_img = VertexImage(image_bytes=image_bytes)
            vertex_mask = VertexImage(image_bytes=mask_byte_arr.getvalue())

            logger.info("🧹 Очистка текста через Imagen Edit...")
            edited_images = self.image_model.edit_images(
                base_image=vertex_img,
                mask=vertex_mask,
                prompt="clean background, remove text, remove typography, seamless texture fill, high quality",
                number_of_images=1
            )
            
            out_byte_arr = io.BytesIO()
            edited_images[0].save(out_byte_arr, format="PNG")
            out_byte_arr.seek(0)
            return out_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
            return None

# Вспомогательный импорт для маски
from PIL import ImageDraw
