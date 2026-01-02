import os
import json
import base64
import tempfile
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel, Image
from vertexai.preview.vision_models import ImageGenerationModel

class GoogleBrain:
    def __init__(self):
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        key_base64 = os.getenv("GOOGLE_KEY_BASE64")
        if key_base64:
            creds_json = base64.b64decode(key_base64).decode('utf-8')
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            aiplatform.init(project=project_id, location=location, credentials=credentials)
        else:
            aiplatform.init(project=project_id, location=location)
        
        self.text_model = GenerativeModel("gemini-1.5-flash")
        self.image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

    def generate_topics(self) -> list[str]:
        prompt = "Предложи 5 трендовых тем для Instagram-карусели про технологии и бизнес. Верни JSON массив строк."
        response = self.text_model.generate_content(prompt)
        return json.loads(response.text.strip("```json\n").strip("```"))

    def generate_carousel_plan(self, topic: str) -> list[dict]:
        prompt = f"""Создай план карусели из 5 слайдов на тему: {topic}
        Верни JSON массив объектов с полями: image_prompt (англ), ru_caption (рус)"""
        response = self.text_model.generate_content(prompt)
        return json.loads(response.text.strip("```json\n").strip("```"))

    def generate_image(self, prompt: str) -> bytes | None:
        try:
            images = self.image_model.generate_images(prompt=prompt, number_of_images=1)
            return images[0]._image_bytes
        except Exception:
            return None

    def remove_text_from_image(self, img_bytes: bytes) -> bytes | None:
        try:
            img = Image.from_bytes(img_bytes)
            prompt = "Remove all text and watermarks from this image, keep the background clean"
            response = self.text_model.generate_content([prompt, img])
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception:
            return None
```

**Railway Variables (итого нужно 4):**
```
TELEGRAM_TOKEN=xxx
GOOGLE_KEY_BASE64=xxx (уже есть)
GOOGLE_PROJECT_ID=твой-project-id
GOOGLE_LOCATION=us-central1
