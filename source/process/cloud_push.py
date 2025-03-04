import os
import cloudinary
import cloudinary.uploader
from datetime import datetime
import cv2
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


class AsyncCloudinaryUploader:
    def __init__(self):
        load_dotenv()
        self.executor = ThreadPoolExecutor(max_workers=3)  # Giới hạn số worker

        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )

    def _upload_to_cloudinary(self, frame):
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            public_id = f"violations/red_light_{timestamp}"

            result = cloudinary.uploader.upload(
                img_encoded.tobytes(),
                public_id=public_id,
                folder="traffic_violations",
                resource_type="image"
            )

            return {
                'success': True,
                'url': result['secure_url'],
                'public_id': result['public_id']
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def upload_violation(self, frame):
        self.executor.submit(self._upload_to_cloudinary, frame.copy())
