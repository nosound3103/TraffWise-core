import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
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

    def _upload_to_cloudinary(self, frame, public_id, folder_path):
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)

            result = cloudinary.uploader.upload(
                img_encoded.tobytes(),
                public_id=public_id,
                folder=folder_path,
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

    def upload_violation(self, frame, public_id, folder_path):
        self.executor.submit(self._upload_to_cloudinary, frame.copy(), public_id, folder_path)

    def file_exists_on_cloudinary(self, public_id_prefix):
        """Check for exists file in Cloudinary."""
        try:
            response = cloudinary.api.resources(
                type="upload",
                prefix=public_id_prefix,
                max_results=1,
            )
            return len(response["resources"]) > 0
        except Exception as e:
            print(f"Error checking file on Cloudinary: {e}")
            return False