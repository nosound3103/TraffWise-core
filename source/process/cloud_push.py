import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cv2
import tempfile
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


class AsyncCloudinaryUploader:
    def __init__(self):
        load_dotenv()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Giới hạn số worker

        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )

    def _upload_to_cloudinary(self, frame, public_id, folder_path):
        """Internal method to perform the actual upload to Cloudinary"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp_filename = temp.name

            # Save the frame as a JPEG file
            cv2.imwrite(temp_filename, frame)

            # Upload the image to Cloudinary
            result = cloudinary.uploader.upload(
                temp_filename,
                public_id=public_id,
                folder=folder_path,
                overwrite=True
            )

            # Remove the temporary file
            os.unlink(temp_filename)
            return result

        except Exception as e:
            print(f"Error in _upload_to_cloudinary: {e}")
            return None

    def upload_violation(self, frame, public_id, folder_path):
        """Upload a violation image to Cloudinary asynchronously"""
        try:
            future = self.executor.submit(
                self._upload_to_cloudinary,
                frame,
                public_id,
                folder_path
            )

            return future.result()
        except Exception as e:
            print(f"Error uploading to Cloudinary: {e}")
            return None

    def file_exists_on_cloudinary(self, prefix):
        """
        Check if a file with the given prefix exists on Cloudinary.

        Args:
            prefix (str): The prefix to search for (typically folder/filename without extension)

        Returns:
            bool: True if a file with the given prefix exists, False otherwise
        """
        try:
            # Search for resources with the given prefix
            result = cloudinary.api.resources(
                type="upload",
                prefix=prefix,
                max_results=1
            )

            # If resources are found, return True
            return len(result.get('resources', [])) > 0
        except Exception as e:
            print(f"Error checking if file exists on Cloudinary: {e}")
            return False

    def __del__(self):
        """Shutdown the thread pool executor when the object is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
