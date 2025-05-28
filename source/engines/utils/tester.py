import io
import base64
import cv2
import numpy as np
from PIL import Image


class Tester:
    def __init__(self, controller):
        self.controller = controller

    def encode_image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def process_image(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {"error": "Could not decode image"}

            annotated_image = image.copy()

            detections = self.controller.vehicle_detector.detect(image)

            vehicles = []
            for det in detections:
                label, conf, box = det
                x1, y1, x2, y2 = box

                class_id = int(label)
                vehicle_type = self.controller.class_names[class_id]

                vehicle_img = image[y1:y2, x1:x2]
                vehicle_img_base64 = None
                if vehicle_img is not None and vehicle_img.size > 0:
                    vehicle_img_base64 = self.encode_image_to_base64(
                        vehicle_img)

                lp_box = self.controller.lp_processor.detect_license_plate(
                    image, [x1, y1, x2, y2])
                plate_text, text_conf = self.controller.lp_processor.extract_text(
                    image, lp_box)

                lp_img = image[lp_box[1]:lp_box[3], lp_box[0]:lp_box[2]] if all(
                    i > 0 for i in lp_box) else None

                lp_img_base64 = None
                if lp_img is not None and lp_img.size > 0:
                    lp_img_base64 = self.encode_image_to_base64(lp_img)

                color = (0, 255, 0)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                label = f"{vehicle_type}: {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if plate_text:
                    cv2.putText(annotated_image, plate_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                vehicles.append({
                    "type": vehicle_type,
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "license_plate": plate_text,
                    "license_plate_confidence": float(text_conf) if text_conf is not None else 0,
                    "license_plate_image": lp_img_base64,
                    "vehicle_image": vehicle_img_base64
                })

            return {
                "vehicles": vehicles,
                "annotated_image": self.encode_image_to_base64(annotated_image)
            }

        except Exception as e:
            import traceback
            print(f"Error processing pipeline: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Error processing pipeline: {str(e)}"}

    def process_lp_image(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {"error": "Could not decode image"}

            plate_text, text_conf = self.controller.lp_processor.extract_text(
                image)

            return {
                "text": plate_text if plate_text else "No text detected",
                "confidence": text_conf * 100
            }
        except Exception as e:
            import traceback
            print(f"Error processing image: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Error processing image: {str(e)}"}
