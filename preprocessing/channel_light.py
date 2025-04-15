import cv2
import numpy as np

# Tải ảnh từ đường dẫn (thay 'image.jpg' bằng đường dẫn ảnh của bạn)
image_path = r'data\anh_da_chinh.jpg'  # Thay bằng đường dẫn tới ảnh của bạn
image = cv2.imread(image_path)

# Chuyển ảnh sang định dạng float để xử lý
image_float = image.astype(np.float32)

# Tính giá trị trung bình độ sáng của ảnh (trên tất cả các kênh BGR)
mean_brightness = np.mean(image_float)

# Mức trung bình mong muốn (128 là trung bình cho thang 0-255)
target_mean = 80

# Tính độ điều chỉnh
adjustment = target_mean - mean_brightness

# Điều chỉnh độ sáng
adjusted_image_float = image_float + adjustment

# Giới hạn giá trị trong khoảng 0-255
adjusted_image_float = np.clip(adjusted_image_float, 0, 255)

# Chuyển lại sang định dạng uint8 (định dạng ảnh tiêu chuẩn)
adjusted_image = adjusted_image_float.astype(np.uint8)

# Lưu ảnh đã chỉnh sửa
cv2.imwrite('adjusted_image_80.jpg', adjusted_image)

# Hiển thị ảnh (tùy chọn)
cv2.imshow('Adjusted Image', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()