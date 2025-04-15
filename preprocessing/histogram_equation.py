import cv2
import numpy as np

def auto_adjust_image(image_path, output_path='anh_da_chinh.jpg'):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc được ảnh!")

    # Chuyển sang không gian màu YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Tính độ sáng trung bình và độ lệch chuẩn trên kênh Y (độ sáng)
    mean_brightness = np.mean(y)
    std_brightness = np.std(y)

    # Quyết định phương pháp dựa trên độ sáng trung bình
    if mean_brightness < 80:  # Ảnh quá tối
        print("Ảnh quá tối, áp dụng CLAHE...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_adjusted = clahe.apply(y)
        
        # Kiểm tra nếu vẫn quá sáng sau CLAHE
        if np.mean(y_adjusted) > 180:
            print("Kết quả quá sáng, giảm độ sáng...")
            y_adjusted = cv2.convertScaleAbs(y_adjusted, beta=-30)

    elif mean_brightness > 180:  # Ảnh quá sáng
        print("Ảnh quá sáng, áp dụng Gamma Correction...")
        gamma = 1.5  # Giá trị gamma tự động dựa trên độ sáng
        if mean_brightness > 200:
            gamma = 2.0  # Ảnh cực sáng
        y_adjusted = np.array(255*(y / 255) ** (1/gamma), dtype='uint8')

    else:  # Ảnh gần bình thường nhưng cần cải thiện
        print("Ảnh gần bình thường, áp dụng điều chỉnh nhẹ...")
        y_adjusted = cv2.convertScaleAbs(y, alpha=1.1, beta=10 if mean_brightness < 120 else -10)

    # Gộp lại các kênh
    ycrcb_adjusted = cv2.merge((y_adjusted, cr, cb))
    adjusted_image = cv2.cvtColor(ycrcb_adjusted, cv2.COLOR_YCrCb2BGR)

    # Kiểm tra kết quả cuối cùng
    final_mean = np.mean(y_adjusted)
    if final_mean < 50 or final_mean > 200:
        print("Kết quả chưa tối ưu, tinh chỉnh thêm...")
        beta_adjust = 100 - final_mean  # Điều chỉnh để đưa về khoảng trung bình (100-150)
        adjusted_image = cv2.convertScaleAbs(adjusted_image, beta=beta_adjust)

    # Lưu ảnh
    cv2.imwrite(output_path, adjusted_image)
    print(f"Đã lưu ảnh tại: {output_path}")

    # Hiển thị ảnh (tùy chọn)
    cv2.imshow('Original Image', image)
    cv2.imshow('Adjusted Image', adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Sử dụng hàm
if __name__ == "__main__":
    image_path = r'data\lai-xe-may-troi-toi-1webp20231018160600.jpg'  # Thay bằng đường dẫn ảnh của bạn
    auto_adjust_image(image_path)