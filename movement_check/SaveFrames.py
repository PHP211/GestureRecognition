import cv2
import os

# Đường dẫn đến video
video_path = "./movement_check/output_video_other.avi"
output_folder = "./movement_check/frames_test"  # Thư mục để lưu frames

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Mở video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy tổng số khung hình
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Số khung hình trên giây (tùy chọn)

print(f"Đang trích xuất {frame_count} frames từ video '{video_path}'...")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tạo tên file cho từng frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_idx:03d}.jpg")
    
    # Lưu frame thành file ảnh
    cv2.imwrite(frame_filename, frame)

    # In ra thông tin (tùy chọn)
    print(f"Đã lưu: {frame_filename}")

    frame_idx += 1

cap.release()
print(f"Đã trích xuất tất cả frames. Ảnh được lưu trong thư mục '{output_folder}'.")
