import cv2
import os
import time
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def draw_landmarks(image, results):
    # Vẽ landmarks lên khung hình
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi màu BGR sang RGB
    image.flags.writeable = False  # Đặt cờ không cho phép ghi
    results = model.process(image)  # Thực hiện dự đoán
    image.flags.writeable = True  # Đặt lại cờ cho phép ghi
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Chuyển đổi lại từ RGB sang BGR
    return image, results

action = 'chop'

# Tạo thư mục data/right nếu chưa tồn tại
output_dir = f"interaction/data/{action}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tìm số thứ tự của video mới nhất và đặt tên tiếp theo
def get_next_video_filename(output_dir):
    files = os.listdir(output_dir)
    video_files = [f for f in files if f.endswith(".avi")]

    if not video_files:
        return os.path.join(output_dir, "video_1.avi")

    max_number = max([int(f.split('_')[1].split('.')[0]) for f in video_files])
    next_number = max_number + 1

    return os.path.join(output_dir, f"video_{next_number}.avi")

# Thiết lập các thông số
frame_width = 640
frame_height = 480
fps = 30.0  # Frames per second
duration = 1  # Độ dài mỗi video (giây)
no_sequences = 20  # Số lượng video cần ghi lại

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)  # Đảm bảo webcam ghi với fps là 30

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

# Thiết lập số lượng khung hình cần ghi lại dựa trên thời gian (duration)
total_frames = 15

# Sử dụng mô hình MediaPipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    for sequence in range(no_sequences):
        # Đặt tên video file và khởi tạo VideoWriter
        video_path = get_next_video_filename(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Sử dụng codec XVID
        out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

        # Đọc frame đầu tiên và hiển thị thông báo STARTING COLLECTION
        ret, frame = cap.read()
        if ret:
            frame, results = mediapipe_detection(frame, holistic)
            draw_landmarks(frame, results)
            
            cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
            # Hiển thị frame lên màn hình
            cv2.imshow('Webcam', frame)
            cv2.waitKey(1000)  # Đợi 2 giây trước khi bắt đầu thu thập

        # Bắt đầu ghi video với số lượng khung hình cần ghi (total_frames)
        frame_count = 0  # Khởi tạo biến đếm khung hình

        while frame_count < total_frames:
            # Đọc frame từ webcam
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ webcam.")
                break

            # Thực hiện dự đoán với MediaPipe Holistic
            frame, results = mediapipe_detection(frame, holistic)

            # Vẽ landmarks lên frame
            draw_landmarks(frame, results)

            # Hiển thị frame đã có landmarks lên màn hình
            cv2.imshow('Webcam', frame)

            # Ghi lại từng frame đã vẽ landmarks vào file video
            out.write(frame)

            # Tăng biến đếm khung hình lên 1
            frame_count += 1

            # Kiểm tra nếu nhấn phím 'q' thì thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Giải phóng VideoWriter cho mỗi video
        out.release()

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

print("Hoàn thành quá trình thu thập dữ liệu và lưu video.")
