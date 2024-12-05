import cv2
import os
import mediapipe as mp

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

# Thiết lập các thông số
output_file = "./movement_check/output_video_other.avi"
frame_width = 640
frame_height = 480
fps = 30.0  # Frames per second

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)  # Đảm bảo webcam ghi với fps là 30

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

# Tạo VideoWriter cho file video đầu ra
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Sử dụng codec XVID
out = cv2.VideoWriter(output_file, fourcc, fps,(frame_width, frame_height))

print(f"Đang ghi video vào file: {output_file}")

# Sử dụng mô hình MediaPipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while True:
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

        # Kiểm tra nếu nhấn phím 'q' thì thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Dừng quay video.")
            break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print("Hoàn thành quá trình quay video và lưu vào file.")
