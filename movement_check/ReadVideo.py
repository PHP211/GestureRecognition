import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = holistic_model.process(image)        # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark])
    else:
        rh = np.zeros((21, 2))  
    return rh

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def normalize_keypoints(keypoints):
    # Kiểm tra nếu có đủ số điểm (21 điểm cho mỗi bàn tay)
    if keypoints.shape[0] != 21:
        raise ValueError(f"Số lượng điểm keypoints không hợp lệ: {keypoints.shape[0]}")

    # Cổ tay là điểm đầu tiên trong keypoints (index 0)
    wrist = keypoints[0]
    
    # Dịch các điểm sao cho cổ tay trở thành gốc tọa độ (0, 0)
    normalized_keypoints = []
    for point in keypoints:
        normalized_point = (point[0] - wrist[0], point[1] - wrist[1])  # Chỉ cần dịch x, y
        normalized_keypoints.append(normalized_point)
    
    normalized_keypoints = np.array(normalized_keypoints)
    
    # x_min, y_min = np.min(normalized_keypoints, axis=0)
    # x_max, y_max = np.max(normalized_keypoints, axis=0)
    
    # if (x_max - x_min) == 0:
    #     print("Cảnh báo: Tọa độ x không thay đổi, bỏ qua chuẩn hóa x.")
    #     x_min, x_max = 0, 1  # Cứ để giá trị x giữ nguyên, hoặc chọn giá trị mặc định
    # if (y_max - y_min) == 0:
    #     print("Cảnh báo: Tọa độ y không thay đổi, bỏ qua chuẩn hóa y.")
    #     y_min, y_max = 0, 1  # Cứ để giá trị y giữ nguyên, hoặc chọn giá trị mặc định
    
    # min_vals = np.array([x_min, y_min])
    # max_vals = np.array([x_max, y_max])
    
    # normalized_keypoints = (normalized_keypoints - min_vals) / (max_vals - min_vals)
    
    return normalized_keypoints

def save_keypoints(action, videoCounter, keypoints):
    npy_dir = os.path.join('interaction/new_data_npy', action)
    os.makedirs(npy_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    # Lưu keypoints vào tệp .npy
    npy_path = os.path.join(npy_dir, f'{videoCounter}.npy')
    np.save(npy_path, keypoints)
    print(f'{action}: saving frame of video {videoCounter}')

def process_npy(action):    
    # Khởi tạo mô hình Mediapipe Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        
        # Lấy danh sách tất cả video trong thư mục của action cụ thể
        video_folder = f'interaction/data/{action}'
        videos = [f for f in os.listdir(video_folder) if f.endswith('.avi')]
        
        for videoCounter, videoFile in enumerate(videos):
            cap = cv2.VideoCapture(os.path.join(video_folder, videoFile))
            framesCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            points = []
            # Không cần skip frame nữa, xử lý tất cả frame
            for frameCounter in range(framesCount):  # Xử lý tất cả các frame trong video
                ret, frame = cap.read()
                
                if not ret:
                    break  # Nếu không đọc được frame thì dừng
                
                # Chạy Mediapipe để phát hiện landmarks
                image, results = mediapipe_detection(frame, holistic)
                
                # Hiển thị kết quả (có thể bỏ qua nếu không cần thiết)
                # cv2.imshow('OpenCV Feed', image)
                
                # Trích xuất keypoints từ kết quả Mediapipe
                keypoints = normalize_keypoints(extract_keypoints(results))
                
                #  Tạo thư mục để lưu các keypoints của video và frame hiện tại
                npy_dir = os.path.join('interaction/data_npy', action, str(videoCounter))
                os.makedirs(npy_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
            
                npy_path = os.path.join('interaction/data_npy', action, str(videoCounter), str(frameCounter))
                np.savetxt(npy_path, keypoints)
                
                # Nếu nhấn 'q', thoát khỏi quá trình xử lý
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            # Giải phóng tài nguyên video sau khi xử lý xong
            cap.release()
    
    # Đóng tất cả cửa sổ OpenCV
    cv2.destroyAllWindows()
    
process_npy('pinch')