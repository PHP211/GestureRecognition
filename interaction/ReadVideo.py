import cv2
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
    # Trích xuất keypoints của bàn tay trái (left_hand_landmarks)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark])
    else:
        lh = np.zeros((21, 2))  # shape (21, 2) vì mỗi keypoint có 2 giá trị (x, y)

    return lh

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
    
    # Chuyển sang numpy array để dễ dàng tính toán min và max
    normalized_keypoints = np.array(normalized_keypoints)
    
    # Tính toán min và max cho x và y
    x_min, y_min = np.min(normalized_keypoints, axis=0)
    x_max, y_max = np.max(normalized_keypoints, axis=0)
    
    # Tránh chia cho 0 nếu max - min = 0
    if (x_max - x_min) == 0:
        print("Cảnh báo: Tọa độ x không thay đổi, bỏ qua chuẩn hóa x.")
        x_min, x_max = 0, 1  # Cứ để giá trị x giữ nguyên, hoặc chọn giá trị mặc định
    if (y_max - y_min) == 0:
        print("Cảnh báo: Tọa độ y không thay đổi, bỏ qua chuẩn hóa y.")
        y_min, y_max = 0, 1  # Cứ để giá trị y giữ nguyên, hoặc chọn giá trị mặc định
    
    # Chuyển min và max về dạng numpy array để có thể tính toán đúng
    min_vals = np.array([x_min, y_min])
    max_vals = np.array([x_max, y_max])
    
    # Chuẩn hóa về phạm vi [-1, 1]
    normalized_keypoints = 2 * (normalized_keypoints - min_vals) / (max_vals - min_vals) - 1
    
    return normalized_keypoints

model = load_model('movement/movement_check.keras')

cap = cv2.VideoCapture(0)

frame_counter = 0

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        
        frame_counter += 1  # Tăng biến đếm lên mỗi khi đọc khung hình

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        
        print(keypoints.shape)
        
        keypoints = normalize_keypoints(keypoints)
        
        # print(f"Shape before expansion: {keypoints.shape}")
        
        # Thêm một chiều để đưa vào mô hình (cần có shape (1, 21, 2))
        keypoints = np.expand_dims(keypoints, axis=0)
        
        # print(f"Shape after expansion: {keypoints.shape}")

        # Dự đoán với mô hình
        if frame_counter % 30 != 0:  # Chỉ xử lý mỗi khung hình thứ 3 (tùy chỉnh)
            res = model.predict(keypoints)
            print(res)

            # Kiểm tra dự đoán
            if res[0] > 0.5:
                print('Other')
            else:
                print('Movement')
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()