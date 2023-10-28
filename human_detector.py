import cv2
import mediapipe as mp
import math
import pickle
import numpy as np

# Hàm tính toán khoảng cách
def calculateDistance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y

    # Calculate the Distance between the two points
    dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Return the calculated Distance.
    return dis

# Khởi tạo đối tượng Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Khởi tạo đối tượng Mediapipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Danh sách để lưu dữ liệu
data = []

# Nhập chiều cao thực tế người dùng
height = float(input("Height (cm): ")) 

# Đọc ảnh từ tệp ảnh trong thư mục
file_path = 'front_img.jpg'
image = cv2.imread(file_path)

# Nhận dạng các điểm mốc trên cơ thể
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    height_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value], 
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    arm_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]) + calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    shoulder_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    
    leg_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]) + calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Tính số đo thực tế từ pixel
    height_ratio = height / height_pixel
    arm_length_cm = height_ratio * arm_length_pixel
    shoulder_length_cm = height_ratio * shoulder_length_pixel
    leg_length_cm = height_ratio * leg_length_pixel

    data_point = [arm_length_cm, shoulder_length_cm, leg_length_cm]  # Thêm các số đo khác vào danh sách ở đây

    data.append(data_point)

    # Hiển thị các chỉ số cơ thể
    cv2.putText(image, f"Arm Length: {arm_length_cm:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Shoulder Length: {shoulder_length_cm:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Leg Length: {leg_length_cm:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)





