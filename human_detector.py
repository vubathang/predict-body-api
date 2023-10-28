import cv2
import mediapipe as mp
import math
import pickle
import numpy as np

# Hàm tính toán khoảng cách
def calculateDistance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

muscle_ratio = 0.023

def calculateCircumference(weight, length): 
    return length * weight * muscle_ratio

# Khởi tạo đối tượng Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Khởi tạo đối tượng Mediapipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Danh sách để lưu dữ liệu
data = []

# Nhập chiều cao thực tế người dùng
# height = float(input("Height (cm): "))
height = 163 
weight = 49

# Đọc ảnh từ tệp ảnh trong thư mục
file_path = 'model_predict/front_img.jpg'
image = cv2.imread(file_path)

# Nhận dạng các điểm mốc trên cơ thể
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    height_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value], 
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    upper_arm_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    
    arm_length_pixel = upper_arm_length_pixel + calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                                                                                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    shoulder_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])/4
    
    leg_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]) + calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                                                                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    back_build_pixel = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    
    
    thigh_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                               landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    
    bicep_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    
    # Tính số đo thực tế từ pixel
    height_ratio = height / height_pixel
    arm_length_cm = height_ratio * arm_length_pixel
    shoulder_length_cm = height_ratio * shoulder_length_pixel
    leg_length_cm = height_ratio * leg_length_pixel
    back_build_cm = (height_ratio * back_build_pixel) - 11
    upper_arm_length_cm = height_ratio * upper_arm_length_pixel
    thigh_length_cm = height_ratio * thigh_length_pixel
    bicep_length_cm = height_ratio * bicep_length_pixel
    
    thigh_circumference = calculateCircumference(weight,thigh_length_cm)
    bicep_circumference = calculateCircumference(weight,bicep_length_cm)

    data_point = [arm_length_cm, shoulder_length_cm, leg_length_cm, back_build_cm, upper_arm_length_cm, thigh_length_cm
                  ,thigh_circumference, bicep_circumference]

    data.append(data_point)

    # Hiển thị các chỉ số cơ thể
    cv2.putText(image, f"Arm Length: {arm_length_cm:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Shoulder Length: {shoulder_length_cm:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Leg Length: {leg_length_cm:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Back Build: {back_build_cm:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Upper arm Length: {upper_arm_length_cm:.2f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Thigh Length: {thigh_length_cm:.2f} cm", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Thigh circumference: {thigh_circumference:.2f} cm", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Bicep circumference: {bicep_circumference:.2f} cm", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Vẽ các điểm mốc trên cơ thể
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Hiển thị ảnh với các chỉ số cơ thể
image = cv2.resize(image, (450, 800))

cv2.imshow('Body Measurements', image)

cv2.waitKey(0)
cv2.destroyAllWindows()



