import cv2
import mediapipe as mp
import math
import base64
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



def predict2D(base64_data, height, weight):
    data = []
    image_data = base64.b64decode(base64_data)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # image.show()
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        height_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value], 
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        upper_arm_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        
        forearm_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        
        shoulder_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        
        leg_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]) + calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                                                                                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        back_build_pixel = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
        
        
        thigh_length_pixel = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        
        
        shouldertoknee_length_pixel = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
        
        # Tính số đo thực tế từ pixel
        height_ratio = height / height_pixel
        upper_arm_length_cm = height_ratio * upper_arm_length_pixel
        forearm_length_cm = height_ratio * forearm_length_pixel
        arm_length_cm = upper_arm_length_cm + forearm_length_cm
        shoulder_length_cm = (height_ratio * shoulder_length_pixel) + 10
        leg_length_cm = height_ratio * leg_length_pixel
        back_build_cm = (height_ratio * back_build_pixel) - 11
        shouldertoknee_length_cm = height_ratio * shouldertoknee_length_pixel
        thigh_length_cm = height_ratio * thigh_length_pixel
        
        thigh_circumference = calculateCircumference(weight,thigh_length_cm)
        bicep_circumference = calculateCircumference(weight,upper_arm_length_cm/1.3)
        calf_circumference = calculateCircumference(weight,height_ratio * calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])/1.44)
        
        data_linear = {
            'Upper arm length': upper_arm_length_cm,
            'Forearm length': forearm_length_cm,
            'Arm length': arm_length_cm,
            'Shoulder length': shoulder_length_cm,
            'Leg length': leg_length_cm,
            'Back build': back_build_cm,
            'Shoulder to knee': shouldertoknee_length_cm,
            'Thigh length': thigh_length_cm,
        }
        

        data_volumetric = {
            'Thigh circumference': thigh_circumference,
            'Bicep circumference': bicep_circumference,
            'Calf circumference': calf_circumference
        }

    return data_linear, data_volumetric




