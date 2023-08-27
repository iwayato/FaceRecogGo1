import cv2
import dlib
import base64
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def getLandmarksFromFacePhoto(input_img_str):
    if input_img_str.startswith("data:image"):
        input_img_str = input_img_str.split(",", 1)[1]
    decoded_img = base64.b64decode(input_img_str)
    image_np_array = np.frombuffer(decoded_img, np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    
    for face in faces:
        shape = predictor(gray_image, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        landmarks_list = landmarks.tolist()
        landmarks_list_string = str(landmarks_list)
    
    return landmarks_list_string