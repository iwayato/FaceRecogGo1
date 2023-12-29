import dlib
import cv2

video = cv2.VideoCapture(0)

predictorPath = './Predictors/shape_predictor_5_face_landmarks.dat'
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(predictorPath)
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)
  
while(True): 
    ret, frame = video.read() 
    cv2.imshow('WebCam', frame) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
video.release() 
cv2.destroyAllWindows() 