import dlib
import cv2

video = cv2.VideoCapture("./Videos/test_bad_res.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
scaleFactor = 0.5
newWidth = int(width * scaleFactor)
newHeight = int(height * scaleFactor)

predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(predictorPath)
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)
  
while(True): 
    ret, frame = video.read()
    frame = cv2.resize(frame, (newWidth, newHeight))
    facesDetected = detector(frame)
    cv2.imshow('Video', frame) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
video.release() 
cv2.destroyAllWindows() 