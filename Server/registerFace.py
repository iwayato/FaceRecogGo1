import cv2 as cv
import dlib
import helpers

# Using camera
# video = cv.VideoCapture(0)
# Or use a video input
video = cv.VideoCapture('./Videos/test_webcam_on_Go1_crop.mp4')
faceDetector = dlib.get_frontal_face_detector()
predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
shapePredictor = dlib.shape_predictor(predictorPath)
  
while(True): 
    sucess, img = video.read()
    imgCopy = img.copy()
    
    facesDetected = faceDetector(img)
    for faceDetected in facesDetected:
        x, y, w, h = faceDetected.left(), faceDetected.top(), faceDetected.width(), faceDetected.height()
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow('Face register', img)
        
    cv.imshow('Face register', img)
    
    if cv.waitKey(1) & 0xFF == ord('c'):    
        name = input("Ingrese el nombre de la persona a registrar: ")
        landmarks = shapePredictor(imgCopy, faceDetected)
        faceChip = dlib.get_face_chip(imgCopy, landmarks)
        faceEncoding = helpers.processFaceChip(faceChip)
        helpers.saveEmbeddingInDB(name, faceEncoding)
        cv.imshow('Face chip', faceChip)
        
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
  
video.release() 
cv.destroyAllWindows() 