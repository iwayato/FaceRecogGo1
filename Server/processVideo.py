import dlib
import cv2
import numpy as np

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

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (newWidth, newHeight))
    facesDetected = detector(frame, 1)
    
    if len(facesDetected) != 0:
        # Se procesan los rostros detectados en la imagen
        for face in facesDetected:    
            shape = shapePredictor(frame, face)
            faceDescriptor = faceRecognitionModel.compute_face_descriptor(frame, shape, 1)
            
            # Se dibuja el rectangulo que encierra el rostro detectado
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Se dibujan los landmarks de cada rostro detectado en la imagen
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            for landmark in landmarks:
                cv2.circle(frame, (landmark[0], landmark[1]), 1, (255, 255, 255), 2)

    cv2.imshow('Face Detector (Server)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

        
        
    
    
