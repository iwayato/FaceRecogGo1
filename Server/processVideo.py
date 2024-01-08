import time
import dlib
import cv2
import numpy as np

video = cv2.VideoCapture("./Videos/test_348_300_crop.mp4")

# Resize
# scaleFactor = 2
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# newWidth = int(width * scaleFactor)
# newHeight = int(height * scaleFactor)

# Super Resolution
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# path = "./Models/LapSRN_x4.pb"
# sr.readModel(path)
# sr.setModel("lapsrn", 4)

# predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
# faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
# shapePredictor = dlib.shape_predictor(predictorPath)
# faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)

while True:
    ret, frame = video.read()
    if ret != False:
        # frame = cv2.resize(frame, (newWidth, newHeight))
        # frame = sr.upsample(frame)
        facesDetected = detector(frame)
        
        for face in facesDetected:
            # shape = shapePredictor(frame, face)
            
            # POSIBILIDAD DE HACER UNA LLAMADA ASINCRONICA A UNA FUNCION QUE RETORNE LA IDENTIDAD
            # DE UNA PERSONA
            
            # faceDescriptor = faceRecognitionModel.compute_face_descriptor(frame, shape, 1)
            
            # Se dibuja el rectangulo que encierra el rostro detectado
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Se dibujan los landmarks de cada rostro detectado en la imagen
            # landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            # for landmark in landmarks:
            #     cv2.circle(frame, (landmark[0], landmark[1]), 1, (255, 255, 255), 2)
                
        # Se muestra el frame
        cv2.imshow('Face Detector (Server)', frame)
    else:
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # cv2.imwrite("./Frames/4m.png", frame)
        break