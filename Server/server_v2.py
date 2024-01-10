import dlib
import imagezmq
import functions as f
import cv2 as cv
import numpy as np

# Parametros
WIDTH = 348
HEIGHT = 300
RESIZE_FACTOR = 1.3
SR_MODEL = "ESPCN"
SR_FACTOR = 2
UPSAMPLE = 1

# SR - Configuracion
sr = cv.dnn_superres.DnnSuperResImpl_create()
path = "./Models/" + SR_MODEL + "_x" + str(SR_FACTOR) + ".pb"
sr.readModel(path)
sr.setModel(SR_MODEL.lower(), SR_FACTOR)

# Se inician imagezmq y modelos
imageHub = imagezmq.ImageHub()
predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(predictorPath)
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    frame = cv.resize(frame, (int(WIDTH * RESIZE_FACTOR), int(HEIGHT * RESIZE_FACTOR)))
    frame = sr.upsample(frame)
    facesDetected = detector(frame, UPSAMPLE)
    
    for face in facesDetected:
        shape = shapePredictor(frame, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        faceDescriptor = faceRecognitionModel.compute_face_descriptor(frame, shape, 1)
        matchName = f.getBestMatches(list(faceDescriptor))
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, matchName, (x - 8, y - 8), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255))
    
    cv.imshow('Server', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()