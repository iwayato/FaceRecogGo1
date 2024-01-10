import cv2 as cv
import dlib
import sys
import numpy as np
import functions as func

# Parametros
DISTANCE = 2
SHARPENING = False
KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
RESIZE = True
RESIZE_FACTOR = 1.3
SR_RES = True
SR_MODEL = "ESPCN"
SR_FACTOR = 2
UPSAMPLE = 1

# Se inicia el detector, modelo de reconocimiento facial y se abre la imagen a procesar
predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(predictorPath)
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)

# Se procesa la imagen
img = cv.imread("./Frames/" + str(DISTANCE) + "m.png")
if img is None:
    sys.exit("No se pudo abrir la imagen")
if SHARPENING:
    img = cv.filter2D(img, -1, KERNEL)
if RESIZE:
    newWidth = int(img.shape[1] * RESIZE_FACTOR)
    newHeight = int(img.shape[0] * RESIZE_FACTOR)
    img = cv.resize(img, (newWidth, newHeight))
if SR_RES:
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "./Models/" + SR_MODEL + "_x" + str(SR_FACTOR) + ".pb"
    sr.readModel(path)
    sr.setModel("espcn", SR_FACTOR)
    img = sr.upsample(img)

# Upsample    
facesDetected = detector(img, UPSAMPLE)

for face in facesDetected:
    shape = shapePredictor(img, face)
    faceDescriptor = faceRecognitionModel.compute_face_descriptor(img, shape, 1)
    matchName = func.getBestMatches(list(faceDescriptor))
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(matchName[0][0])

cv.imshow("Face at " + str(DISTANCE) + " m", img)
    
if cv.waitKey(0) == ord("q"):
    exit