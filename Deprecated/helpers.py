import psycopg2 as psy
import cv2 as cv
import dlib
import numpy as np

# Parametros
WIDTH = 348
HEIGHT = 300
RESIZE_FACTOR = 2
SR_MODEL = "ESPCN"
SR_FACTOR = 4
UPSAMPLE = 2

# Configuracion Super Resolution
sr = cv.dnn_superres.DnnSuperResImpl_create()
path = "./Models/" + SR_MODEL + "_x" + str(SR_FACTOR) + ".pb"
sr.readModel(path)
sr.setModel(SR_MODEL.lower(), SR_FACTOR)

# Modelos
predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(predictorPath)
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)

def getBestMatches(embedding: list):
    conn = psy.connect('postgres://avnadmin:AVNS_zE37AWvXFs2CulFIeea@face-recog-db-face-recog-db.a.aivencloud.com:17725/defaultdb?sslmode=require')
    cur = conn.cursor()
    stringRep = "[" + ",".join(str(x) for x in embedding) + "]"
    cur.execute("SELECT name FROM embeddings ORDER BY embedding <-> %s;", (stringRep,))
    rows = cur.fetchone()
    return rows[0]
    
def processFrame(frame):
    # frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    # frame = cv.resize(frame, (int(WIDTH * RESIZE_FACTOR), int(HEIGHT * RESIZE_FACTOR)))
    # frame = sr.upsample(frame)
    facesDetected = detector(frame, UPSAMPLE)
    for face in facesDetected:
        shape = shapePredictor(frame, face)
        faceDescriptor = faceRecognitionModel.compute_face_descriptor(frame, shape, 1)
        matchName = getBestMatches(list(faceDescriptor))
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, matchName, (x - 8, y - 8), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255)) 
    return frame