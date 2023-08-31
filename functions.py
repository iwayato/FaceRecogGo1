import dlib
import requests
import numpy as np

def getDetector():
    return dlib.get_frontal_face_detector()

def getPredictor():
    return dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def extractLandmarks(gray_frame, face):
    predictor = getPredictor()
    shape = predictor(gray_frame, face)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
    return landmarks

def euclideanDistance(landmarks1, landmarks2):
    return np.sqrt(np.sum((landmarks1 - landmarks2) ** 2))

def getDataBaseLandMarks(url):
    response = requests.get(url)
    if response.status_code == 200:
        database_landmarks = list(response.json())
        for l in database_landmarks:
            l[2] = np.array(eval(l[2]))
        print("Se obtuvieron los landmarks desde la BBDD")
    else:
        print(f"Error: {response.status_code}")
    return database_landmarks

def euclideanDistance(detectedLandmarks, dbLandmarks):
    totalDistance = 0.0
    for i in range(len(detectedLandmarks)):
        detectedPoint = detectedLandmarks[i]
        databasePoint = dbLandmarks[i]
        totalDistance += np.sqrt(np.sum((detectedPoint - databasePoint) ** 2))
    return totalDistance

def getBestMatchIndex(detectedLandmarks, databaseLandmarks):
    return np.argmin([euclideanDistance(detectedLandmarks, dbLandmarks[2]) for dbLandmarks in databaseLandmarks])