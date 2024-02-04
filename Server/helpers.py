import psycopg2 as psy
import dlib
import numpy as np

# Modelos
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)

def saveEmbeddingInDB(name: str, embedding: list):
    try:
        conn = psy.connect('postgres://avnadmin:AVNS_zE37AWvXFs2CulFIeea@face-recog-db-face-recog-db.a.aivencloud.com:17725/defaultdb?sslmode=require')
        cur = conn.cursor()
        cur.execute('INSERT INTO embeddings values (%s,%s);', (name, embedding))
        conn.commit()
        conn.close()
        print("Rostro guardado exitosamente")
    except:
        print("Ha ocurrido un error")

def getEmbeddingsFromDB():
    conn = psy.connect('postgres://avnadmin:AVNS_zE37AWvXFs2CulFIeea@face-recog-db-face-recog-db.a.aivencloud.com:17725/defaultdb?sslmode=require')
    cur = conn.cursor()
    cur.execute("SELECT * FROM embeddings;")
    rows = cur.fetchall()
    conn.close()
    data = [(e[0], eval(e[1])) for e in rows]
    return data

def distanceBetweenFaces(faceToRecognize, faceInDB):
    return np.linalg.norm(np.array(faceToRecognize) - np.array(faceInDB), axis=0)

def processFaceChip(faceChip):
    return list(faceRecognitionModel.compute_face_descriptor(faceChip))