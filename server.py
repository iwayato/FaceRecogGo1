import psycopg2
import helpers
import imagezmq
import cv2
import functions
import threading
import os
from dotenv import load_dotenv
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from querys import insertFaceQuery, getLandmarksQuery

load_dotenv()

dbHost = os.getenv('DB_HOST')
dbName = os.getenv('DB_NAME')
dbUser = os.getenv('DB_USER')
dbPassword = os.getenv('DB_PASSWORD')

try:
    conn = psycopg2.connect(
        dbname = dbName,
        user = dbUser,
        password = dbPassword,
        host = dbHost
    )
    print("Conexión establecida con la BBDD")

except psycopg2.Error as error:
    print("Error", error)

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
CORS(app)

def faceRecognitionService():
    global outputFrame, lock
    imageHub = imagezmq.ImageHub()
    detector = functions.getDetector()
    databaseLandmarks = functions.getDataBaseLandMarks("http://127.0.0.1:5000/getlandmarks")

    while True:
        (msg, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(grayFrame)

        for face in faces:    
            detectedLandmarks = functions.extractLandmarks(grayFrame, face)
            bestMatchIndex = functions.getBestMatchIndex(detectedLandmarks, databaseLandmarks)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, databaseLandmarks[bestMatchIndex][1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with lock:
            outputFrame = frame.copy()

def generateStream():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def checkHealth():
    return "<h3>API funcionando correctamente</h3>"

@app.route("/upload", methods = ['POST'])
def uploadFacePhoto():
    jsonData = request.get_json()
    response = jsonify(jsonData)
    name = response.json['name']
    faceImage = response.json['face_image']
    if faceImage:
        cursor = conn.cursor()
        faceLandmarks = helpers.getLandmarksFromFacePhoto(faceImage)
        cursor.execute(insertFaceQuery, (name, faceLandmarks))
        conn.commit()
        cursor.close()
        return "Imagen procesada correctamente"

@app.route("/getlandmarks", methods = ['GET'])
def getLandmarks():
    cursor = conn.cursor()
    cursor.execute(getLandmarksQuery)
    registeredLandmarks = cursor.fetchall()
    cursor.close()
    return registeredLandmarks

@app.route("/videostream")
def videoStream():
	return Response(generateStream(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    thread = threading.Thread(target = faceRecognitionService)
    thread.daemon = True
    thread.start()
    app.run(debug = True, threaded = True)