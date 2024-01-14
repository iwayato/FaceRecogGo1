# This does not work
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2 as cv
import base64
import threading
import dlib
import imagezmq
# import helpers as f
import socket

app = Flask(__name__)
socketio = SocketIO(app)

# Parametros
WIDTH = 348
HEIGHT = 300
RESIZE_FACTOR = 1.3
SR_MODEL = "ESPCN"
SR_FACTOR = 2
UPSAMPLE = 0

# SR - Configuracion
sr = cv.dnn_superres.DnnSuperResImpl_create()
path = "./Models/" + SR_MODEL + "_x" + str(SR_FACTOR) + ".pb"
sr.readModel(path)
sr.setModel(SR_MODEL.lower(), SR_FACTOR)

# Se inician imagezmq y modelos
imageHub = imagezmq.ImageHub()
sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:5556")
rpiName = socket.gethostname()
predictorPath = './Predictors/shape_predictor_68_face_landmarks.dat'
faceRecogPath = './Models/dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(predictorPath)
faceRecognitionModel = dlib.face_recognition_model_v1(faceRecogPath)

def processFrame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    frame = cv.resize(frame, (int(WIDTH * RESIZE_FACTOR), int(HEIGHT * RESIZE_FACTOR)))
    frame = sr.upsample(frame)
    facesDetected = detector(frame, UPSAMPLE)
    for face in facesDetected:
        shape = shapePredictor(frame, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        faceDescriptor = faceRecognitionModel.compute_face_descriptor(frame, shape, 1)
        # matchName = f.getBestMatches(list(faceDescriptor))
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.putText(frame, matchName, (x - 8, y - 8), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255)) 
    return frame

def videoThread():
    while True:
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')

        # Process the frame
        processedFrame = processFrame(frame)

        _, buffer = cv.imencode('.jpg', processedFrame)
        frameAsText = base64.b64encode(buffer).decode('utf-8')

        # Send the processed frame to all connected clients
        socketio.emit('processedFrame', frameAsText)
        
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
if __name__ == '__main__':
    video_thread = threading.Thread(target = videoThread)
    video_thread.daemon = True
    video_thread.start()

    socketio.run(app, debug=True)