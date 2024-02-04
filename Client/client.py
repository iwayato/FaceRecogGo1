import cv2 as cv
import imagezmq as zmq
import dlib

webcam = cv.VideoCapture(0)
serverIP = "192.168.100.147"
sender = zmq.ImageSender(connect_to = "tcp://" + serverIP + ":5555")

faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor("./Predictors/shape_predictor_5_face_landmarks.dat")

processFrame = True
while True:
    success, frame = webcam.read()
    if processFrame:
        facesDetected = faceDetector(frame, 0) # Upsample
        for faceRectangle in facesDetected:
            landmarks = shapePredictor(frame, faceRectangle)
            faceChip = dlib.get_face_chip(frame, landmarks) # faceChip es la imagen de solo el rostro detectado, alineado y escalado por default a 150x150
            faceChip = cv.cvtColor(faceChip, cv.COLOR_BGR2GRAY)
            cv.imshow("Go1", faceChip)
            sender.send_image("Img from Go1", faceChip)
    processFrame = not processFrame
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    
webcam.release()
cv.destroyAllWindows()