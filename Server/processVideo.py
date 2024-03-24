import cv2
import helpers
import dlib

faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor("./Predictors/shape_predictor_5_face_landmarks.dat")
namesAndEncodings = helpers.getEmbeddingsFromDB()

sourceVideo = cv2.VideoCapture("./Videos/test_webcam_on_Go1_crop.mp4")

# Video de salida
fps = int(sourceVideo.get(cv2.CAP_PROP_FPS))
outputFile = './Videos/test_webcam_on_Go1_process_threshold_06.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideo = cv2.VideoWriter(outputFile, fourcc, fps, (640, 480))

while True:
    ret, frame = sourceVideo.read()
    if not ret:
        break
    facesDetected = faceDetector(frame, 2)
    for faceRectangle in facesDetected:
        x, y, w, h = (faceRectangle.left(), faceRectangle.top(), faceRectangle.width(), faceRectangle.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        landmarks = shapePredictor(frame, faceRectangle)
        faceChip = dlib.get_face_chip(frame, landmarks)
        faceEncoding = helpers.processFaceChip(faceChip)
        name = "Desconocido"
        for nameAndEncoding in namesAndEncodings:
            distanceBetweenFaces = helpers.distanceBetweenFaces(faceEncoding, nameAndEncoding[1])
            if distanceBetweenFaces <= 0.6:
                name = nameAndEncoding[0]
                break
        cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    outputVideo.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sourceVideo.release()
outputVideo.release()