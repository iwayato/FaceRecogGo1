import imagezmq
import cv2
import functions
import dlib
import numpy as np

imageHub = imagezmq.ImageHub()
detector = functions.getDetector()
# databaseLandmarks = functions.getDataBaseLandMarks("http://127.0.0.1:5000/getlandmarks")
predictor = dlib.shape_predictor('./Utils/shape_predictor_68_face_landmarks.dat')

if __name__ == '__main__':
    while True:
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecta rostros en la imagen
        faces = detector(grayFrame)
            
        for face in faces:    
            shape = predictor(grayFrame, face)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Este metodo no es tan preciso para calcular el match
            # bestMatchIndex = functions.getBestMatchIndex(landmarks, databaseLandmarks)
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # ccv2.putText(frame, databaseLandmarks[bestMatchIndex][1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        cv2.imshow('Face Detector (Server)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()