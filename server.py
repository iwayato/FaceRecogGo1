import imagezmq
import cv2
import functions

imageHub = imagezmq.ImageHub()
detector = functions.getDetector()
databaseLandmarks = functions.getDataBaseLandMarks("http://127.0.0.1:5000/getlandmarks")

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(grayFrame)
        
    for face in faces:    
        detectedLandmarks = functions.extractLandmarks(grayFrame, face)
        bestMatchIndex = functions.getBestMatchIndex(detectedLandmarks, databaseLandmarks)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, databaseLandmarks[bestMatchIndex][1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Face Detector (Server)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()