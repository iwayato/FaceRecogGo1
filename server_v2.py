import imagezmq
import functions
import cv2

imageHub = imagezmq.ImageHub()
detector = functions.getDetector()

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LANCZOS4)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(grayFrame)
            
    for face in faces: 
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Face Detector (Server)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()