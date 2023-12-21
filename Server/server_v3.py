import face_recognition
import imagezmq
import cv2

imageHub = imagezmq.ImageHub()

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    
    rgbFrame = frame[:, :, ::-1]
    
    faceLocations = face_recognition.face_locations(rgbFrame)
    
    for (top, right, bottom, left) in faceLocations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    cv2.imshow('Server', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()