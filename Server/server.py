import imagezmq
import cv2 as cv
import socket
import helpers

imageHub = imagezmq.ImageHub()

while True:
    (landmarks, faceChip) = imageHub.recv_image()
    # frame = helpers.processFrame(frame)
    name = helpers.processFaceChip(faceChip, landmarks)
    print(name)
    cv.imshow('Server', faceChip)
    imageHub.send_reply(b'OK')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()