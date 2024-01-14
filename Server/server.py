import imagezmq
import cv2 as cv
import socket
import helpers

imageHub = imagezmq.ImageHub()
rpiName = socket.gethostname()

while True:
    (rpiName, frame) = imageHub.recv_image()
    frame = helpers.processFrame(frame)
    cv.imshow('Server', frame)
    imageHub.send_reply(b'OK')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()