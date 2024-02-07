import imagezmq as zmq
import cv2 as cv
import helpers
import time 

imageHub = zmq.ImageHub()
namesAndEncodings = helpers.getEmbeddingsFromDB()

while True:
    (msg, faceChip) = imageHub.recv_image()
    start = time.time()
    print("Tiempo de envió por imagezmq", start - msg)
    faceEncoding = helpers.processFaceChip(faceChip)
    name = "Desconocido"
    for nameAndEncoding in namesAndEncodings:
        distanceBetweenFaces = helpers.distanceBetweenFaces(faceEncoding, nameAndEncoding[1])
        if distanceBetweenFaces <= 0.60:
            name = nameAndEncoding[0]
    end = time.time()
    print(name, end - start)
    cv.imshow("Facechip from Go1", cv.resize(faceChip, (300, 300)))
    imageHub.send_reply(b'OK')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()