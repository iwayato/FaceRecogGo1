import imagezmq as zmq
import cv2 as cv
import helpers

imageHub = zmq.ImageHub()
# namesAndEncodings = helpers.getEmbeddingsFromDB()

# Size image webcam in Go1: 1280 x 960 (x0.5 = 640 x 480)

while True:
    (msg, faceChip) = imageHub.recv_image()
    # faceEncoding = helpers.processFaceChip(faceChip)
    # name = "Desconocido"
    # for nameAndEncoding in namesAndEncodings:
    #     distanceBetweenFaces = helpers.distanceBetweenFaces(faceEncoding, nameAndEncoding[1])
    #     if distanceBetweenFaces <= 0.60:
    #         name = nameAndEncoding[0]
    # print(name)
    cv.imshow("Facechip from Go1", faceChip)
    imageHub.send_reply(b'OK')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()