import cv2 as cv
import helpers

DISTANCE = 4

# Se procesa la imagen
img = cv.imread("./Frames/" + str(DISTANCE) + "m.png")
img_process = helpers.processFrame(img)
cv.imshow("Face at " + str(DISTANCE) + " m", img_process)
    
if cv.waitKey(0) == ord("q"):
    exit