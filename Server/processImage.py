import cv2 as cv
import dlib
import sys

# Se inicia el detector y se abre la imagen a procesar
detector = dlib.get_frontal_face_detector()
img = cv.imread("./Frames/4m.png")
if img is None:
    sys.exit("No se pudo abrir la imagen")

# Resize
scaleFactor = 4
newWidth = int(img.shape[1] * scaleFactor)
newHeight = int(img.shape[0] * scaleFactor)
img = cv.resize(img, (newWidth, newHeight))

# Super Resolution
sr = cv.dnn_superres.DnnSuperResImpl_create()
path = "./Models/ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn", 4)
img = sr.upsample(img)

# Upsample    
facesDetected = detector(img, 1)

for face in facesDetected:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("Img", img)
    
if cv.waitKey(0) == ord("q"):
    exit