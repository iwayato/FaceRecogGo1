import imagezmq
# import functions
import cv2

# RESIZE_FACTOR = 2
# FRAME_WIDTH = 348
# FRAME_HEIGHT = 300

# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# path = "./Models/EDSR_x4.pb"
# sr.readModel(path)
# sr.setModel("edsr", 4)

imageHub = imagezmq.ImageHub()
# detector = functions.getDetector()

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    # frame = cv2.resize(frame, (FRAME_WIDTH * RESIZE_FACTOR, FRAME_HEIGHT * RESIZE_FACTOR))
    # frame = sr.upsample(frame)
    
    # faces = detector(frame, 0)
            
    # for face in faces: 
    #     x, y, w, h = face.left(), face.top(), face.width(), face.height()
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    
    cv2.imshow('Server', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()