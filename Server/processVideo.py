import cv2
import helpers

video = cv2.VideoCapture("./Videos/test_348_300_crop.mp4")

while True:
    ret, frame = video.read()
    if ret != False:
        frame = helpers.processFrame(frame)    
        cv2.imshow('Face Detector (Server)', frame)
    else:
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break