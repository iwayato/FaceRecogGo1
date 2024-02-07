import cv2

# Video de entrada
sourceVideo = cv2.VideoCapture("./Videos/test_webcam_on_Go1.mp4")

# Video de salida
fps = int(sourceVideo.get(cv2.CAP_PROP_FPS))
outputFile = './Videos/test_webcam_on_Go1_crop.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideo = cv2.VideoWriter(outputFile, fourcc, fps, (640, 480))

while True:
    ret, frame = sourceVideo.read()
    
    if not ret:
        break
    
    frame = frame[0:480, 0:640]
    
    outputVideo.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sourceVideo.release()
outputVideo.release()