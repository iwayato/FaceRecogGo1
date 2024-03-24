import cv2

# Video de entrada
sourceVideo = cv2.VideoCapture("./Videos/test_go1.mp4")

# Video de salida
fps = int(sourceVideo.get(cv2.CAP_PROP_FPS))
outputFile = './Videos/test_go1_final.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideo = cv2.VideoWriter(outputFile, fourcc, fps, (640, 480))

while True:
    ret, frame = sourceVideo.read()
    
    if not ret:
        break
    
    frame = frame[300:780, 640:1280]
    
    outputVideo.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sourceVideo.release()
outputVideo.release()