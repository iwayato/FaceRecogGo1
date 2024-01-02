import cv2

# Video de entrada
sourceVideo = cv2.VideoCapture("./Videos/test_sm_res.mp4")

# Video de salida
fps = int(sourceVideo.get(cv2.CAP_PROP_FPS))
outputFile = 'test_sm_res_crop.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideo = cv2.VideoWriter(outputFile, fourcc, fps, (160, 120))

while True:
    ret, frame = sourceVideo.read()
    
    if not ret:
        break
    
    frame = frame[0:120, 0:160]
    
    outputVideo.write(frame)

sourceVideo.release()
outputVideo.release()