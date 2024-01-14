import cv2 as cv
import imagezmq as zmq
import socket

webcam = cv.VideoCapture(0)
serverIP = "192.168.100.147"
sender = zmq.ImageSender(connect_to = "tcp://" + serverIP + ":5555")
rpiName = socket.gethostname()

while True:
    ret, frame = webcam.read()
    sender.send_image(rpiName, frame)
    
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    
webcam.release()
cv.destroyAllWindows()