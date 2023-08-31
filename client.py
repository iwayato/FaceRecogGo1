from imutils.video import VideoStream
# import imutils
import imagezmq
import socket
import time

serverIP = "192.168.100.147"
sender = imagezmq.ImageSender(connect_to = "tcp://" + serverIP + ":5555")

rpiName = socket.gethostname()
# vs = VideoStream(src = 0, resolution = (160, 120)).start()
vs = VideoStream(src = 0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    # frame = imutils.resize(frame, width = 320)
    sender.send_image(rpiName, frame)