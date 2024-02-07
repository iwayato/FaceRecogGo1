import cv2 as cv
import imagezmq as zmq
import dlib
import time

class camera:
    def __init__(self, cam_id = None, serverIP = None):
        self.cam_id = cam_id
        self.serverIP = serverIP

    def get_img(self):
        IpLastSegment = "15"
        cam = self.cam_id
        udpstrPrevData = "udpsrc address=192.168.123."+ IpLastSegment + " port="
        udpPORT = [9201,9202,9203,9204,9205]
        udpstrBehindData = " ! application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        udpSendIntegratedPipe = udpstrPrevData +  str(udpPORT[cam-1]) + udpstrBehindData
        self.cap = cv.VideoCapture(udpSendIntegratedPipe)

    def start(self):
        faceDetector = dlib.get_frontal_face_detector()
        shapePredictor = dlib.shape_predictor("./Predictors/shape_predictor_5_face_landmarks.dat")
        sender = zmq.ImageSender(connect_to = "tcp://" + self.serverIP + ":5555")
        self.get_img()
        processFrame = True
        while(True):
            if processFrame:
                self.success, self.frame = self.cap.read()
                self.frame = cv.flip(self.frame, -1)
                start = time.time()
                self.frame = cv.resize(self.frame, (1856, 800))
                facesDetected = faceDetector(self.frame, 0)
                for faceRectangle in facesDetected:
                    landmarks = shapePredictor(self.frame, faceRectangle)
                    faceChip = dlib.get_face_chip(self.frame, landmarks)
                    sender.send_image("Img from Go1", faceChip)
            end = time.time()
            print(end - start)
            processFrame = not processFrame
            if cv.waitKey(2) & 0xFF == ord('q'):
                break
        self.cap.release()