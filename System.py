import time

import cv2

from Detector import Detector
from VideoPlayer import VideoPlayer


class System:
    def __init__(self, video_path):
        self.video_path = video_path

    def run(self):
        video_player = VideoPlayer()
        detector = Detector()
        capture = cv2.VideoCapture(self.video_path)

        # Check if camera opened successfully
        if not capture.isOpened():
            print("Error opening video file")
            return 404

        ret, frame = capture.read()
        new_capture = []

        print("Preparing detections...")
        start_time = time.time()
        while ret:
            new_frame = detector.detectYOLOv4(frame)
            new_capture.append(new_frame)
            ret, frame = capture.read()

        print("Detections are ready in %s." % (time.time() - start_time))
        capture.release()
        video_player.play(new_capture)
