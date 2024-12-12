import math
import time

import cv2

from BoundingBox import BoundingBox
from Detector import Detector
from Tracker import Tracker
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
        trackers = []

        while ret:
            row = []
            boxes = detector.detectYOLOv4(frame)
            for i in range(len(boxes)):
                x, y, w, h, center_x, center_y = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(trackers) > 0:
                    column = []
                    value = 99999  # big value
                    tracker_id = None
                    for j in range(len(trackers)):
                        column.append(self.count_distance(boxes[i], trackers[j]))
                        if value > column[j] and trackers[j].is_used is False:
                            value = column[j]
                            tracker_id = j
                    row.append(column)

                    if value > 1000 or tracker_id is None:
                        tracker = Tracker(center_x, center_y)
                        trackers.append(tracker)
                    else:
                        column.insert(tracker_id, None)
                        tracker = trackers[tracker_id]

                else:
                    tracker = Tracker(center_x, center_y)
                    trackers.append(tracker)
                pred_x, pred_y = tracker.predict()
                tracker.update(center_x, center_y)
                tracker.is_used = True

                cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(tracker.tracker_id), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                            1, cv2.LINE_AA)

            for t in trackers:
                t.is_used = False
            new_capture.append(frame)
            ret, frame = capture.read()
            print(Tracker.tracker_next_id)

        print("Detections are ready in %s." % (time.time() - start_time))
        capture.release()
        video_player.play(new_capture)

    def count_distance(self, box: [BoundingBox], tracker: [Tracker]) -> float:
        pred_x, pred_y = tracker.predict()
        distance_x = box.center_x - pred_x
        distance_y = box.center_y - pred_y
        return math.sqrt((distance_x ** 2) + (distance_y ** 2))
