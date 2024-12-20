import time

import cv2
import numpy as np

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

            if len(trackers) == 0:
                for box in boxes:
                    tracker = Tracker(box.center_x, box.center_y, box.w, box.h)
                    trackers.append(tracker)
                continue

            data = np.array([[traker.pred_x, traker.pred_y] for traker in trackers])
            covariance_matrix = np.cov(data.T)
            epsilon = 1e-6  # Mała wartość dla stabilności
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * epsilon
            inv_cov_matrix = np.linalg.inv(covariance_matrix)

            for i in range(len(boxes)):
                x, y, w, h, center_x, center_y = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(trackers) > 0:
                    column = []
                    value = 99999  # big value
                    tracker_id = None

                    for j in range(len(trackers)):
                        column.append(self.count_mahalanobis_distance((boxes[i].center_x, boxes[i].center_y), (trackers[j].pred_x, trackers[j].pred_y), inv_cov_matrix))
                        if value > column[j] and trackers[j].is_used is False:
                            value = column[j]
                            tracker_id = j
                    row.append(column)

                    if value > 90 or tracker_id is None:
                        tracker = Tracker(center_x, center_y, w, h)
                        trackers.append(tracker)
                    else:
                        column.insert(tracker_id, None)
                        tracker = trackers[tracker_id]

                else:
                    tracker = Tracker(center_x, center_y, w, h)
                    trackers.append(tracker)
                pred_x, pred_y = tracker.pred_x, tracker.pred_y
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

    def count_mahalanobis_distance(self, point1, point2, inv_cov_matrix) -> float:

        delta = np.array(point1) - np.array(point2)
        distance = np.sqrt(np.dot(np.dot(delta.T, inv_cov_matrix), delta))
        return distance
