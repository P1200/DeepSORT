from statistics import mean

import numpy as np

import PersonReIdentifier
from CustomKalmanFilter import CustomKalmanFilter


class Tracker:
    tracker_next_id = 1

    def __init__(self, u: [int], v: [int], w, h):
        self.age = 0
        self.time_without_use = 0
        self.is_tentative = True

        self.descriptors = []

        F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # H
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)

        # Kalman filter settings
        self.kalman = CustomKalmanFilter(8, 4)  # 8 states (x, y, w, h, dx, dy, dw, dh), 4 observations (x, y, w, h)

        self.kalman.measurementMatrix = H
        self.kalman.transitionMatrix = F

        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03  # Process noice (Q)
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1  # Measurement noice (R)

        measurement = np.array([[np.float32(u)],  # Z
                                [np.float32(v)],
                                [np.float32(w)],
                                [np.float32(h)]])

        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        self.pred_x, self.pred_y, self.pred_w, self.pred_h = int(prediction[0]), int(prediction[1]), int(
            prediction[2]), int(prediction[3])

        self.tracker_id = Tracker.tracker_next_id
        Tracker.tracker_next_id += 1
        self.is_used = True

    def update(self, center_x, center_y, w, h):
        measurement = np.array([[np.float32(center_x)],
                                [np.float32(center_y)],
                                [np.float32(w)],
                                [np.float32(h)]])
        self.kalman.correct(measurement)

        prediction = self.kalman.predict()
        self.pred_x, self.pred_y, self.pred_w, self.pred_h = int(prediction[0]), int(prediction[1]), int(
            prediction[2]), int(prediction[3])

    def compare_descriptor(self, descriptor) -> float:
        return mean(PersonReIdentifier.compare_descriptors(descriptor, d) for d in self.descriptors)

    def __iter__(self):
        return iter((self.pred_x, self.pred_y, self.pred_w, self.pred_h))

    def __str__(self):
        return str(self.tracker_id) + "|" + str(Tracker.tracker_next_id) + "|" + str(self.is_used) + "|" + str(self.age)

    def __repr__(self):
        return str(self.tracker_id) + "|" + str(Tracker.tracker_next_id) + "|" + str(self.is_used) + "|" + str(self.age)
