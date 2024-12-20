import cv2
import numpy as np


class Tracker:
    tracker_next_id = 1

    def __init__(self, u: [int], v: [int], w, h):

        maxAge = 3
        isTentative = True

        F = [[1, 0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]]
        a = w/h


        # Ustawienia dla filtru Kalmana
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 stany (x, y, dx, dy), 2 obserwacje (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Szum procesu
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1  # Szum pomiaru

        measurement = np.array([[np.float32(u)],
                                [np.float32(v)]])
        self.kalman.correct(measurement)  # Korekcja na podstawie pomiaru
        prediction = self.kalman.predict()
        self.pred_x, self.pred_y = int(prediction[0]), int(prediction[1])


        self.tracker_id = Tracker.tracker_next_id
        Tracker.tracker_next_id += 1
        self.is_used = True

    def update(self, center_x, center_y):
        measurement = np.array([[np.float32(center_x)],
                                [np.float32(center_y)]])
        self.kalman.correct(measurement)  # Korekcja na podstawie pomiaru

        prediction = self.kalman.predict()
        self.pred_x, self.pred_y = int(prediction[0]), int(prediction[1])

    def __str__(self):
        return str(self.tracker_id) + "|" + str(Tracker.tracker_next_id) + "|" + str(self.is_used)

    def __repr__(self):
        return str(self.tracker_id) + "|" + str(Tracker.tracker_next_id) + "|" + str(self.is_used)
