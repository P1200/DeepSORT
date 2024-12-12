import cv2
import numpy as np


class Tracker:

    tracker_next_id = 1

    def __init__(self, init_x: [int], init_y: [int]):
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

        measurement = np.array([[np.float32(init_x)],
                                [np.float32(init_y)]])
        self.kalman.correct(measurement)  # Korekcja na podstawie pomiaru
        self.tracker_id = Tracker.tracker_next_id
        Tracker.tracker_next_id += 1
        self.is_used = True

    def predict(self):

        prediction = self.kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        return pred_x, pred_y

    def update(self, center_x, center_y):
        measurement = np.array([[np.float32(center_x)],
                                [np.float32(center_y)]])
        self.kalman.correct(measurement)  # Korekcja na podstawie pomiaru

    def __str__(self):
        return str(self.tracker_id) + "|" + str(Tracker.tracker_next_id) + "|" + str(self.is_used)

    def __repr__(self):
        return str(self.tracker_id) + "|" + str(Tracker.tracker_next_id) + "|" + str(self.is_used)