import numpy as np


class CustomKalmanFilter:
    def __init__(self, state_dim, meas_dim, control_dim=0):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.control_dim = control_dim

        # Matrices initialization
        self.transition_matrix = np.eye(state_dim)  # F
        self.measurement_matrix = np.eye(meas_dim, state_dim)  # H
        self.control_matrix = np.zeros((state_dim, control_dim)) if control_dim > 0 else None  # B

        self.process_noise_cov = np.eye(state_dim)  # Q
        self.measurement_noise_cov = np.eye(meas_dim)  # R
        self.error_cov_post = np.eye(state_dim)  # P(k|k)
        self.error_cov_pre = np.eye(state_dim)  # P(k|k-1)

        self.state_post = np.zeros((state_dim, 1))  # x(k|k)
        self.state_pre = np.zeros((state_dim, 1))  # x(k|k-1)

    def predict(self, control_input=None):
        # Predict state
        self.state_pre = self.transition_matrix @ self.state_post

        if self.control_dim > 0 and control_input is not None:
            self.state_pre += self.control_matrix @ control_input

        # Predict error covariance
        self.error_cov_pre = (
                self.transition_matrix @ self.error_cov_post @ self.transition_matrix.T
                + self.process_noise_cov
        )

        return self.state_pre

    def correct(self, measurement):
        # Compute Kalman gain
        S = (
                self.measurement_matrix @ self.error_cov_pre @ self.measurement_matrix.T
                + self.measurement_noise_cov
        )
        K = self.error_cov_pre @ self.measurement_matrix.T @ np.linalg.inv(S)

        # Update state with measurement
        y = measurement - (self.measurement_matrix @ self.state_pre)  # Residual
        self.state_post = self.state_pre + K @ y

        # Update error covariance
        I = np.eye(self.state_dim)
        self.error_cov_post = (I - K @ self.measurement_matrix) @ self.error_cov_pre

        return self.state_post
