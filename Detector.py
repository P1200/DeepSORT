import cv2
import numpy as np

from BoundingBox import BoundingBox


class Detector:
    def __init__(self):
        # Load YOLO model
        self.net = cv2.dnn.readNetFromONNX("./Model/yolo11n.onnx")
        self.netv4 = cv2.dnn.readNet("./Model/yolov4-tiny.weights", "./Model/yolov4-tiny.cfg")

    def detect(self, image):

        # Get image dimensions
        (height, width) = image.shape[:2]

        # Define the neural network input
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Perform forward propagation
        output_layer_name = self.net.getUnconnectedOutLayersNames()
        output_layers = self.net.forward(output_layer_name)

        boxes = []

        # Loop over the output layers
        for output in output_layers:
            people_detections = output[0, 4]
            for i in range(0, len(people_detections)):

                if people_detections[i] > 0.3:
                    # Object detected
                    center_x = int(output[0, 0, i])
                    center_y = int(output[0, 1, i])
                    w = int(output[0, 2, i])
                    h = int(output[0, 3, i])

                    # Rectangle coordinates
                    x = int(center_x * width / 640 - w / 2)
                    y = int(center_y * height / 640 - h / 2)
                    boxes.append(BoundingBox(x, y, w, h))
        return boxes

    def detectYOLOv4(self, image):

        # Get image dimensions
        (height, width) = image.shape[:2]

        # Define the neural network input
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.netv4.setInput(blob)

        # Perform forward propagation
        output_layer_name = self.netv4.getUnconnectedOutLayersNames()
        output_layers = self.netv4.forward(output_layer_name)

        boxes = []

        # Loop over the output layers
        for output in output_layers:
            # Loop over the detections
            for detection in output:
                # Extract the class ID and confidence of the current detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only keep detections with a high confidence
                if class_id == 0 and confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append(BoundingBox(x, y, w, h))
        return boxes
