import time

import cv2
import numpy as np

import BoundingBox
import PersonReIdentifier
from Detector import Detector
from Tracker import Tracker
from VideoPlayer import VideoPlayer

MAX_TIME_TO_BE_TENTATIVE = 3

ID_THICKNESS = 2

ID_SCALE = 1.0

ID_COLOR = (255, 0, 0)

RECTANGLE_COLOR = (0, 255, 0)

MAHALANOBIS_DISTANCE_THRESHOLD = 120

MAX_TIME_WITHOUT_USE_FOR_TRACKER = 20

MAX_TIME_WITHOUT_USE_FOR_TENTATIVE_TRACKER = 10


class System:
    def __init__(self, video_path):
        self.video_path = video_path

    def run_with_hungarian(self):
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
            boxes = detector.detectYOLOv4(frame)

            boxes = remove_duplicate_boxes(boxes)

            if not boxes:
                print("No objects detected")
                continue

            #  Initialize trackers
            if len(trackers) == 0:
                for box in boxes:
                    draw_box_with_id(box, frame, trackers)
                    new_capture.append(frame)
                    ret, frame = capture.read()
                continue

            sorted_trackers = sorted(trackers, key=lambda tr: tr.age, reverse=False)
            useless_trackers = sorted_trackers.copy()
            cost_matrix = prepare_cost_matrix(frame, boxes, sorted_trackers)
            row_ids, column_ids = hungarian_algorithm(cost_matrix)

            assignment(boxes, column_ids, frame, row_ids, sorted_trackers, useless_trackers)

            last_chance_assignment(boxes, frame, useless_trackers)

            create_new_trackers(boxes, frame, trackers)

            give_penalty_for_bad_trackers(trackers, useless_trackers)

            new_capture.append(frame)
            ret, frame = capture.read()

        print("Detections are ready in %s." % (time.time() - start_time))
        print("Number of used trackers is: " + str(Tracker.tracker_next_id - 1))
        capture.release()
        video_player.play(new_capture)
        return new_capture


def give_penalty_for_bad_trackers(trackers, useless_trackers):
    for tracker in useless_trackers:

        if tracker.is_tentative and tracker.time_without_use > MAX_TIME_WITHOUT_USE_FOR_TENTATIVE_TRACKER:
            trackers.remove(tracker)
            continue

        tracker.time_without_use += 1
        if tracker.time_without_use > MAX_TIME_WITHOUT_USE_FOR_TRACKER:
            trackers.remove(tracker)
            continue
        x, y, w, h = tracker
        tracker.update(x, y, w, h)


def create_new_trackers(boxes, frame, trackers):
    for box in boxes:
        if not box.is_used:
            draw_box_with_id(box, frame, trackers)


def count_mahalanobis_distance(point1, point2, inv_cov_matrix) -> float:  # Four dimensions?

    delta = np.array(point1) - np.array(point2)
    distance = np.sqrt(np.dot(np.dot(delta.T, inv_cov_matrix), delta))
    return distance


def cover_zeros(matrix):
    covered_rows = set()
    covered_cols = set()

    while True:
        row_covered_this_iteration = False
        col_covered_this_iteration = False

        # Find rows with only one zero and cover them
        for i in range(matrix.shape[0]):
            if i not in covered_rows and np.sum(matrix[i] == 0) == 1:
                zero_col = np.where(matrix[i] == 0)[0][0]
                covered_cols.add(zero_col)
                covered_rows.add(i)
                row_covered_this_iteration = True
                break  # Exit the loop after covering a row

        # Find columns with only one zero and cover them
        for j in range(matrix.shape[1]):
            if j not in covered_cols and np.sum(matrix[:, j] == 0) == 1:
                zero_row = np.where(matrix[:, j] == 0)[0][0]
                covered_rows.add(zero_row)
                covered_cols.add(j)
                col_covered_this_iteration = True
                break  # Exit the loop after covering a column

        # If no new rows or columns were covered, break the while loop
        if not row_covered_this_iteration and not col_covered_this_iteration:
            break

    return covered_rows, covered_cols


def hungarian_algorithm(cost_matrix):
    cost_matrix = cost_matrix.copy()  # Avoid modifying the original matrix
    num_rows, num_cols = cost_matrix.shape

    # Step 1: Row reduction
    for i in range(num_rows):
        cost_matrix[i] -= cost_matrix[i].min()

    # Step 2: Column reduction
    for j in range(num_cols):
        cost_matrix[:, j] -= cost_matrix[:, j].min()

    # Step 3: Cover zeros with a minimum number of lines
    while True:
        covered_rows, covered_cols = cover_zeros(cost_matrix)

        # Check if we have enough lines to cover all zeros
        if len(covered_rows) + len(covered_cols) >= max(num_rows, num_cols):
            break

        # Step 4: Adjust the matrix
        uncovered_values = cost_matrix[
            np.ix_(
                [i for i in range(num_rows) if i not in covered_rows],
                [j for j in range(num_cols) if j not in covered_cols]
            )
        ]
        min_uncovered = uncovered_values.min()
        for i in range(num_rows):
            for j in range(num_cols):
                if i not in covered_rows and j not in covered_cols:
                    cost_matrix[i, j] -= min_uncovered
                elif i in covered_rows and j in covered_cols:
                    cost_matrix[i, j] += min_uncovered

    # Step 5: Assign tasks
    row_ind, col_ind = [], []
    assigned_rows = set()
    assigned_cols = set()

    zero_locs = np.argwhere(cost_matrix == 0)  # Find all zeros in the matrix

    while zero_locs.size > 0:  # Continue while there are zeros to process
        for i, j in zero_locs:
            if i not in assigned_rows and j not in assigned_cols:
                row_ind.append(i)
                col_ind.append(j)
                assigned_rows.add(i)  # Mark the row as assigned
                assigned_cols.add(j)  # Mark the column as assigned
                break  # Move to the next zero after assignment

        # Recalculate the remaining zeros after this iteration
        zero_locs = np.array([
            (i, j) for i, j in zero_locs
            if i not in assigned_rows and j not in assigned_cols
        ])

    return np.array(row_ind), np.array(col_ind)


def prepare_cost_matrix(frame, boxes, trackers):
    num_detections = len(boxes)
    num_trackers = len(trackers)
    cost_matrix = np.zeros((num_detections, num_trackers))

    for i, box in enumerate(boxes):
        x, y, w, h = box
        current_cropped_box = frame[y:y + h, x:x + w]
        box_descriptor = PersonReIdentifier.extract_descriptor(current_cropped_box)
        box.descriptor = box_descriptor
        for j, tracker in enumerate(trackers):
            appearance = tracker.compare_descriptor(box_descriptor)
            cost_matrix[i, j] = appearance

    return cost_matrix


def draw_box_with_id(box, frame, trackers):
    x, y, w, h = box
    tracker = Tracker(x, y, w, h)
    current_cropped_box = frame[y:y + h, x:x + w]
    box_descriptor = PersonReIdentifier.extract_descriptor(current_cropped_box)
    tracker.descriptors.append(box_descriptor)
    trackers.append(tracker)
    cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
    cv2.putText(frame, str(tracker.tracker_id), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, ID_SCALE, ID_COLOR,
                ID_THICKNESS, cv2.LINE_AA)


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.w, box2.x + box2.w)
    y2 = min(box1.y + box1.h, box2.y + box2.h)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    area_box1 = box1.w * box1.h
    area_box2 = box2.w * box2.h

    union_area = area_box1 + area_box2 - intersection_area

    if union_area == 0:
        return 0.0

    # IoU calculation
    return intersection_area / union_area


def calculate_iou_with_tracker(box: BoundingBox, tracker: Tracker) -> float:
    x1 = max(box.x, tracker.pred_x)
    y1 = max(box.y, tracker.pred_y)
    x2 = min(box.x + box.w, tracker.pred_x + tracker.pred_w)
    y2 = min(box.y + box.h, tracker.pred_y + tracker.pred_h)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    area_box1 = box.w * box.h
    area_box2 = tracker.pred_w * tracker.pred_h

    union_area = area_box1 + area_box2 - intersection_area

    if union_area == 0:
        return 0.0

    # IoU calculation
    return intersection_area / union_area


def remove_duplicate_boxes(boxes, iou_threshold: float = 0.5):
    if not boxes:
        return []

    # Sort boxes by area (descending) to prioritize larger boxes
    boxes = sorted(boxes, key=lambda box: box.w * box.h, reverse=True)

    result = []
    while boxes:
        current_box = boxes.pop(0)
        result.append(current_box)

        # Filter out boxes with high IoU with the current box
        boxes = [box for box in boxes if calculate_iou(current_box, box) < iou_threshold]

    return result


def assignment(boxes, column_ids, frame, row_ids, sorted_trackers, useless_trackers):
    for box_id, tracker_id in zip(row_ids, column_ids):
        data = np.array([[traker.pred_x, traker.pred_y] for traker in sorted_trackers])
        covariance_matrix = np.cov(data.T)
        epsilon = 1e-6  # To be able to count covariance matrix
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * epsilon
        inv_cov_matrix = np.linalg.inv(covariance_matrix)

        box = boxes[box_id]
        tracker = sorted_trackers[tracker_id]

        box_point = (box.x, box.y)
        tracker_point = (tracker.pred_x, tracker.pred_y)
        mahalanobis_distance = count_mahalanobis_distance(box_point, tracker_point, inv_cov_matrix)

        if mahalanobis_distance < MAHALANOBIS_DISTANCE_THRESHOLD:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
            cv2.putText(frame, str(tracker.tracker_id), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, ID_SCALE,
                        ID_COLOR,
                        ID_THICKNESS, cv2.LINE_AA)

            tracker.descriptors.append(box.descriptor)
            useless_trackers.remove(tracker)
            box.is_used = True
            tracker.age += 1
            tracker.time_without_use = 0
            if tracker.age == MAX_TIME_TO_BE_TENTATIVE:
                tracker.is_tentative = False


def last_chance_assignment(boxes, frame, useless_trackers):
    last_assignment = []
    for tracker in useless_trackers:
        best_iou = (0, None)
        for box in boxes:
            if not box.is_used:
                iou = calculate_iou_with_tracker(box, tracker)
                if iou > best_iou[0]:
                    best_iou = (iou, box)
        if best_iou[0] > 0.1:
            last_assignment.append((best_iou[1], tracker))
    for box, tracker in last_assignment:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
        cv2.putText(frame, str(tracker.tracker_id), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, ID_SCALE,
                    ID_COLOR,
                    ID_THICKNESS, cv2.LINE_AA)

        tracker.descriptors.append(box.descriptor)
        useless_trackers.remove(tracker)
        box.is_used = True
        tracker.age += 1
        tracker.time_without_use = 0
        if tracker.age == MAX_TIME_TO_BE_TENTATIVE:
            tracker.is_tentative = False
