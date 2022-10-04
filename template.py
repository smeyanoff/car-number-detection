import cv2
import numpy as np

from detect_car_YOLO import ObjectDetection
from track_logic import *


def get_frames(video_src: str) -> np.ndarray:
    """
    Генератор, котрый читает видео и отдает фреймы
    """
    cap = cv2.VideoCapture(video_src)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            print("End video")
            break
    return None


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Препроцесс перед отправкой на YOLO
    Ресайз, нормализация и т.д.
    """
    image = cv2.resize(
        image, (720, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC  # resolution
    )
    return image


def plot_get_boxes(results, frame):
    """
    plots boxes and labels on frame.
    return dict with labels and cords
    :param results: inferences made by model
    :param frame: frame on which to  make the plots
    :return: new frame with boxes and labels plotted.
    :return: dict with labels and cords
    """

    labels, cord = results

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    labls_cords = {}
    numbers = []
    cars = []
    trucks = []
    buses = []

    for i in range(n):

        row = cord[i]
        x1, y1, x2, y2 = (
            int(row[0] * x_shape),
            int(row[1] * y_shape),
            int(row[2] * x_shape),
            int(row[3] * y_shape),
        )

        if labels[i] == 0:
            numbers.append((x1, y1, x2, y2))
            bgr = (0, 0, 255)
        elif labels[i] == 1:
            cars.append((x1, y1, x2, y2))
            bgr = (0, 255, 0)
        elif labels[i] == 2:
            trucks.append((x1, y1, x2, y2))
            bgr = (255, 0, 0)
        elif labels[i] == 3:
            buses.append((x1, y1, x2, y2))
            bgr = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

    labls_cords["numbers"] = numbers
    labls_cords["cars"] = cars
    labls_cords["trucks"] = trucks
    labls_cords["busses"] = buses

    return frame, labls_cords


def main():
    cv2.startWindowThread()
    detector = ObjectDetection("YOLOS_cars.pt", conf=0.3, iou=0.3)
    i = 0
    for frame in get_frames("test/videos/test.mp4"):
        frame = preprocess(frame)
        results = detector.score_frame(frame)
        frame, labls_cords = plot_get_boxes(results, frame)
        if i != 0:
            pf_detected_cars = detected_cars
        detected_cars = detect_car(labls_cords)
        if i != 0:
            values = track_cars(pf_detected_cars, detected_cars)
            # make_commit_to_db(connection, values)
        cv2.imshow("video", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
