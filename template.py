import cv2
import torch
import numpy as np

import time

from LPRnet.data.load_data import CHARS
from LPRnet.model.LPRNet import build_lprnet
from LPRnet.rec_plate import rec_plate
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
        image, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC  # resolution
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
    draw_frame = frame.copy()

    n = len(labels)
    x_shape, y_shape = draw_frame.shape[1], draw_frame.shape[0]

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

        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), bgr, 2)

    labls_cords["numbers"] = numbers
    labls_cords["cars"] = cars
    labls_cords["trucks"] = trucks
    labls_cords["busses"] = buses

    cv2.rectangle(draw_frame, (0, 500), (1920, 1000), (0, 0, 0), 2)
    return draw_frame, labls_cords

def check_roi(coords):
    xc = int((coords[0] + coords[2])/2)
    yc = int((coords[1] + coords[3])/2)
    if (0 < xc < 1920) and (500 < yc < 1000):
        return True
    else:
        return False

def main():
    cv2.startWindowThread()
    detector = ObjectDetection("YOLOS_cars.pt", conf=0.3, iou=0.3)

    LPRnet =  build_lprnet(lpr_max_len=9, phase=False, class_num=len(CHARS), dropout_rate=0)
    LPRnet.to(torch.device("cuda:0"))
    LPRnet.load_state_dict(torch.load("LPRnet/weights/LPRNet__iteration_2000_28.09.pth"))

    i = 0
    for raw_frame in get_frames("test/videos/test.mp4"):
        # time.sleep(1)
        current_plates = ''
        proc_frame = preprocess(raw_frame)
        results = detector.score_frame(proc_frame)
        draw_frame, labls_cords = plot_get_boxes(results, raw_frame)
        #lp recognition
        for idx, plate_coords in enumerate(labls_cords['numbers']):
            if check_roi(plate_coords):
                x1, y1 = plate_coords[0], plate_coords[1]
                x2, y2 = plate_coords[2], plate_coords[3]
                plate = raw_frame[y1:y2, x1:x2]
                plate_text = rec_plate(LPRnet, plate)
                current_plates += plate_text + ' || '
        cv2.putText(draw_frame, current_plates, (10, 50), 0, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        if i != 0:
            pf_detected_cars = detected_cars
        detected_cars = detect_car(labls_cords)
        if i != 0:
            values = track_cars(pf_detected_cars, detected_cars)
            # make_commit_to_db(connection, values)
        cv2.imshow("video", draw_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
