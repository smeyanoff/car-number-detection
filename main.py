import time

import cv2
import numpy as np
import torch

from colour_detection.detect_color import detect_color
from lpr_net.model.lpr_net import build_lprnet
from lpr_net.rec_plate import rec_plate, CHARS
from object_detection.detect_car_YOLO import ObjectDetection
from track_logic import *

import settings

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


def preprocess(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Препроцесс перед отправкой на YOLO
    Ресайз, нормализация и т.д.
    """
    image = cv2.resize(
        image, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC  # resolution
    )
    return image


def get_boxes(results, frame):

    """
    return dict with labels and cords
    :param results: inferences made by model
    :param frame: frame on which cords calculated
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
        elif labels[i] == 1:
            cars.append((x1, y1, x2, y2))
        elif labels[i] == 2:
            trucks.append((x1, y1, x2, y2))
        elif labels[i] == 3:
            buses.append((x1, y1, x2, y2))

    labls_cords["numbers"] = numbers
    labls_cords["cars"] = cars
    labls_cords["trucks"] = trucks
    labls_cords["busses"] = buses

    return labls_cords


def plot_boxes(cars_list: list, frame: np.ndarray) -> np.ndarray:

    n = len(cars_list)

    for car in cars_list:

        car_type = car[2]

        x1_number, y1_number, x2_number, y2_number = car[0][0]
        number = car[0][1]

        x1_car, y1_car, x2_car, y2_car = car[1][0]
        colour = car[1][1]

        if car_type == "car":
            car_bgr = (0, 0, 255)
        elif car_type == "truck":
            car_bgr = (0, 255, 0)
        elif car_type == "bus":
            car_bgr = (255, 0, 0)

        number_bgr = (255, 255, 255)

        cv2.rectangle(frame, (x1_car, y1_car), (x2_car, y2_car), car_bgr, 2)
        cv2.putText(
            frame,
            car_type + " " + colour,
            (x1_car, y2_car + 15),
            0,
            1,
            car_bgr,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.rectangle(
            frame, (x1_number, y1_number), (x2_number, y2_number), number_bgr, 2
        )
        cv2.putText(
            frame,
            number,
            (x1_number - 20, y2_number + 30),
            0,
            1,
            number_bgr,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    detection_area = settings.DETECTION_AREA

    cv2.rectangle(frame, detection_area[0], detection_area[1], (0, 0, 0), 2)

    return frame


def check_roi(coords):

    detection_area = settings.DETECTION_AREA

    xc = int((coords[0] + coords[2]) / 2)
    yc = int((coords[1] + coords[3]) / 2)
    if (
        (detection_area[0][0] < xc < detection_area[1][0]) 
        and 
        (detection_area[0][1] < yc < detection_area[1][1])
        ):
        return True
    else:
        return False


def main(
    video_file_path, 
    yolo_model_path,
    yolo_conf, 
    yolo_iou,
    lpr_model_path,
    lpr_max_len,
    lpr_dropout_rate, 
    device
    ):

    cv2.startWindowThread()
    detector = ObjectDetection(
        yolo_model_path, 
        conf=yolo_conf, 
        iou=yolo_iou,
        device = device
        )

    LPRnet = build_lprnet(
        lpr_max_len=lpr_max_len, 
        phase=False, 
        class_num=len(CHARS), 
        dropout_rate=lpr_dropout_rate
    )
    LPRnet.to(torch.device(device))
    LPRnet.load_state_dict(
        torch.load(lpr_model_path)
    )

    for raw_frame in get_frames(video_file_path):
        
        time_start = time.time()

        proc_frame = preprocess(raw_frame, (640, 480))
        results = detector.score_frame(proc_frame)
        labls_cords = get_boxes(results, raw_frame)
        new_cars = check_numbers_overlaps(labls_cords)

        # list to write cars that've been defined
        cars = []

        for car in new_cars:

            plate_coords = car[0]
            car_coords = car[1]

            if check_roi(plate_coords):

                x1_car, y1_car = car_coords[0], car_coords[1]
                x2_car, y2_car = car_coords[2], car_coords[3]

                # define car's colour
                car_box_image = raw_frame[y1_car:y2_car, x1_car:x2_car]
                colour = detect_color(car_box_image)

                car[1] = [car_coords, colour]

                x1_plate, y1_plate = plate_coords[0], plate_coords[1]
                x2_plate, y2_plate = plate_coords[2], plate_coords[3]

                # define number on the plate
                plate_box_image = raw_frame[y1_plate:y2_plate, x1_plate:x2_plate]
                plate_text = rec_plate(LPRnet, plate_box_image)

                # check if number mutchs russian number type
                if (
                    not re.match("[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{2,3}", plate_text)
                    is None
                ):

                    car[0] = [plate_coords, plate_text + "_OK"]

                else:

                    car[0] = [plate_coords, plate_text + "_NOK"]

                cars.append(car)

        drawn_frame = plot_boxes(cars, raw_frame)
        proc_frame = preprocess(drawn_frame, settings.FINAL_FRAME_RES)

        time_end = time.time()

        cv2.imshow("video", proc_frame)
        
        # wait 5 sec if push 's'
        if cv2.waitKey(30) & 0xFF == ord("s"):

            time.sleep(5)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main(
        settings.FILE_PATH,
        settings.YOLO_MODEL_PATH,
        settings.YOLO_CONF, 
        settings.YOLO_IOU,
        settings.LPR_MODEL_PATH,
        settings.LPR_MAX_LEN, 
        settings.LPR_DROPOUT, 
        settings.DEVICE
    )
