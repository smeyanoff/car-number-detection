
import math

import torch
import cv2
import sqlite3

import settings
from DBlogic import make_commit_to_db
import re
import time


class ObjectDetection:

    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """
    
    def __init__(self):

        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.model = self.load_model()

        #хз что это такое
        self.model.conf = 0.3 # set inference threshold at 0.3
        self.model.iou = settings.IOU_THRESHOLD # set inference IOU threshold at 0.3

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.db_connection = sqlite3.connect(settings.DATABASE)

    def get_video_from_file(self):

        """
        Function creates a streaming object to read the video from the file frame by frame.
        :param self:  class object
        :return:  OpenCV object to stream video frame by frame.
        """
        cap = cv2.VideoCapture(settings.VIDEO_INPUT)
        self.width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        assert cap is not None
        return cap

    def load_model(self):

        """
        Function loads the yolo5 model from PyTorch Hub.
        """

        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./YOLOS_cars.pt')
        return model

    def score_frame(self, frame):

        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = (
            results.xyxyn[0][:, -1].to("cpu").numpy(),
            results.xyxyn[0][:, :-1].to("cpu").numpy()
            )
        return labels, cord

    def plot_get_boxes(self, results, frame):

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
                int(row[0]*x_shape), 
                int(row[1]*y_shape), 
                int(row[2]*x_shape), 
                int(row[3]*y_shape)
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

        labls_cords['numbers'] = numbers
        labls_cords['cars'] = cars
        labls_cords['trucks'] = trucks
        labls_cords['busses'] = buses

        return frame, labls_cords

    def __call__(self):

        cv2.startWindowThread()

        player = self.get_video_from_file() # create streaming service for application
        assert player.isOpened()
        frame_rate = player.get(cv2.CAP_PROP_FPS)
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        i = 0

        fps = settings.CAM_FMS
        if fps > frame_rate:
            fps = frame_rate

        current_frame = 0

        while True:
            ret, frame = player.read()

            if not ret:
                break
            
            #resize current frame 
            frame = cv2.resize( 
                frame,
                (720,480), #resolution
                fx=0,
                fy=0, 
                interpolation = cv2.INTER_CUBIC
                )

            #get calculated boxes
            results = self.score_frame(frame) 

            frame, labls_cords = self.plot_get_boxes(results, frame)

            new_cars = []

            for number in labls_cords['numbers']:

                for car in labls_cords['cars']:
                    
                    # check if number's bounding box fully overlaps car's 
                    if ((
                        car[0]
                        <= number[0]
                        <= number[2]
                        <= car[2]
                    )
                    and 
                    (
                        car[1]
                        <= number[1]
                        <= number[3]
                        <= car[3]
                    )):
                        new_cars.append([number, car, "car"])
                
                for car in labls_cords['trucks']:
                    
                    # check if number's bounding box fully overlaps car's 
                    if ((
                        car[0]
                        <= number[0]
                        <= number[2]
                        <= car[2]
                    )
                    and 
                    (
                        car[1]
                        <= number[1]
                        <= number[3]
                        <= car[3]
                    )):
                        new_cars.append([number, car, "truck"])

                for car in labls_cords['busses']:
                    
                    # check if number's bounding box fully overlaps car's 
                    if ((
                        car[0]
                        <= number[0]
                        <= number[2]
                        <= car[2]
                    )
                    and 
                    (
                        car[1]
                        <= number[1]
                        <= number[3]
                        <= car[3]
                    )):
                        new_cars.append([number, car, "busses"])

            
            if i != 0:
                pf_detected_cars = detected_cars

            detected_cars = []
            
            
            #only for the test
            import itertools
            import numpy as np
            nums = ['АА231К34', 'АО823Е63', 'ТТ621Н45']
            colours = ['yellow', 'green', 'red', 'orange']

            test_list = list(itertools.product(nums, colours))
            a = np.random.randint(0,11)

            for car in new_cars:
                
                #send car's number to the number model

                #check the number ???
                number = test_list[a][0]

                if not re.match(
                    "[А-Я]{2}[0-9]{3}[А-Я]{1}[0-9]{2}",
                    number
                    ) is None:

                    car[0] = number

                    #send car to the colour model
                    colour = test_list[a][1]
                    car[1] = colour

                    detected_cars.append(tuple(car))


            #campare cars in the past frame with
            #cars on the current frame
            if i != 0:

                for pf_det_car in pf_detected_cars:

                    for det_car in detected_cars:
                        
                        #if cars types are the same 
                        if pf_det_car[2] == det_car[2]:

                            #if cars colours are the same
                            if pf_det_car[1] == det_car[1]:
                                
                                #if numbers are closely equails
                                if (len(
                                    set(pf_det_car[0])
                                    .symmetric_difference(
                                        set(det_car[0])
                                    )) <= 2
                                    ):
                                        #I think this is the same car
                                        pf_detected_cars.remove(det_car)

                values =  list(set(pf_detected_cars) - set(detected_cars))

                print(values)

                # make_commit_to_db(self.db_connection, values)
                
            i += 1

            cv2.imshow('video', frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        player.release()

        # finally, close the window
        cv2.destroyAllWindows()
        cv2.waitKey(10)

if __name__ == '__main__':
    a = ObjectDetection()
    a()
