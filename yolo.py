
import math

import torch
import cv2

import settings


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

        # докинуть сюда нашу модель
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

    def plot_boxes(self, results, frame):

        """
        plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """

        labels, cord = results
        print(labels)
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):

            row = cord[i]
            x1, y1, x2, y2 = (
                int(row[0]*x_shape), 
                int(row[1]*y_shape), 
                int(row[2]*x_shape), 
                int(row[3]*y_shape)
                )
            # bgr = (0, 0, 255)
            if labels[i] == 0:
                bgr = (0, 0, 255)
            elif labels[i] == 1:
                bgr = (0, 255, 0)
            elif labels[i] == 2:
                bgr = (255, 0, 0)
            elif labels[i] == 3:
                bgr = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

            cv2.putText(
                frame,
                f"Total Targets: {n}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
                )

        return frame

    def __call__(self):

        cv2.startWindowThread()

        player = self.get_video_from_file() # create streaming service for application
        assert player.isOpened()
        frame_rate = player.get(cv2.CAP_PROP_FPS)
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = settings.CAM_FMS
        if fps > frame_rate:
            fps = frame_rate

        current_frame = 0

        while True:
            ret, frame = player.read()
            if not ret:
                break

            #get calculated boxes
            results = self.score_frame(frame) 

            frame = self.plot_boxes(results, frame)

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
