import torch


class ObjectDetection:

    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    Upcoming Features:
    """

    def __init__(self, model_path, conf, iou, device):

        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.__model_path = model_path
        self.model = self.load_model()

        self.model.conf = conf
        self.model.iou = iou

        self.device = device

    def load_model(self):

        """
        Function loads the yolo5 model from PyTorch Hub then use our custom weights.
        """

        model = torch.hub.load("ultralytics/yolov5", "custom", path=self.__model_path)
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
            results.xyxyn[0][:, :-1].to("cpu").numpy(),
        )
        return labels, cord
