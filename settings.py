
import os

FILE_PATH = os.environ.get('file_path', "test/videos/test.mp4")
YOLO_CONF = os.environ.get('yolo_conf', 0.5)
YOLO_IOU = os.environ.get('yolo_iou', 0.4)
FINAL_FRAME_RES = os.environ.get('final_frame_resolution', (1080, 720))
DETECTION_AREA = os.environ.get('detection_area', [(0, 650), (1920, 1000)])