from ultralytics import YOLO
import cv2
from sort import Sort
from utils import get_car, write_csv
import torch
import numpy as np
from pipeline import get_vehicles,get_license_plate_score

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = torch.load('artifacts/licence_plate_detection.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    print(frame_nmr)
    if frame_nmr == 10:
        break

    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = get_vehicles(detections)

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # read license plate number
                license_plate_text, license_plate_text_score = get_license_plate_score(license_plate_crop)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    
# write results
write_csv(results, './test.csv')