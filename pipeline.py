import cv2
from utils import read_license_plate
vehicles = [2, 3, 5, 7]



def get_vehicles(detections):
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    return detections_

def get_license_plate_score(license_plate_crop):
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

    # read license plate number
    return read_license_plate(license_plate_crop_thresh)