# Real Time Vehicle Licence Plate Detection with Tracking


This project focuses on real-time vehicle and license plate detection, tracking, and recognition using Ultralytics YOLO (You Only Look Once) models and SORT (Simple Online and Realtime Tracking) algorithm.

## Overview

The primary goal of this project is to demonstrate the integration of YOLO models for vehicle and license plate detection along with SORT for tracking. The system processes a video feed, identifies vehicles, tracks them over time, and recognizes license plates.



## Installation
To set up the project locally, follow these steps:

```bash
git clone https://github.com/mallikarjuna-pokuri/Real-Time-Vehicle-Number-Plate-Detection-with-YOLOv8.git
cd Real-Time-Vehicle-Number-Plate-Detection-with-YOLOv8
pip install -r requirements.txt
```
- [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort)

## Project Structure

 **Frame Reading**
- The project begins by reading frames from the video file `sample.mp4` using the OpenCV library (`cv2.VideoCapture`).

 **Vehicle Detection**
- The frame data is processed using a YOLO (You Only Look Once) model (`YOLO('yolov8n.pt')`) to detect vehicles. 
- Detected vehicles are filtered based on predefined class IDs (2, 3, 5, 7).

**Vehicle Tracking**
- The Sort algorithm (`Sort()`) is employed for tracking vehicles across frames. 
- The tracking IDs are updated as new frames are processed.

**License Plate Recognition**
- Another YOLO model (`YOLO('./artifacts/license_plate_detector.pt')`) is utilized to detect license plates in each frame.
- License plate information is associated with the tracked vehicles using the `get_car` function.
- To know how I trained my YOLO model to detect license plates, please take a look at the ***license_plate_detection.ipynb*** in the repository

**Results**
- The results are stored in a ***test.csv*** with frame numbers as keys.
- For each tracked vehicle, the results include:
  - Vehicle information: Bounding box coordinates.
  - License plate information: Bounding box coordinates, text, score, and text score.

## Writing Results
- The results are written to a CSV file (`test.csv`) using the `write_csv` function.

# How to Run
- Ensure you have the necessary dependencies installed (`requirements.txt`).
- Run the script (`python main.py`) to write the results to test.csv
- Run the below script for interpolation of missing data
 ```bash
 python interpolate_missing_data.py
```
- Run the below script to save the results to (`output.mp4`)
 ```bash
 python visualize.py
```


## Acknowledgments
Ultralytics YOLO for powerful object detection models.
SORT algorithm for efficient online tracking.
OpenCV for computer vision and image processing.

## Contributing
Contributions are welcome! Follow these steps to contribute:


1. Fork the repository
2. Create a new branch (git checkout -b feature/improvement)
3. Make changes and commit (git commit -am 'Add feature')
4. Push to the branch (git push origin feature/improvement)
5. Create a pull request

## License
This project is licensed under the MIT License.

Feel free to reach out to mallikarjunapokuri595@gmail.com for any questions or concerns.

Happy coding! 🚀
