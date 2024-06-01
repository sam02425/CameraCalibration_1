
# Real-time Object Detection and Coordinate Transformation with Stereo Cameras

This project demonstrates real-time object detection and coordinate transformation using stereo cameras. It utilizes camera calibration, stereo calibration, and the YOLO object detection model to detect objects in stereo camera frames and transform their coordinates to a common world coordinate system.

## Requirements
- Python 3.x
- OpenCV
- NumPy

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLO model files:
   - `yolo.cfg`: YOLO configuration file
   - `yolo.weights`: YOLO pre-trained weights
   - `coco.names`: COCO dataset class names

   Place these files in the project directory.

## Usage
1. Prepare the calibration images:
   - Capture chessboard images from both the left and right cameras.
   - Place the left camera images in the `left_images/` directory.
   - Place the right camera images in the `right_images/` directory.

2. Run the camera calibration script:
   ```
   python camera_calibration.py
   ```
   This script will calibrate the left and right cameras individually and save the calibration parameters.

3. Run the stereo calibration script:
   ```
   python stereo_calibration.py
   ```
   This script will perform stereo calibration using the calibrated left and right cameras and save the stereo calibration parameters.

4. Run the main script for real-time object detection and coordinate transformation:
   ```
   python main.py
   ```
   This script will start the real-time object detection and coordinate transformation process using the stereo cameras. It will display the detected objects and their transformed coordinates.

5. Press 'q' to quit the program.

## Configuration
The `config.py` file contains various configuration parameters such as camera IDs, YOLO model paths, and calibration file paths. You can modify these parameters according to your setup.

## Calibration
The camera calibration and stereo calibration processes require chessboard images captured from both the left and right cameras. Make sure to capture a sufficient number of chessboard images from different angles and positions to ensure accurate calibration.

The calibration images should be placed in the `left_images/` and `right_images/` directories, respectively.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- The YOLO object detection model is based on the work by Joseph Redmon et al.
- The camera calibration and stereo calibration techniques are based on the OpenCV library.



## Here's a complete list of files and directories for project:

```
your-project/
├── camera_calibration.py
├── stereo_calibration.py
├── object_detection.py
├── config.py
├── utils.py
├── main.py
├── left_images/
│   ├── left_01.png
│   ├── left_02.png
│   ├── ...
├── right_images/
│   ├── right_01.png
│   ├── right_02.png
│   ├── ...
├── yolo.cfg
├── yolo.weights
├── coco.names
├── left_calibration.pkl
├── right_calibration.pkl
├── stereo_calibration.pkl
├── requirements.txt
└── README.md
```

Here's a description of each file and directory:

- `camera_calibration.py`: Python script for calibrating individual cameras (left and right) using chessboard images.
- `stereo_calibration.py`: Python script for performing stereo calibration using the calibrated left and right cameras.
- `object_detection.py`: Python script containing the object detection and coordinate transformation functions.
- `config.py`: Python script containing configuration parameters such as camera IDs, YOLO model paths, and calibration file paths.
- `utils.py`: Python script containing utility functions for loading calibration data and drawing bounding boxes on images.
- `main.py`: Python script for running the real-time object detection and coordinate transformation process using stereo cameras.
- `left_images/`: Directory containing chessboard images captured from the left camera for calibration.
- `right_images/`: Directory containing chessboard images captured from the right camera for calibration.
- `yolo.cfg`: YOLO configuration file.
- `yolo.weights`: YOLO pre-trained weights file.
- `coco.names`: COCO dataset class names file.
- `left_calibration.pkl`: Pickle file containing the calibration parameters for the left camera.
- `right_calibration.pkl`: Pickle file containing the calibration parameters for the right camera.
- `stereo_calibration.pkl`: Pickle file containing the stereo calibration parameters.
- `requirements.txt`: Text file specifying the required Python packages and their versions.
- `README.md`: Markdown file providing an overview of the project, installation instructions, usage guidelines, and other relevant information.

Note: The `left_images/` and `right_images/` directories should contain the chessboard images captured from the left and right cameras, respectively. The number of images and their filenames may vary depending on your specific setup.

Make sure to place all the Python scripts, configuration files, and calibration files in the root directory of your project. The YOLO model files (`yolo.cfg`, `yolo.weights`, and `coco.names`) should also be placed in the root directory.

The `requirements.txt` and `README.md` files provide information about the project dependencies and serve as documentation for users and contributors.

Remember to adjust the file paths and configuration parameters in the scripts according to your specific setup and requirements.
```

