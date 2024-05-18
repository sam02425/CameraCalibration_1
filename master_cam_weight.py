import cv2
import torch
import numpy as np
import time
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from hx711 import HX711

class ShelfCheckoutDetector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.5, device='cpu'):
        self.model = attempt_load(model_path, map_location=torch.device(device))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.load_cells = self.initialize_load_cells()

    def initialize_load_cells(self):
        load_cells = []
        for i in range(4):
            try:
                load_cell = HX711(dout_pin=i*2, pd_sck_pin=i*2+1)
                load_cell.set_scale(1)
                load_cell.tare()
                load_cells.append(load_cell)
            except Exception as e:
                print(f"Error initializing load cell {i}: {str(e)}")
        return load_cells

    def detect_objects(self, image):
        try:
            results = self.model(image)
            detections = non_max_suppression(results, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
            return detections[0]
        except Exception as e:
            print(f"Error during object detection: {str(e)}")
            return []

    def process_frame(self, frame):
        detections = self.detect_objects(frame)
        detected_objects = []

        for detection in detections:
            xmin, ymin, xmax, ymax, conf, cls = detection
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            plot_one_box([xmin, ymin, xmax, ymax], frame, label=label, color=(0, 255, 0), line_thickness=2)
            detected_objects.append({'label': self.model.names[int(cls)], 'confidence': conf})

        return frame, detected_objects

    def run_detection(self, camera_ids, display=True):
        caps = []
        for camera_id in camera_ids:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"Failed to open camera {camera_id}")
                continue
            caps.append(cap)

        if not caps:
            print("No cameras available. Exiting.")
            return

        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if not frames:
                break

            start_time = time.time()
            processed_frames = []
            detected_objects_list = []

            for frame in frames:
                processed_frame, detected_objects = self.process_frame(frame)
                processed_frames.append(processed_frame)
                detected_objects_list.append(detected_objects)

            weights = []
            for load_cell in self.load_cells:
                try:
                    weight = load_cell.get_weight(5)
                    if weight < 0:
                        weight = 0
                    weights.append(weight)
                except Exception as e:
                    print(f"Error reading weight from load cell: {str(e)}")
                    weights.append(0)

            total_weight = sum(weights)

            fps = 1 / (time.time() - start_time)

            if display:
                for i, processed_frame in enumerate(processed_frames):
                    cv2.putText(processed_frame, f"Camera {camera_ids[i]} - FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Weight: {total_weight:.2f} g", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f"Shelf Checkout - Camera {camera_ids[i]}", processed_frame)

            self.process_detected_objects_and_weight(detected_objects_list, total_weight)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    def process_detected_objects_and_weight(self, detected_objects_list, total_weight):
        try:
            # Implement your shelf checkout logic here
            for detected_objects in detected_objects_list:
                for obj in detected_objects:
                    label = obj['label']
                    confidence = obj['confidence']
                    # Perform actions based on the detected objects and weight
                    print(f"Detected object: {label} (Confidence: {confidence:.2f}), Total Weight: {total_weight:.2f} g")
        except Exception as e:
            print(f"Error processing detected objects and weight: {str(e)}")

if __name__ == '__main__':
    model_path = 'path/to/yolov5/model.pt'
    conf_thres = 0.5
    iou_thres = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    camera_ids = [0, 2, 4]

    detector = ShelfCheckoutDetector(model_path, conf_thres, iou_thres, device)
    detector.run_detection(camera_ids, display=True)