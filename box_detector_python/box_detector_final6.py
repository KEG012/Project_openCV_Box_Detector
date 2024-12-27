import sys
import cv2
import numpy as np
import math
import pytesseract
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
from datetime import datetime

# 1cm당 픽셀 수 초기화
PIXELS_PER_CM = 18

# Configure the path to the Tesseract executable if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def sobel_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)
    edged = cv2.convertScaleAbs(sobel_combined)
    _, edged = cv2.threshold(edged, 80, 255, cv2.THRESH_BINARY)
    _, thresholded = cv2.threshold(edged, 50, 255, cv2.THRESH_BINARY)
    return thresholded


def is_cal_box(frame):
    is_cal_box = False
    low_H = 7
    high_H = 64
    low_S = 45
    high_S = 212
    low_V = 0
    high_V = 255

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    max_area = 0.0
    max_area_index = -1
    for i, contour in enumerate(contours):
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if area < 4:
            continue
        if max_area < area:
            max_area = area
            max_area_index = i
    if max_area_index != -1:
        contour = contours[max_area_index]
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        x, y = rect[0]
        w, h = rect[1]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        dist = 50
        cv2.line(frame_HSV, (x-dist, y-dist), (x+dist, y+dist), (0, 0, 255), 5)
        cv2.line(frame_HSV, (x+dist, y-dist), (x-dist, y+dist), (0, 0, 255), 5)

        frame_roi = frame_HSV[y-10:y+10, x-10:x+10, :]

        w = max(w, h)
        h = min(w, h)
        ang = rect[2]
        global PIXELS_PER_CM
        PIXELS_PER_CM = int(w)/20

        roi_mean = cv2.mean(frame_roi)

        if roi_mean[1] > 20 and roi_mean[1] < 50:
            is_cal_box = True

    return is_cal_box


def measure_and_draw_boxes(frame, edged) -> cv2.Mat:
    global PIXELS_PER_CM
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0.0
    max_area_index = -1
    for i, contour in enumerate(contours):
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if area < 4:
            continue
        if max_area < area:
            max_area = area
            max_area_index = i

    if max_area_index != -1 and is_cal_box(frame):
        contour = contours[max_area_index]
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        x, y = rect[0]
        w, h = rect[1]
        w = max(w, h)
        h = min(w, h)
        ang = rect[2]

        box_B = 26
        box_B_w = 400
        box_M = 20
        box_M_w = 300
        box_S = 16
        box_S_w = 200

        if w > box_B_w:
            PIXELS_PER_CM = int(w)/box_B
        elif w > box_M_w and w < box_B_w:
            PIXELS_PER_CM = int(w)/box_M
        elif w > box_S_w and w < box_M_w:
            PIXELS_PER_CM = int(w)/box_S

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        width_cm = w / PIXELS_PER_CM
        height_cm = h / PIXELS_PER_CM
        box_area_cm2 = math.sqrt(width_cm**2 + height_cm**2)

        # 박스 분류
        if box_area_cm2 > 27:
            box_category = 'L'
        elif box_area_cm2 >= 22 and box_area_cm2 <= 27:
            box_category = 'M'
        elif box_area_cm2 < 22:
            box_category = 'S'

        if width_cm <= 8 and height_cm <= 8:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = f"{width_cm:.2f}cm x {height_cm:.2f}cm ({box_category})"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_height - baseline),
                      (x + text_width, y), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def extract_text_from_image(image: np.ndarray) -> str:
    height, width, _ = image.shape
    third_quadrant = image[height//2:, :width//2]
    gray_third_quadrant = cv2.cvtColor(third_quadrant, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_third_quadrant, config='--psm 6')
    return text.strip()


def draw_extracted_text(frame: np.ndarray, text: str) -> np.ndarray:
    height, width, _ = frame.shape
    cv2.putText(frame, text, (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Camera Display")
        self.setGeometry(100, 100, 1280, 800)

        self.initUI()

        self.cap1 = cv2.VideoCapture(0)  # 첫 번째 카메라
        self.cap2 = cv2.VideoCapture(1)  # 두 번째 카메라

        # Create a QTimer to update the frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.resize(1600, 1000)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        self.start_camera_btn = QPushButton("Camera Start")
        self.start_camera_btn.clicked.connect(self.start_camera)

        self.stop_camera_btn = QPushButton("Camera Stop")
        self.stop_camera_btn.clicked.connect(self.stop_camera)

        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_camera_btn)
        button_layout.addWidget(self.stop_camera_btn)
        button_layout.addWidget(self.capture_btn)

        layout = QHBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(button_layout)

        self.central_widget.setLayout(layout)

    def start_camera(self):
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()

    def capture(self) -> None:
        if hasattr(self, "current_frame"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_img_path = f"Capture_image/capture_image_{timestamp}.png"

            cv2.imwrite(output_img_path, self.current_frame)
            print(f"Image captured and saved as '{output_img_path}'")

            # Extract text from the third quadrant of the captured image
            extracted_text = extract_text_from_image(self.current_frame)

            # Load the captured image, overlay the text, and save the updated image
            capture_img = cv2.imread(output_img_path)
            capture_img_with_text = draw_extracted_text(
                capture_img, extracted_text)
            updated_output_img_path = f"Capture_image/capture_image_{
                timestamp}_with_text.png"
            cv2.imwrite(updated_output_img_path, capture_img_with_text)
            print(f"Updated image with text saved as '{
                  updated_output_img_path}'")

            cv2.imshow("Captured Image with Text", capture_img_with_text)

    def update_frame(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if not ret1 or not ret2:
            return

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        edged_frame1 = sobel_edge_detection(frame1)
        mask_frame1 = np.zeros(frame1.shape, np.uint8)

        contours1, _ = cv2.findContours(
            edged_frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask_frame1, [hull], -1, (255, 255, 255), -1)

        edged_out1 = cv2.bitwise_and(frame1, mask_frame1)
        result_frame1 = measure_and_draw_boxes(frame1, edged_frame1)

        edged_frame2 = sobel_edge_detection(frame2)
        mask_frame2 = np.zeros(frame2.shape, np.uint8)

        contours2, _ = cv2.findContours(
            edged_frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours2:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask_frame2, [hull], -1, (255, 255, 255), -1)

        edged_out2 = cv2.bitwise_and(frame2, mask_frame2)
        result_frame2 = measure_and_draw_boxes(frame2, edged_frame2)

        camera2_name_text = "CAMERA 2 (FRONT)"
        (text_width, text_height), baseline = cv2.getTextSize(
            camera2_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.rectangle(result_frame2, (frame2.shape[1] - text_width - 10, 0),
                      (frame2.shape[1], text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(result_frame2, camera2_name_text,
                    (frame2.shape[1] - text_width - 3, text_height + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        height, width, _ = frame1.shape
        combined_frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

        combined_frame[:height, :width] = edged_out1
        combined_frame[:height, width:] = result_frame1
        combined_frame[height:, :width] = edged_out2
        combined_frame[height:, width:] = result_frame2

        # Convert the frame to QImage for display
        height, width, channel = combined_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(combined_frame.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)

        # Display the frame on the QGraphicsView
        q_map = QPixmap.fromImage(q_img)

        self.scene.clear()
        pixmap_item = QGraphicsPixmapItem(q_map)
        self.scene.addItem(pixmap_item)

        self.current_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
