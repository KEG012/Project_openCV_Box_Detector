# %%
import sys
import cv2
import numpy as np
import math
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
from datetime import datetime

# frame으로 들어오는 데이터를 gray scale로 변경하여 gaussianblur로 noise 제거
# gray scale 데이터를 sobel 필터를 사용하여 엣지 추출
# edge의 위치를 더 명확하게 표현하기 위하여 edge강도 계산하여 x와 y 의 sobel 데이터를 합침
# 합쳐진 sobel 데이터를 64비트 float형에서 8비트 int로 변환하여 edged에 저장
# cv2.threshold 함수를 사용하여 데이터를 2진화 시킴.
# return value로 threshholded를 반환

def sobel_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)
    edged = cv2.convertScaleAbs(sobel_combined)
    _, thresholded = cv2.threshold(edged, 50, 255, cv2.THRESH_BINARY)
    return thresholded

# HSV 데이터를 이용하여 특정한 색을 가지는 박스만 검사 진행.


def is_cal_box(frame):
    is_cal_box = False
    low_H = 7
    high_H = 64
    low_S = 45
    high_S = 212
    low_V = 0
    high_V = 255

    # 영상에서 받아온 데이터를 BGR을 HSV로 변환, 지정된 값에서만 검사를 진행할 수 있도록 범위 지정.
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    # contours에 frame_threshold 값의 경계를 찾아 저장
    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0.0
    max_area_index = -1

    # contours에서 i값과 contour값을 추출하여 for문 진행.
    for i, contour in enumerate(contours):
        # cv2.convexHull 함수를 사용하여 contour의 경계선을 꼭지점을 지정하여 영역 구성
        hull = cv2.convexHull(contour)
        # hull 윤곽선의 면적을 계산
        area = cv2.contourArea(hull)
        # 면적이 4보다 작으면 진행
        if area < 4:
            continue
        # max_area가 area 보다 작으면 max_area에 area값 저장, max_area_index에 i 값 대입.
        if max_area < area:
            max_area = area
            max_area_index = i
    # max_area_index가 -1이 아니라면 contour값에 contours[max_area_index] 값을 대입.
    if max_area_index != -1:
        contour = contours[max_area_index]
        # cv2.minAreaRect로 가장 작은 면적의 값을 반환 받아서 저장. 값을 튜플로 저장됨.
        rect = cv2.minAreaRect(contour)
        x, y = rect[0]
        w, h = rect[1]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # 박스를 찾아 물체의 중간을 찾아 감지.
        dist = 50
        cv2.line(frame_HSV, (x-dist, y-dist), (x+dist, y+dist), (0, 0, 255), 5)
        cv2.line(frame_HSV, (x+dist, y-dist), (x-dist, y+dist), (0, 0, 255), 5)

        frame_roi = frame_HSV[y-10:y+10, x-10:x+10, :]

        w = max(w, h)
        h = min(w, h)

        # roi_mean 값을 cv2.mean(frame_roi)를 사용하여 평균값 대입.
        roi_mean = cv2.mean(frame_roi)

        # roi_mean의 값이 20보다 크거나 50보다 작으면 is_cal_box에 True를 대입
        if roi_mean[1] > 20 and roi_mean[1] < 50:
            is_cal_box = True

        # is_cal_box를 반환
    return is_cal_box

# frame과 edged를 받아서 frame을 return 하는 함수


def measure_and_draw_boxes(frame, edged):
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

    # max_area_index가 -1이 아니고 is_cal_box(frame)의 값이 True일 경우 실행.
    if max_area_index != -1 and is_cal_box(frame):
        contour = contours[max_area_index]
        rect = cv2.minAreaRect(contour)
        x, y = rect[0]
        w, h = rect[1]
        w = max(w, h)
        h = min(w, h)

        # 각 박스마다 값을 초기화
        box_B = 26
        box_B_w = 400
        box_M = 20
        box_M_w = 300
        box_S = 16
        box_S_w = 200

        # 범위에 따라서 픽셀 당 cm값을 정하는 조건을 만듦.
        if w > box_B_w:
            PIXELS_PER_CM = int(w)/box_B
        elif w > box_M_w and w < box_B_w:
            PIXELS_PER_CM = int(w)/box_M
        elif w > box_S_w and w < box_M_w:
            PIXELS_PER_CM = int(w)/box_S

    # contours에서 contour의 값을 추출하여 area에 대입
    for contour in contours:
        area = cv2.contourArea(contour)  # contourArea() = 지정 값의 면적을 계산하는 함수.
        if area < 100:  # area가 100보다 작으면 진행.
            continue
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # 조건에 맞춰 박스의 크기를 cm로 계산
        width_cm = w / PIXELS_PER_CM
        height_cm = h / PIXELS_PER_CM

        # 박스를 크기로 분류하기 위해서 삼각법을 통하여 대각선의 길이 계산
        box_area_cm2 = math.sqrt(width_cm**2 + height_cm**2)

        # 박스 분류
        if box_area_cm2 > 27:
            box_category = 'L'
        if box_area_cm2 >= 22 and box_area_cm2 <= 27:
            box_category = 'M'
        if box_area_cm2 < 22:
            box_category = 'S'

        # 박스의 크기가 8cm 이하인 것은 검출하지 않음 (예외처리)
        if width_cm <= 8 and height_cm <= 8:
            continue

        # 검출된 박스에 맞춰 사각 frame 생성.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 검출된 박스의 크기를 text로 만들어 이미지에 text를 입힘.
        text = f"{width_cm:.2f}cm x {height_cm:.2f}cm ({box_category})"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_height - baseline),
                      (x + text_width, y), (0, 0, 0), cv2.FILLED)

        cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # frame 값을 반환
    return frame

# PySide6를 사용하여 window 창 구현.


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Camera Display")
        self.setGeometry(100, 100, 1280, 800)

        self.initUI()

        # 각각의 camera로부터 데이터를 받아옴.
        self.cap1 = cv2.VideoCapture(0)  # 첫 번째 카메라
        self.cap2 = cv2.VideoCapture(1)  # 두 번째 카메라

        # frame 업데이트를 위해서 QTimer 함수를 사용.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # self.timer.start(30)  # 30ms 마다 업데이트

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.resize(1600, 1000)

        # 화면에 영상을 출력하기 위한 구문. self.scene을 받아와 view에 전달
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # 버튼 생성 및 명명. 클릭시 진행할 함수 연결
        self.start_camera_btn = QPushButton("Camera Start")
        self.start_camera_btn.clicked.connect(self.start_camera)

        self.stop_camera_btn = QPushButton("Camera Stop")
        self.stop_camera_btn.clicked.connect(self.stop_camera)

        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture)

        # button layout 설정. 버튼은 수직으로 설정되어야 하므로 QVBoxLayout()사용
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_camera_btn)
        button_layout.addWidget(self.stop_camera_btn)
        button_layout.addWidget(self.capture_btn)

        # window layout 설정. 각 창은 수평으로 설정되어야 하므로 QHBoxLayout()사용
        layout = QHBoxLayout()
        layout.addWidget(self.view)
        # 버튼을 추가할때는 addWidget, window를 추가할 때는 addLayout()
        layout.addLayout(button_layout)

        # 각 위젝을 가운데 정렬로 맞춤.
        self.central_widget.setLayout(layout)

    # start를 눌러야지 촬영 시작
    def start_camera(self):
        self.timer.start(30)

    # stop을 누르면 화면 clear
    def stop_camera(self):
        self.timer.stop()
        self.scene.clear()

    # capture를 하기 위한 함수
    def capture(self) -> None:
        if hasattr(self, "current_frame"):
            # capture가 저장될 때 이름에 날짜가 들어가기 위해서 datetime lib를 사용
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_img_path = f"Capture_image/capture_image_{timestamp}.png"

            # self.current_frame을 받아와 output_img_path에 저장.
            cv2.imwrite(output_img_path, self.current_frame)
            print(f"image captured and saved as '{output_img_path}'")

            # out_img_path에 있는 파일을 불러옴, cv2.IMREAD_COLOR로 컬러로 이미지를 읽어옴.
            # 기본적으로 BRG로 읽어옴.
            capture_img = cv2.imread(output_img_path, cv2.IMREAD_COLOR)
            cv2.imshow("Captured Image", capture_img)

    def update_frame(self):
        # cv2.captureVideo() 함수에서 ret1값과 frame1 값을 읽어옴. ret1은 bool값()
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        # ret1과 ret2 중 하나라도 flase 값이 들어오면 반환.
        if not ret1 or not ret2:
            return

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # sobel_edge_detection()을 통하여 frame에서 받아들인 값의 edge를 검출 (return threshold)
        edged_frame1 = sobel_edge_detection(frame1)
        mask_frame1 = np.zeros(frame1.shape, np.uint8)

        # cv2.findContours()를 통하여 contours의 값을 추출.
        contours1, _ = cv2.findContours(
            edged_frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask_frame1, [hull], -1, (255, 255, 255), -1)

        # frame1과 mask_frame 간의 &연산 진행. frame1은 현재 이미지, mask_frame은 인식된 박스만 binary 값으로 가지고 있음.
        edged_out1 = frame1 & mask_frame1
        result_frame1 = measure_and_draw_boxes(frame1, edged_frame1)

        # frame1.shape로 frame1의 배열의 rows와 cols를 가져옴.
        rows1, cols1, _ = frame1.shape

        # 전체 테두리 생성.
        cv2.rectangle(edged_out1, (0, 0), (cols1, rows1), (255, 255, 255), 1)
        cv2.rectangle(result_frame1, (0, 0),
                      (cols1, rows1), (255, 255, 255), 1)

        # 실제 영상의 오른쪽 테두리 끝에 이름 블록 생성.
        camera1_name_text = "CAMERA 1 (TOP)"
        (text_width, text_height), baseline = cv2.getTextSize(
            camera1_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.rectangle(result_frame1, (cols1 - text_width - 10, 0),
                      (cols1, text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(result_frame1, camera1_name_text, (cols1-text_width-3,
                                                       text_height+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 위의 frame1 과 동일.
        edged_frame2 = sobel_edge_detection(frame2)
        mask_frame2 = np.zeros(frame2.shape, np.uint8)

        contours2, _ = cv2.findContours(
            edged_frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours2:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask_frame2, [hull], -1, (255, 255, 255), -1)

        edged_out2 = frame2 & mask_frame2
        result_frame2 = measure_and_draw_boxes(frame2, edged_frame2)

        rows2, cols2, _ = frame2.shape

        cv2.rectangle(edged_out2, (0, 0), (cols2, rows2), (255, 255, 255), 1)
        cv2.rectangle(result_frame2, (0, 0),
                      (cols2, rows2), (255, 255, 255), 1)

        camera2_name_text = "CAMERA 2 (FRONT)"
        (text_width, text_height), baseline = cv2.getTextSize(
            camera2_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.rectangle(result_frame2, (cols2 - text_width - 10, 0),
                      (cols2, text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(result_frame2, camera2_name_text, (cols2-text_width-3,
                                                       text_height+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # 여기까지 frame1과 동일.

        # frame1의 높이와 길이를 받아옴.
        height, width, _ = frame1.shape
        # height와 width의 두 배의 행렬을 생성, ch은 3, 원소 형태는 uint8
        combined_frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

        # combined_frame을 4분면으로 나눠 각각의 영상을 대입함. 리스트 안의 변수에 주의할 것.
        combined_frame[:height, :width] = edged_out1
        combined_frame[:height, width:] = result_frame1
        combined_frame[height:, :width] = edged_out2
        combined_frame[height:, width:] = result_frame2

        # 화면에 표기하기 위해서 combined_frame을 q_img로 변경
        height, width, _ = combined_frame.shape
        bytes_per_line = 3 * width  # 한줄 당 사용되는 바이트 수.
        # combined_frame.data는 픽셀 데이터를 나타내는 포인터
        # QImage.Format_RGB888은 이미지를 RGB로 표현하는 것으로 R,G,B 당 8비트로 구성된다는 의미
        # QImage는 객체
        q_img = QImage(
            combined_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # q_img를 바탕으로 새로운 객체를 생성하는 class method.
        q_map = QPixmap.fromImage(q_img)

        # scene 초기화.
        self.scene.clear()
        # q_map의 객체를 바탕으로 pixmap_item 객체를 생성.
        pixmap_item = QGraphicsPixmapItem(q_map)
        # self.scene에 pixmap_item 객체를 추가함.
        self.scene.addItem(pixmap_item)

        # capture를 위하여 combined_frame을 RGB로 변경하여 self.current_frame에 대입.
        self.current_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

    # 창이 닫힐 때 작업을 정의하는 method.
    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()
        event.accept()


def main():
    # PySide에서 반드시 하나만 생성되어야 하는 클래스.
    # GUI application의 전반적인 관리와 주요기능(이벤트 루프, 윈도운 생성 및 관리)을 담당
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    # 이 함수가 실행되면 반환하여 종료
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
