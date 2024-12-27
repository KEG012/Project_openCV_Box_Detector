# %%
import cv2
import numpy as np
import math

# 1cm당 픽셀 수 초기화
PIXELS_PER_CM = 18


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
        if box_area_cm2 >= 22 and box_area_cm2 <= 27:
            box_category = 'M'
        if box_area_cm2 < 22:
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


cap1 = cv2.VideoCapture(0)  # 첫 번째 카메라
cap2 = cv2.VideoCapture(1)  # 두 번째 카메라

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    edged_frame1 = sobel_edge_detection(frame1)
    mask_frame1 = np.zeros(frame1.shape, np.uint8)

    contours1, _ = cv2.findContours(
        edged_frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours1):
        hull = cv2.convexHull(contour)
        cv2.drawContours(mask_frame1, [hull], -1, (255, 255, 255), -1)

    edged_out1 = frame1 & mask_frame1
    result_frame1 = measure_and_draw_boxes(frame1, edged_frame1)

    rows1, cols1, _ = frame1.shape
    
    cv2.rectangle(edged_out1, (0,0), (cols1, rows1), (255,255,255), 1)
    cv2.rectangle(result_frame1, (0,0), (cols1, rows1), (255,255,255), 1)
    
    camera1_name_text = "CAMERA 1 (TOP)"
    (text_width, text_height), baseline = cv2.getTextSize(
        camera1_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(result_frame1, (cols1 - text_width - 10, 0),
                  (cols1, text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(result_frame1, camera1_name_text, (cols1 - text_width - 3,
                text_height + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    edged_frame2 = sobel_edge_detection(frame2)
    mask_frame2 = np.zeros(frame2.shape, np.uint8)

    contours2, _ = cv2.findContours(
        edged_frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours2):
        hull = cv2.convexHull(contour)
        cv2.drawContours(mask_frame2, [hull], -1, (255, 255, 255), -1)

    edged_out2 = frame2 & mask_frame2
    result_frame2 = measure_and_draw_boxes(frame2, edged_frame2)

    rows2, cols2, _ = frame2.shape

    cv2.rectangle(edged_out2, (0,0), (cols2, rows2), (255,255,255), 1)
    cv2.rectangle(result_frame2, (0,0), (cols2, rows2), (255,255,255), 1)
    
    camera2_name_text = "CAMERA 2 (FRONT)"
    (text_width, text_height), baseline = cv2.getTextSize(
        camera2_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(result_frame2, (cols2 - text_width - 10, 0),
                  (cols2, text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(result_frame2, camera2_name_text, (cols2-text_width-3,
                text_height+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    height, width, _ = frame1.shape
    combined_frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    combined_frame[:height, :width] = edged_out1
    combined_frame[:height, width:] = result_frame1
    combined_frame[height:, :width] = edged_out2
    combined_frame[height:, width:] = result_frame2
    
    cv2.imshow("Combined Result", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
# %%
import cv2
import numpy as np
import math

# 1cm당 픽셀 수 초기화
PIXELS_PER_CM = 18


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
        if box_area_cm2 >= 22 and box_area_cm2 <= 27:
            box_category = 'M'
        if box_area_cm2 < 22:
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


cap1 = cv2.VideoCapture(0)  # 첫 번째 카메라
cap2 = cv2.VideoCapture(1)  # 두 번째 카메라

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    edged_frame1 = sobel_edge_detection(frame1)
    mask_frame1 = np.zeros(frame1.shape, np.uint8)

    contours1, _ = cv2.findContours(
        edged_frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours1):
        hull = cv2.convexHull(contour)
        cv2.drawContours(mask_frame1, [hull], -1, (255, 255, 255), -1)

    edged_out1 = frame1 & mask_frame1
    result_frame1 = measure_and_draw_boxes(frame1, edged_frame1)

    rows1, cols1, _ = frame1.shape
    
    cv2.rectangle(edged_out1, (0,0), (cols1, rows1), (255,255,255), 1)
    cv2.rectangle(result_frame1, (0,0), (cols1, rows1), (255,255,255), 1)
    
    camera1_name_text = "CAMERA 1 (TOP)"
    (text_width, text_height), baseline = cv2.getTextSize(
        camera1_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(result_frame1, (cols1 - text_width - 10, 0),
                  (cols1, text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(result_frame1, camera1_name_text, (cols1 - text_width - 3,
                text_height + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    edged_frame2 = sobel_edge_detection(frame2)
    mask_frame2 = np.zeros(frame2.shape, np.uint8)

    contours2, _ = cv2.findContours(
        edged_frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours2):
        hull = cv2.convexHull(contour)
        cv2.drawContours(mask_frame2, [hull], -1, (255, 255, 255), -1)

    edged_out2 = frame2 & mask_frame2
    result_frame2 = measure_and_draw_boxes(frame2, edged_frame2)

    rows2, cols2, _ = frame2.shape

    cv2.rectangle(edged_out2, (0,0), (cols2, rows2), (255,255,255), 1)
    cv2.rectangle(result_frame2, (0,0), (cols2, rows2), (255,255,255), 1)
    
    camera2_name_text = "CAMERA 2 (FRONT)"
    (text_width, text_height), baseline = cv2.getTextSize(
        camera2_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(result_frame2, (cols2 - text_width - 10, 0),
                  (cols2, text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(result_frame2, camera2_name_text, (cols2-text_width-3,
                text_height+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    height, width, _ = frame1.shape
    combined_frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    combined_frame[:height, :width] = edged_out1
    combined_frame[:height, width:] = result_frame1
    combined_frame[height:, :width] = edged_out2
    combined_frame[height:, width:] = result_frame2
    
    cv2.imshow("Combined Result", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
