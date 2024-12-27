#%%
import cv2
import numpy as np

# 1cm당 픽셀 수 (예: 1cm = 18픽셀)
PIXELS_PER_CM = 18

def sobel_edge_detection(frame):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러를 사용해 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel 필터를 사용하여 엣지 검출
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # X 방향 경계 검출
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Y 방향 경계 검출

    # 두 방향의 경계 검출 결과 결합
    sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)

    # 엣지 이미지를 8비트 형식으로 변환
    edged = cv2.convertScaleAbs(sobel_combined)
    min_threshold = 80
    _, edged = cv2.threshold(edged, min_threshold, 255, cv2.THRESH_BINARY)

        
        
        
    # 이진화 처리 (경계값 조정 가능)
    _, thresholded = cv2.threshold(edged, 50, 255, cv2.THRESH_BINARY)

    return thresholded

def measure_and_draw_boxes(frame, edged):
    # 외곽선 찾기
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area=0.0
    max_area_index=-1
    for i, contour in enumerate(contours):
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if area <  4: 
            continue
        if max_area < area :  
            max_area = area
            max_area_index = i
    
    if max_area_index != -1:
        contour = contours[max_area_index]
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        x, y = rect[0]
        w, h = rect[1]
        w = max(w, h)
        h = min(w, h) 
        ang = rect[2]
        PIXELS_PER_CM = int(w)/20
        
    for contour in contours:

        
        
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        # 각 외곽선에 대한 경계 박스 그리기
        
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)


        # 박스의 실제 크기 계산
        width_cm = w / PIXELS_PER_CM
        height_cm = h / PIXELS_PER_CM

        if width_cm <= 8 and height_cm <= 8:
            continue
        
        # 원본 이미지에 경계 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 텍스트 배경 박스 그리기 (가독성을 위해)
        text = f"{width_cm:.2f}cm x {height_cm:.2f}cm"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 0), cv2.FILLED)
        
        # 텍스트 추가 (크기, 두께, 색상 변경)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Sobel 필터를 사용한 엣지 검출
    edged_frame = sobel_edge_detection(frame)
    mask_frame = np.zeros(frame.shape, np.uint8)    
    
    contours, _ = cv2.findContours(edged_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        hull = cv2.convexHull(contour)
        cv2.drawContours(mask_frame, [hull], -1, (255, 255, 255), -1)
    
    #cv2.imshow("Sobel Edge", frame)
    #cv2.imshow("Sobel Edge", mask_frame)
    edged_out = frame & mask_frame
    #cv2.imshow("Sobel Edge", edged_out)
    #cv2.waitKey(0)
    

    # Copy the thresholded image.
    if False:
        im_floodfill = edged_frame.copy()    
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = edged_frame.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)    
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)    
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)    
        # Combine the two images to get the foreground.
        im_out = edged_frame | im_floodfill_inv       
        
        edged_frame = im_out
        
    cv2.imshow("Sobel Edge", edged_out)
    #cv2.waitKey(0)
    
    

    # 외곽선을 따라 객체 크기 측정 및 박스 그리기
    result_frame = measure_and_draw_boxes(frame, edged_frame)

    # 결과 프레임 출력
    cv2.imshow("Sobel Edge Detection with Size Measurement", result_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()
