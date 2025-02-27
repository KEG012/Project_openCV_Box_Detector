# Box Detect and Measure Size and OCR
- 택배 물류가 증가하고 상자의 다향함으로 인력으로 물체를 분류하는데 어려움이 있습니다. 이를 해결하기 위해 간단한 openCV를 사용한 방식으로 물류를 분류하고자 합니다.
- 이 프로젝트는 박스가 감지 영역안에 들어오면 박스의 색을 판단하여 구별할 박스를 찾아냅니다. 박스를 찾으면 박스의 크기를 측정하고 사이즈를 구분합니다. 박스에 글씨가 써 있다면 글씨를 인식하여 화면에 표시해줍니다.


# 프로젝트 개요
- 수행기간 : 2024.09.02 ~ 2024.09.06
- 구현 목표:
  - 특정한 크기의 박스를 사람이 측정하지 않고 분류할 수 있도록 함.
  - 실시간으로 분류가 이루어져야함
  - 박스에 문자가 기록되어 있을 시 이를 화면에 표기함.
<p align="center">
<img src="./box_detect_image_and_video/택배물류.jpg">
</p>

# 기술 스택
|기술          |설명                            |
|--------------|--------------------------------|
|**python**    | 주요 프로그래밍 언어            |
|**OpenCV**    | 이미지 처리                    |
|**OCR**       | 문자 인식                      |
|**QT**        | GUI 기반 사용자 인터페이스 구현  |

# 주요 기능

## 박스 감지
- 화면에 들어온 박스의 Edge를 판별하여 추출
- 빛으로 인한 Edge 인식이 어려운 부분을 위해 보정 작업 진행
- edge로 보정된 이미지를 frame의 박스와 결합하기 위하여 contour로 내부 채움
<p align="center">
<img src="./box_detect_image_and_video/binary image.PNG">
<img src="./box_detect_image_and_video/fill box.PNG">
<img src="./box_detect_image_and_video/replace box.PNG">
</p>

## 적합한 박스인지 판별 및 크기 측정
- 박스의 색을 HSV로 바꿔 특정한 색이 들어올 시 rectangle 표시
- 다른 박스가 들어올 시 rectangle 생성하지 않음
- 박스의 크기를 측정하여 이를 화면에 표시함
<p align="center">
<img src="./box_detect_image_and_video/HSV Range.PNG">
<img src="./box_detect_image_and_video/hsv roi image.PNG">
<img src="./box_detect_image_and_video/capture image.PNG">
</p>

## 박스의 글씨 인식
- 박스에 글씨가 있을 시 OCR을 사용하여 글씨를 인식
<p align="center">
<img src="./box_detect_image_and_video/capture_image_20240904_194452_with_text.png">
</p>

# 구현 결과
## 박스 판별 알고리즘
- 박스의 HSV값을 기준으로 판별하여 박스가 맞다면 X를 표시하여 추적함.
<p align="left">
<img src="./box_detect_image_and_video/box_select.gif">
</p>

## 박스 감지 및 크기 표시
- 박스가 감지되면 알고리즘을 기준으로 크기를 측정하고 이를 화면에 표시해 줌.
<p align="left">
<img src="./box_detect_image_and_video/box_detect.gif">
</p>
