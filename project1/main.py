# 영상처리를 위해 OpenCV를 사용
import cv2
# tracker.py 에서 정의한 EuclideanDistTraker를 사용하기 위해 traker를 불러옴
from tracker import *

# tracker.py 파일에 정의한 EuclideanDistTracker Class를 tracker에 저장
# tracker를 통해 물체를 추적하는 것이 가능하게 함
tracker = EuclideanDistTracker()

# cap에 highway.mp4 의 파일 저장
cap = cv2.VideoCapture("highway.mp4")

# cv2에 createBackgroundSubtractorMOG2() 함수를 이용하여 물체를 detection 함
# createBackgroundSubtractorMOG2() 는 고정된 카메라에서 움직이는 물체만을 detection 함
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


while True:
    # highway.mp4 파일의 프레임별 이미지를 읽는 변수 ret과 frame 정의
    ret, frame = cap.read()
    
    # highway.mp4 파일의 높이와 넓이를 추출함
    height, width, _ = frame.shape

    # Region of interest를 지정함
    roi = frame[340: 720,500: 800]

    # mask에 roi에 해당하는 영역에서 object를 detection하도록 함
    # 영상을 보면 createBackgroundSubtractorMOG2()에서 흑백처리 한 후
    # 움직이는 물체만 흰색으로 표시한 것을 확인할 수 있음
    mask = object_detector.apply(roi)
    # threshold를 통하여 흑백 mask 에서 thresh값을 254, maxval 값을 255로 세팅함
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # 흑백의 mask를 통해 윤곽선을 검색하는 findContours 함수를 이용하여 윤곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 물체가 있는 사각형의 bound를 저장하는 변수 detections 정의
    detections = []
    for cnt in contours:
        # 물체의 윤곽선의 넓이를 area에 넣음
        area = cv2.contourArea(cnt)
        # area를 일정 크기 이상으로 설정하여 너무 작은 물체는 무시
        if area > 100:
            # drawContours를 통하여 검출한 외곽선을 그림
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            # 외곽선을 둘러싸는 bound를 x, y, w, h에 저장 
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 추출한 x, y, w, h를 detections에 저장
            detections.append([x, y, w, h])


    # tracker.py에서 정의한 update를 사용하여 물체를 track함
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        # roi영역에 id를 표시함 
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # roi영역에 rectagle를 이용하여 boundary를 그림
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    
    # roi와 frame과 mask 이미지 출력을 함
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # 특정 키를 누르면 정지되도록 함
    # 32는 ASCII 에서 Space bar를 뜻함
    key = cv2.waitKey(32)
    if key == 32:
        break
    
#cap 객체를 release하여 해제하고 모든 창을 끔
cap.release()
cv2.destroyAllWindows()