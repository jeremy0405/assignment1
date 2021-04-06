# 2. 자율주행 인지에 관련된 2종 이상 Open Source 조사, 정리
# 3. 2의 정리한 코드 중 하나 실행해서 결과 확인

project2 에서는 data/test_images와 data/test_videos 의 이미지와 비디오 파일에서
line_detection 한 새로운 이미지를 out/images , out/videos 에 저장하는 코드이다.

코드 실행 방법은 Anaconda prompt 에서 cd <파일의 위치> 후 python main.py를 입력하면 실행된다.
코드는 data/test_images 에 이미지 파일과 data/test_videos에 비디오 파일, lane_detection.py, Line.py, main.py 파일과 out/images, out/videos 폴더로 구성된다.


코드 설명은 "인지"에 해당되는 lane_detection.py 에 주석처리하여 설명하였다.

line detection 을 하기 위해 이미지에 가우시안 블러(cv2.GaussianBlur)와 Canny edge detection(cv2.Canny)을 한 후 허프변환(cv2.HoughLinesP)을 통해 직선을 검출하였다.

가우시안 블러란 중심에 있는 픽셀에 높은 가중치를 둔 후 본래의 이미지와 컨볼루션 연산을 하는 것이다.
가우시안 블러 처리를 하면 이미지의 노이즈를 제거할 수 있다. Canny Edge Detection을 하기 전 거의 필수적으로 하는 작업이다.

아래의 이미지는 (5,5) 사이즈의 가우시안 블러이며 lane_detection.py에서는  (17,17) 의 가우시안 블러를 사용하였다.
![gaussianblur](https://user-images.githubusercontent.com/81368630/113736706-49f9c300-9738-11eb-88a7-04a17006382d.gif)


Canny edge detection은 edge를 검출하는 방법이다.
수평방향의 gradient와 수직방향의 gradient를 구한 후 특정한 구역에서 gradient의 최대값을 가진 픽셀만 남긴 후 나머지는 0으로 처리하는 것을 반복하여 전 영역에 걸쳐 edge의 후보를 추린 후 minVal과 maxVal 값을 통해 edge 후보 중 진짜 edge라고 판단되는 부분만을 남겨놓게 된다. 


허프변환이란 이미지에서 직선을 검출하는 방법으로 원리는 다음과 같다.

한 평면에서 원소 (x1, y1)을 지나는 모든 직선은 a(x-x1)+b(y-y1) = 0 을 성립한다.
따라서 위의 식에 약간의 변화를 주어 (a/(a^2+b^2)^(1/2)) * (x-x1) + (b/(a^2+b^2)^(1/2)) * (y-y1) = 0 이며 이는 xcosθ + ysinθ = r 로 표현이 가능하다.

x y 평면에서 (x1, y1)한 점을 지나는 xcosθ + ysinθ = r 식은 θ 와 r은 상수인 식이지만
r θ 평면에서 (x1, y1)한 점을 지나는 xcosθ + ysinθ = r 식은 x 와 y가 상수이며  θ와 r이 변수가 된다.
이를 이용하여 r θ 평면에서 x1cosθ + y1sinθ = r 과 x2cosθ + y2sinθ = r 이 만나는 교점이 있다면
(x1, y1)과 (x2, y2)를 잇는 직선이 존재한다는 것을 의미하게 된다. 따라서 허프변환은 Canny 변환된 이미지의 edge 주변의 모든픽셀에대해서 xcosθ + ysinθ = r 에 대한 변수 θ와 r에 대한 그래프를 그리고 교점이 있는 경우에만 추출을 하여 직선을 추출하는 원리를 갖고 있다. 


project2 실행 결과는 다음과 같다.
![solidWhiteRight](https://user-images.githubusercontent.com/81368630/113741882-09507880-973d-11eb-8c9e-897b19a6d3ed.jpg)
![whiteCarLaneSwitch](https://user-images.githubusercontent.com/81368630/113741959-179e9480-973d-11eb-96b3-356206f0a46a.jpg)




