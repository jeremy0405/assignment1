# 2. 자율주행 인지에 관련된 2종 이상 Open Source 조사, 정리
# 3. 2의 정리한 코드 중 하나 실행해서 결과 확인

project1 에서는 highway.mp4 파일에서 object를 detect한 후 tracking하는 코드이다.

코드 실행 방법은 Anaconda prompt 에서 cd <파일의 위치> 후 python main.py 를 입력하면 실행된다.

코드 설명은 각 main.py와 tracker.py 파일에 주석 처리로 설명하였다.
코드는 main.py, tracker.py, highway.mp4 파일만 있다면 실행이 되며
tracker.py 파일에서 object tracking하는 코드가 있으며
main.py 파일에는 object detect하는 코드가 담겨 있고 tracker.py파일을 통해 tracking을 수행하게 된다.


main.py에서 object detection을 위해 OpenCV의 createBackgroundSubtractorMOG2 함수를 이용하였다.

createBackgroundSubtractorMOG2는 background model과 currentframe을 비교하여 배경을 제거하여 객체만 남기는 방법이다. 

이 함수의 단점은 background가 동일하여야 한다는 것(고정된 카메라)이다.

따라서 highway.mp4 파일에서 바람등에 의한 노이즈로 화면이 흔들리게 되면 다음 그림과 같이 벽 또는 도로면을 tracking하는 엉뚱한 경우가 발생한다.

![noise](https://user-images.githubusercontent.com/81368630/113730392-d1443800-9732-11eb-896b-e55bbb27a15e.png)

카메라가 정확히 고정되어 있을 때는 정확한 tracking을 수행한다.
![nomal](https://user-images.githubusercontent.com/81368630/113730557-f5a01480-9732-11eb-939c-5d0865cf79d9.png)


tracker.py 에서 핵심은 dist 변수로서 math 모듈의 hypot 함수를 통해 detect된 물체가 이동한 거리를 통해 tracking을 수행한다.


