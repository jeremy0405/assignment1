# numpy 배열 사용
import numpy as np
# 영상처리 하기 위해 OpenCV 사용
import cv2
# Line.py 에서 정의한 함수 사용
from Line import Line


# roi 설정
def region_of_interest(img, vertices):

    # mask에 img에 맞춰 0의 값을 넣어 초기화하여 정의
    mask = np.zeros_like(img)

    # img의 채널에 따라(흑백사진 or 컬러사진) ignore_mask_color를 255로 정의
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # vertices에 정의된 좌표에 따라 mask에 255 color로 다각형 그림
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # bitwise_and 를 통해 img 와 mask가 0이 아닌 특정 영역만 masked_image에 저장
    masked_image = cv2.bitwise_and(img, mask)

    #  masked_image와 mask return
    return masked_image, mask


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):

    # OpenCV의 HoughLInesP를 이용하여 img파일에서 line detect를 함
    # HoughLinesp 는 이미지에서 직선을 검출하는 함수
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


# 두개의 이미지를 가중치를 두어 하나의 이미지로 변환하는 함수
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):

    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))
        
    # addWeighted 를 통해 initial_img * α + img * β + λ 한 이미지를 return
    return cv2.addWeighted(initial_img, α, img, β, λ)


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    도로의 차선에 가까운 line을 계산하는 함수    
    :param line_candidates: 허프 변환한 line
    :param img_shape: 허프 변환한 img shape
    :return: 왼쪽과 오른쪽 line에 가까운 line인 left_lane, right_lane return
    """

    # 기울기가 양수이면 pos_lines , 기울기가 음수이면 neg_lines 으로 분류
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]


    # bias와 slope를 이용하여 Line을 left line에 근접하게 개선
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    # bias와 slope를 이용하여 Line을 right line에 근접하게 개선
    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane


# get_lane_lines는 color_image로부터 차선을 추론하는 함수
def get_lane_lines(color_image, solid_lines=True):
    
    # color_image를 960 x 540로 변환
    color_image = cv2.resize(color_image, (960, 540))

    # color_image를 img_gray에 gray scale로 변환
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # GaussianBLur를 통하여 가우시안 블러처리
    # GaussianBLur 처리를 하면 노이즈를 제거하여 차선을 더 잘 검출 할 수 있음
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

    # Canny를 통하여 차선의 edge를 detect 함
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

    # 허프 변환을 통해 GaussianBlur와 Canny 처리가 된 이미지에서 직선을 추출
    detected_lines = hough_lines_detection(img=img_edge,
                                           rho=2,
                                           theta=np.pi / 180,
                                           threshold=1,
                                           min_line_len=15,
                                           max_line_gap=5)

    # 추출된 (x1, y1, x2, y2) tuples을 line으로 바꿈
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

    # solid_lines가 True인 경우
    if solid_lines:
        candidate_lines = []
        for line in detected_lines:
                # 기울기가 0.5 이상, 2이하인 직선만을 추출
                if 0.5 <= np.abs(line.slope) <= 2:
                    candidate_lines.append(line)
        # compute_lane_from_candidates 함수를 통해 도로의 차선에 가까운 line을 추출
        lane_lines = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    else:
        # solid_lines가 False인 경우 허프 변환을 통해 직선을 추출한 detected_lines으로부터 line 추출
        lane_lines = detected_lines

    # 추출한 line인 lane_lines를 return
    return lane_lines


# mean을 통하여 추출한 lane_lines를 부드럽게 만들어주는 함수
def smoothen_over_time(lane_lines):

    # avg_line_lt, avg_line_rt 초기화 후 정의
    avg_line_lt = np.zeros((len(lane_lines), 4))
    avg_line_rt = np.zeros((len(lane_lines), 4))

    # Line.py의 함수인 get.coords를 이용하여 avg_line_lt, avg_line_rt에 left_line, right_line 저장
    for t in range(0, len(lane_lines)):
        avg_line_lt[t] += lane_lines[t][0].get_coords()
        avg_line_rt[t] += lane_lines[t][1].get_coords()

    # left line의 mean값과, right line의 mean값을 통해 line을 부드럽게 만듬
    return Line(*np.mean(avg_line_lt, axis=0)), Line(*np.mean(avg_line_rt, axis=0))



# 최종적으로 위에서 정의한 함수들로 차선을 검출하고 차선과 이미지파일이 합쳐져 하나의 이미지 파일을 return하는 함수
def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):

    is_videoclip = len(frames) > 0

    # img_h, img_w로 높이와 너비 대입
    img_h, img_w = frames[0].shape[0], frames[0].shape[1]

    lane_lines = []
    for t in range(0, len(frames)):
        # 차선을 추론하는 함수인 get_lane_lines를 이용하여 inferred_lanes에 추론된 차선 저장
        inferred_lanes = get_lane_lines(color_image=frames[t], solid_lines=solid_lines)
        # lane_lines에 추론된 차선 저장
        lane_lines.append(inferred_lanes)

    if temporal_smoothing and solid_lines:
        # temporal_smoothing, solid_lines가 True이면 lane_lines를 smoothen_over_time함수를 통해 부드럽게 함
        lane_lines = smoothen_over_time(lane_lines)
    else:
        lane_lines = lane_lines[0]

    # line이 그려질 이미지 파일 line_img 정의 후 초기화
    line_img = np.zeros(shape=(img_h, img_w))

    # Line.py의 draw 함수를 이용하여 line_img에 lane을 그림
    for lane in lane_lines:
        lane.draw(line_img)

    # vertices 에 roi 영역의 범위를 넣어준 후 region_of_interest 함수를 통해 roi 영역 설정
    vertices = np.array([[(50, img_h),
                          (450, 310),
                          (490, 310),
                          (img_w - 50, img_h)]],
                        dtype=np.int32)
    img_masked, _ = region_of_interest(line_img, vertices)

    # weighted_img를 이용하여 컬러 이미지와 라인을 혼합하여 img_blend에 저장
    img_color = frames[-1] if is_videoclip else frames[0]
    img_blend = weighted_img(img_masked, img_color, α=0.8, β=1., λ=0.)
    
    # 컬러 이미지와 라인이 함께 있는 img_blend를 return하여 line detection 완료
    return img_blend

