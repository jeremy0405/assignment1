# EuclideanDistTracker class를 만들기 위해 math 모듈 호출
import math

#물체를 Track 하기 위해 EuclideanDistTracker class 생성
class EuclideanDistTracker:
    def __init__(self):
        # 중심 위치를 저장하는 center_points 생성 
        self.center_points = {}
        # 물체의 개수를 count 하기 위해 id_count 생성
        self.id_count = 0

    #update 함수 생성
    def update(self, objects_rect):
        # 물체의 boundary와 id를 저장하는 objects_bbs_ids 생성
        objects_bbs_ids = []

        # 새로운 물체의 중심 위치를 추출하는 코드
        for rect in objects_rect:
            #objects_rect에 저장되어 있는 픽셀의 가로값, 세로값, 너비, 높이를 x, y, w, h에 넣어줌
            x, y, w, h = rect
            # cx, cy는 물체의 중심 픽셀값을 나타냄
            # floor division을 이용하여 자연수의 값이 나오도록 함
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # same_object_detected 를 False로 정의함
            # same_object_detected 는 동일한 물체에 대해서는 ture값을 유지하고 새로운 물체에는 False값
            # same_object_detected를 통하여 새로운 물체인지 아닌지 판단이 가능
            same_object_detected = False
            
            
            # center_points.items()를 이용하여
            # key 값인 id_count와 값인 cx, cy를 각각 id와 pt에 대응하여 for문 수행  
            for id, pt in self.center_points.items():
                # dist는 math의 hypot 함수를 이용하여 물체가 이동한 길이를 나타냄
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    # 새로운 center_points를 넣어줌
                    self.center_points[id] = (cx, cy)
                    # 새로운 center_points를 print하여 출력
                    print(self.center_points)
                    # objects_bbs_ids에 픽셀의 가로, 세로, 너비, 높이, id값을 넣어줌
                    objects_bbs_ids.append([x, y, w, h, id])
                    # same_object_detected를 True로 해줌으로서 한번 검출된 물체라는 것을 인식시킴
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            # 새로운 물체가 감지되면 if 문이 실행됨
            if same_object_detected is False:
               # 물체의 고유 id값과 중심 위치를 저장함
               self.center_points[self.id_count] = (cx, cy)
               # objects_bbs_ids에 픽셀의 가로, 세로, 너비, 높이, id값을 넣어줌
               objects_bbs_ids.append([x, y, w, h, self.id_count])
               # 새로운 물체가 감지된 경우이므로 id_count를 1 높여
               # 다음에 들어올 새로운 물체의 id값과 현재 id값에 차이를 둠
               self.id_count += 1

        # center_points와 id를 초기화하는 new_center_points 생성
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # 사용하지 않는 id와 center_points를 제거하도록 하는 코드
        self.center_points = new_center_points.copy()
        
        # x, y, w, h, id 정보를 가지고 있는 objects_bbs_ids를 return함
        return objects_bbs_ids
