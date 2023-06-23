# Vision
import cv2
import numpy as np
import sklearn
from skimage import measure

# Torch
import torch
import torch.nn.functional as F
import torchvision.transforms as T


import os
import shutil
import math
from queue import Queue

# Personal
from FANet import FANet
from UNet import UNet
from utils import color_mapping, reset_and_creat_folder

# models =['unet',
#          'aunet',
#          'fanet']


color_map = np.array([
    [0, 0, 0],       # 0 배경 (검정색)  : background
    [128, 0, 0],     # 클래스 1 (어두운 빨강) : avalanche
    [0, 128, 0],     # 클래스 2 (어두운 초록) : building_undamaged
    [128, 128, 0],   # 클래스 3 (어두운 옥색) : building_damaged
    [0, 0, 128],     # 클래스 4 (어두운 파랑) : cracks/fissure/subsidence
    [128, 0, 128],   # 클래스 5 (어두운 보라) : debris/mud/rock/flow
    [0, 128, 128],   # 클래스 6 (어두운 시안) : fire/flare
    [128, 128, 128], # 클래스 7 (회색) : flood/water/river/sea
    [64, 0, 0],      # 클래스 8 (진한 빨강) : ice_jam_flow
    [192, 0, 0],     # 클래스 9 (밝은 빨강) : lava_flow
    [64, 128, 0],    # 클래스 10 (노란) : person
    [192, 128, 0],   # 클래스 11 (주황) : pyroclastic_flow
    [64, 0, 128],    # 클래스 12 (보라) : road/railway/bridge
    [192, 0, 128],   # 클래스 13 (분홍) : vehicle
])

def color_map_rgb2bgr(color_map):
    bgr_color_map = []
    for color in color_map:
        # color.reverse()
        bgr_color_map.append(np.flip(color))
    return bgr_color_map

bgr_color_map = color_map_rgb2bgr(color_map)

class_map =np.array(['background',
                        'avalanche',
                        'building_undamaged',
                        'building_damaged',
                        'cracks/fissure/subsidence',
                        'debris/mud/rock_flow',
                        'fire/flare',
                        'flood/water/river/sea',
                        'ice_jam_flow',
                        'lava_flow',
                        'person',
                        'pyroclastic_flow',
                        'road/railway/bridge',
                        'vehicle'])

# 초기화
input_dirs = ['../disaster_video', '../clipped_video']
input_dir = input_dirs[1]
model_dir = './pkl_models'
output_dir = './segmented_video'

# 모델 선택 및 경로 설정
what_model = 'FANet.pkl'
model_names = os.path.join(output_dir)
for model_name in model_names:
    if model_name == what_model:
        break
model_path = os.path.join(model_dir, what_model)
frame_dir = './frames'

# object pixel 면적 threshold
OBJECT_AREA_TRHESHOLD = 100

# 몇번째 frame 마다 class 라벨링을 붙여줄지
FRAME_TEXT_LABELING_TRHESHOLD = 3

# LABEL BOX 의 좌표 비교 threshold
LABEL_BOX_THRESHOLD = 200

# Frame per second
FPS = 30

# segmentatoin 을 수행할 video 들
video_names = os.listdir(input_dir)

# W, H
output_size = (512, 512)

reset_and_creat_folder(output_dir)
reset_and_creat_folder(frame_dir)

# modle 불러오기
# model: UNet = UNet()
model: FANet = FANet()
model.load_state_dict(
                state_dict=torch.load(model_path,
                                      map_location=torch.device("cuda")),
                strict=False
        )
model.eval()

# 입력 영상 전처리를 위한 변환기 설정
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 현재 frame 에서 obejct 계산해서 (class, centroid, area) 리스트를 리턴한다.
# frame (512, 512), centroid (row, col)
def labeling(frame):
    # print(frame)
    frame_properties = measure.regionprops(frame)
    objects_info = []
    # print(f'object_cout={len(frame_properties)}')
    for object in frame_properties:
        # label object.label
        # 중심 좌표 object.centroid
        # 면적 object.area: 작은 면적은 박스 만들지 않도록 (만약)
        objects_info.append((object.label, object.centroid, object.area))
    return objects_info


# 영상 Segmentation 을 위한 함수 정의
def segment_image(image, use_cuda=True, idx=0, model_type='UNet'):
    # 영상 전처리
    # x=torch.Size([1, 3, 512, 512])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # GPU를 사용할 경우
    if torch.cuda.is_available() and use_cuda:
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Segmentation 수행
    with torch.no_grad():
        output = model(input_batch)

    if model_type == 'FANet':
        mask: torch.Tensor = F.interpolate(output, [512, 512],
                                                mode="bilinear",
                                                align_corners=True)
        mask: np.ndarray = mask.cpu().data.max(1)[1].numpy()
        mask: np.ndarray = mask.astype(np.uint8)
        mask: np.ndarray = np.squeeze(mask, axis=0)
    else:
        output_predictions = output.argmax(dim=1)
        # Segmentation 결과를 이미지로 변환
        mask = output_predictions.cpu().squeeze().numpy().astype(np.uint8)
    return mask


def set_centroid_compare_with_prev(current, class_centroids):
    for centroid in class_centroids:
        distance = math.sqrt((centroid[0] - current[0])**2 + (centroid[1] - current[1])**2)
        # 가까운 값이 있다면 이전 좌표로 설정
        if distance < LABEL_BOX_THRESHOLD:
            return centroid
    # 없다면 현재 좌표로 설정하고 추가
    class_centroids.append(current)
    return current


# 영상을 실시간으로 Segmentation하는 함수 정의
# 마스크를 원래 이미지에 적용하여 Segmentation 결과 시각화
def realtime_segmentation(path, is_imshow=False, output_video=None, frame_path=None):
    # 웹캠에서 영상을 받아오기 위한 비디오 캡처 객체 생성
    capture = cv2.VideoCapture(path)

    # frame 번호
    idx = 0

    # video 파일 하나가 가지고 있는 object의 좌표들
    #   labeling 좌표 설정에 사용
    #   해당하는 클래스의 이전 좌표와 비교해서 위치 실정
    objects_centroids = []
    for i in range(14):
        objects_centroids.append([])

    while True:
        # 영상 프레임 읽기
        ret, original_frame = capture.read() # frame=np.ndarray

        # 영상 끝
        if not ret:
            print(f'DONE={path}')
            break
        # print(objects_centroids)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        # print(f'original_frame={original_frame.shape}')

        resized_frame = cv2.resize(original_frame, dsize=output_size, interpolation=cv2.INTER_AREA)
        # print(f'resized_frame={resized_frame.shape}')

        # frame 에 대해서 segmentation 예측 수행
        # Return: color mapped frame
        mask = segment_image(resized_frame, False, idx, model_type='FANet')

        # mask 에서 object 의 정보 계산 => class 정보를 표시하기 위해서 사용
        objects_info = labeling(mask)

        # mask to colored mask (RGB 순서)
        segmented_frame = color_mapping(mask).astype(np.uint8)

        # RGB to cv2 format(BGR)
        segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)

        # frame별 mask 이미지 파일(RGB) 생성
        cv2.imwrite(os.path.join(frame_path, f'{idx}.png'), segmented_frame)
        # print(f'frame={idx}')

        # frame 번호
        idx += 1

        if is_imshow:
            # Segmentation 결과 출력
            cv2.imshow('Segmented Frame', segmented_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Segmentation된 영상과 원본 영상 블렌딩

            # conditon True: x(resized_frame)
            # condition False: y(segmented_frame)
            # flood 에서 보라색으로 나옴
            # mask = segmented_frame != [0, 0, 0] # background 가 아닌 것들만 True
            # blended_frame = np.where(mask, segmented_frame, resized_frame)

            # 완전 진하게 나옴
            # mask = np.all(segmented_frame == [0, 0, 0], axis=2)
            # blended_frame = resized_frame.copy()
            # blended_frame[~mask] = segmented_frame[~mask]

            alpha = 0.4
            blended_frame = cv2.addWeighted(segmented_frame, alpha, resized_frame, 1 - alpha, 0)

            # 영역별 info box 생성
            for object_info in objects_info: # object_info: (class, centroid)
                # 텍스트 내용 및 위치 설정
                # class_no:class 번호, centroid: text 위치(object 의 중앙), area: object 의 면적
                class_no, centroid, area = object_info

                if area < OBJECT_AREA_TRHESHOLD:
                    continue
                # if not idx % FRAME_TEXT_LABELING_TRHESHOLD:
                #     continue

                text = class_map[class_no]

                # 폰트 및 스케일 설정
                # font = cv2.FONT_HERSHEY_COMPLEX
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.3
                thickness = 1

                # 텍스트 크기 계산
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                centroid = set_centroid_compare_with_prev(centroid, objects_centroids[class_no])

                # 박스 좌표 계산 (h, w), 형변환
                pt1 = (int(centroid[0]), int(centroid[1] - text_height - 5))
                pt2 = (int(centroid[0] + text_width + 5), int(centroid[1] + 5))

                if pt2[0] >= 512:
                    pt2 = (511, pt2[1])
                if pt2[1] >= 512:
                    pt2 = (pt2[1], 511)

                box_coords = (pt1, pt2)

                # 텍스트 좌표(centroid) 계산 (h, w), 형변환
                centroid = (int(centroid[0]), int(centroid[1]))
                if centroid[0] >= 512:
                    centroid = (511, centroid[1])
                if centroid[1] >= 512:
                    centroid = (centroid[1], 511)


                overlay = blended_frame.copy()
                # 박스 그리기
                cv2.rectangle(overlay, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

                # # 텍스트 추가
                # cv2.putText(overlay, text, centroid, font, font_scale, tuple(int(c) for c in color_map[class_no]), thickness)

                alpha = 0.4
                blended_frame = cv2.addWeighted(overlay, alpha, blended_frame, 1 - alpha, 0)
                # 텍스트 추가 (row + 5: 텍스트 중앙정렬)
                cv2.putText(blended_frame, text, (centroid[0]+ 2,centroid[1]), font, font_scale, tuple(int(c) for c in bgr_color_map[class_no]), thickness)

            # video 파일에 frame 하나 생성
            output_video.write(blended_frame)

    if is_imshow:
        cv2.destroyAllWindows()
    else:
        # 비디오 파일 생성 객체 해제
        output_video.release()
    # 비디오 캡처 객체 해제
    capture.release()

video_names.sort()
for video in video_names:

    # 긴 영상만
    if video.find('long') < 0:
        continue

    # 불 제외
    # if video.find('fire') >= 0:
    #     continue

    print(f'current_video={video}')
    # if video == 'fire01.mkv':
    file_name, file_extension = os.path.splitext(video)
    input_path = os.path.join(input_dir, video)
    output_path = os.path.join('./segmented_video', file_name + '.mp4')
    frame_path = os.path.join(frame_dir, file_name)

    reset_and_creat_folder(frame_path)
    # reset_and_creat_folder(output_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 비디오 코덱 설정
    output_video = cv2.VideoWriter(output_path, fourcc, FPS, output_size) # path, codec, fps, frame_size
    # 영상 실시간 Segmentation 실행
    realtime_segmentation(path=input_path, is_imshow=False, output_video=output_video, frame_path=frame_path)
