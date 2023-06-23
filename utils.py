import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# 폴더의 파일들이 있을 경우 비워주고
# 폴더를 생성해줍니다.
def reset_and_creat_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# src folder 에 파일들을 dst 폴더를 생성하고 복사해줍니다.
def create_folder_and_copy_files(src_folder, dst_folder, copy_file_list=None):
    if os.path.isdir(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder, exist_ok=True)

    if copy_file_list == None:
        # 소스 폴더의 파일 목록 얻기
        files = os.listdir(src_folder)
    else:
        files = copy_file_list

    # 파일들을 대상 폴더로 복사
    for file_name in files:
        source_file = os.path.join(src_folder, file_name)
        destination_file = os.path.join(dst_folder, file_name)
        shutil.copy2(source_file,   destination_file)


# file 을 읽어 리스트로 리턴하는 함수
def read_txt_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        string_list = [line.rstrip(',\n').strip() for line in lines]
        return string_list


# numpy grayscale 이미지를 컬러로 변환해줍니다.
# RGB 순서
def color_mapping(image):
    color_map = np.array([
        [0, 0, 0],       # 0 배경 (검정색)  : background
        [128, 0, 0],     # 클래스 1 (어두운 빨강) : avalanche
        [0, 128, 0],     # 클래스 2 (어두운 초록) : building_undamaged
        [128, 128, 0],   # 클래스 3 (어두운 옥색) : building_damaged
        [0, 0, 128],     # 클래스 4 (어두운 파랑) : cracks/fissure/subsidence
        [128, 0, 128],   # 클래스 5 (어두운 보라) : debris/mud/rock : flow
        [0, 128, 128],   # 클래스 6 (어두운 시안) : fire/flare
        [128, 128, 128], # 클래스 7 (회색) : flood/water/river/sea
        [64, 0, 0],      # 클래스 8 (진한 빨강) : ice_jam_flow
        [192, 0, 0],     # 클래스 9 (밝은 빨강) : lava_flow
        [64, 128, 0],    # 클래스 10 (노란) : person
        [192, 128, 0],   # 클래스 11 (주황) : pyroclastic_flow
        [64, 0, 128],    # 클래스 12 (보라) : road/railway/bridge
        [192, 0, 128],   # 클래스 13 (분홍) : vehicle
    ])

    # 색으로 변환
    return color_map[image]

# 정확도, loss 출력 함수
def plot_loss(history):
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('plot_loss.png')
    plt.close()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_miou', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('plot_score.png')
    plt.close()