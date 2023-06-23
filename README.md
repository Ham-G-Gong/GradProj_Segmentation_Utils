# GradProj_segmentation_utils
AI 영역 검출 모델, 서버에서 사용하는 유용한 기능 모음



## utils.py

1. reset_and_creat_folder(path)

    폴더의 파일들이 있을 경우 비워주고
    폴더를 생성해줍니다.


2. create_folder_and_copy_files(src_folder, dst_folder, copy_file_list=None)

    src folder 에 파일들을 dst 폴더를 생성하고 복사해줍니다.

3. read_txt_file(filename)

    file 을 읽어 리스트로 리턴하는 함수

4. def color_mapping(image)

    numpy grayscale 이미지를 컬러로 변환해줍니다.
    RGB 순서

5. plot_loss(history), plot_score(history)

    정확도, loss 출력 함수


## video-segmentation.py
- 학습한 모델을 불러와 입력 영상에 대해 프레임 단위로 segmentation 을 수행
- 영역 검출한 object(region-영역)에 대해 labeling 막스를 frame 별로 생성
- 영상을 mp4 파일로 저장

## 실행
- video-segmentation.py 에서 모델의 경로, 영상을 만들기 위핸 설정값 변수들을 수정
- `python video-segmentation.py` 명령어 실행
