import cv2
import os
import time

def extract_all_frames(video_path, output_folder):
    """
    .webm 비디오 파일의 모든 프레임을 이미지로 저장하고,
    FPS, 총 프레임 수, 영상 길이, 처리 시간 정보를 출력하는 함수
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        return

    # 프레임 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"FPS (Frames per Second): {fps}")
    print(f"총 프레임 수: {total_frames}")
    print(f"영상 길이: {duration_sec:.2f}초")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    elapsed_time = time.time() - start_time

    print(f"\n총 {frame_count}개의 프레임이 저장되었습니다.")
    print(f"프레임 추출 처리 시간: {elapsed_time:.2f}초")
    print(f"1프레임당 평균 처리 시간: {elapsed_time/frame_count:.4f}초")

# 실행 예시
video_file = "output_splits_107/split_1.webm"  # 입력 비디오 파일
output_dir = "frames_output/107_people_count_3"  # 프레임을 저장할 폴더
extract_all_frames(video_file, output_dir)  # 1초당 1프레임 저장
