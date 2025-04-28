import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import textwrap
from pathlib import Path
import argparse 

def find_closest_key_value(video_captions, frame_idx):
    keys = list(map(int, video_captions.keys()))
    closest_key = min(keys, key=lambda x: abs(x - frame_idx))
    return closest_key, json.loads(video_captions[str(closest_key)])

def visualize_video(
    video_name,
    json_file,
    video_path,
    video_fps,
    save_path,
    normal_label=0,
    imagefile_template="frame_{:05d}.png",
    font_size=18,
    caption_update_interval=1,
):
    with open(json_file, 'r') as f:
        data = json.load(f)

    fall = []    
    leak = []
    video_scores = []
    video_captions = {}
    video_labels = []

    for frame_idx, (frame_name, frame_data) in enumerate(data.items()):
        fall.append(int(frame_data.get("Fall", 0)))
        leak.append(int(frame_data.get("Leak", 0)))
        video_scores.append(frame_data.get("Number_of_people", 0))
        video_captions[str(frame_idx)] = frame_data.get("description", "")
        video_labels.append(1 if fall[-1] or leak[-1] else normal_label)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    video_writer = None
    x = np.arange(len(video_scores))
    ax3.plot(x, fall, color="#4e79a7", linewidth=1, label="Fall")
    ax3.plot(x, leak, color="#f28e2b", linewidth=1, label="Leak")
    ymin, ymax = -0.1, 1.1
    ax3.set_xlim([0, len(video_labels)])
    ax3.set_ylim([ymin, ymax])
    ax3.legend()

    start_idx = None
    for frame_idx, label in enumerate(video_labels):
        if label != normal_label and start_idx is None:
            start_idx = frame_idx
        elif label == normal_label and start_idx is not None:
            ax3.add_patch(plt.Rectangle((start_idx, ymin), frame_idx - start_idx, ymax - ymin, color="#e15759", alpha=0.5))
            start_idx = None
    if start_idx is not None:
        ax3.add_patch(plt.Rectangle((start_idx, ymin), len(video_labels) - start_idx, ymax - ymin, color="#e15759", alpha=0.5))
    
    previous_line = None
    current_caption = ""
    last_caption_frame = -1

    frame_keys = sorted(data.keys())  # 키는 이미지 파일명 (예: "frame_00528.png")

    for i, frame_name in enumerate(frame_keys):
        frame_data = data[frame_name]

        img_path = os.path.join(video_path, frame_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue

        fall_val = int(frame_data.get("Fall", 0))
        leak_val = int(frame_data.get("Leak", 0))
        num_people = frame_data.get("Number_of_people", 0)
        caption = frame_data.get("description", "")

        # 기존 변수 대체
        is_anomaly = fall_val or leak_val
        box_color = (0, 255, 0) if not is_anomaly else (255, 0, 0)
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), box_color, 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.axis("off")

        # 캡션 업데이트
        if (i // (video_fps * caption_update_interval)) != last_caption_frame:
            current_caption = textwrap.fill(caption, width=35)
            last_caption_frame = (i // (video_fps * caption_update_interval))

        ax2.text(0.5, 0.7, current_caption, fontsize=18, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
        ax2.text(0.5, 0.4, f"Number of People: {num_people}", fontsize=16, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
        ax2.text(0.5, 0.1, f"Anomaly: {'Yes' if is_anomaly else 'No'}", fontsize=16, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
        ax2.axis("off")

        if previous_line is not None:
            previous_line.remove()
        previous_line = ax3.axvline(x=i, color="red")

        fig.tight_layout()

        if video_writer is None:
            video_width, video_height = 1200, 800
            fourcc = cv2.VideoWriter_fourcc(*"VP80")
            video_writer = cv2.VideoWriter(str(save_path), fourcc, video_fps, (video_width, video_height))

        fig.canvas.draw()
        frame_img = np.array(fig.canvas.renderer.buffer_rgba())
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
        video_writer.write(frame_img)

        ax1.cla()
        ax2.cla()
    
    plt.close()
    video_writer.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True, help="Name of the video.")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory path to the frames.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the annotations file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_name = args.video_name
    json_file = f'./query_results/{args.json_file}'
    video_path = f'./frames_output/{args.frames_dir}'
    video_fps = 30
    save_path = Path('./results_video')
    save_path.mkdir(parents=True, exist_ok=True)
    visualize_video(video_name, json_file, video_path, video_fps, save_path / f"{video_name}.webm")
