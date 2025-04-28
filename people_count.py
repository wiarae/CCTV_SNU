from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
import argparse
import os 
from tqdm import tqdm
import ast
from collections import deque
import json 


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def main(args):
    # default: Load the model on the available device(s)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )


    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    results = {}
    N = 100  # 유지할 프레임 개수
    memory_buffer = deque(maxlen=N) 

    with open(f'query/{args.prompt}') as f:
        message = f.read()
    query = message
    # query = "Analyze the image for any visible signs of liquid leakage, water pooling, or wet spots near the equipment and floor area."
    # query = "Does the image depict a person who appears to be in a collapsed or falling state?"
    # query = "Is there any visible stream, flow, or pooling of liquid in the image? "
    frames_dir = f"frames_output/{args.frames_dir}"
    frame_files = sorted(os.listdir(frames_dir))
    # frame_file = '000329.jpg'
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        if frame_file.endswith(".png"):
            frame_path = os.path.join(frames_dir, frame_file)

            base64_image = encode_image(frame_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_image}",
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = ast.literal_eval(output_text[0])
            num_people = result.get("Number_of_people", 0)
            is_anomaly = result.get("Fall", 0)
            if num_people == 0:  
                if any(anomaly["Fall"] for anomaly in memory_buffer): 
                    is_anomaly = 1  # anomaly 유지
                    print("Maintaining anomaly state due to missing person!")

            memory_buffer.append({"frame": frame_file, "Fall": is_anomaly, "num_people": num_people})
            result["Fall"] = is_anomaly
            results[frame_file] = result
            print(output_text)
    output_dir = "query_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, args.json_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--videos_dir",
    #     type=str,
    #     required=True,
    #     help="Directory path to the videos.",
    # )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory path to the frames.",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)