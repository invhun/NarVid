import torch
import os
import json
import re
import math
import argparse
from tqdm import tqdm
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def remove_extension(filename):
    """Remove the file extension from a filename."""
    name, _ = os.path.splitext(filename)
    return name

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    """Get the k-th chunk from the list."""
    chunks = split_list(lst, n)
    return chunks[k]

def append_to_json_file(file_path, data):
    """Append data to a JSON file, creating the file if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump([], file, indent=4)

    with open(file_path, 'r+') as file:
        file_data = json.load(file)
        file_data.append(data)
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def main():
    parser = argparse.ArgumentParser(description='Generate captions for video frames')
    parser.add_argument('--video_frames_path', type=str, required=True, help='Path to the folder containing video frames')
    parser.add_argument('--video_id_list_path', type=str, required=True, help='Path to the JSON file containing video IDs')

    args = parser.parse_args()

    disable_torch_init()

    model_path = "liuhaotian/llava-v1.5-7b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    # Prompt for generating narration
    questions = "Please describe this image for image-captioning task."

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], questions)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    base_path = args.video_frames_path
    video_id_list_path = args.video_id_list_path

    with open(video_id_list_path, 'r') as file:
        folder_names = json.load(file)

    start_index = 0
    end_index = len(folder_names)

    # Process each video and generate captions
    for video_id in tqdm(folder_names[start_index:end_index]):
        file_list = os.listdir(os.path.join(base_path, str(video_id)))
        file_list = sorted(file_list, key=extract_number)
        caption_dict = {'video_file': video_id}
        idx = 0
        
        for file_name in file_list:
            idx += 1
            image = Image.open(os.path.join(base_path, str(video_id), file_name))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            temperature = 0.2
            top_p = None
            num_beams = 1
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=1024,
                    use_cache=True
                )
                
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            caption_dict[f'caption_{idx}'] = caption

        append_to_json_file(f"narration.json", caption_dict)

if __name__ == "__main__":
    main()
