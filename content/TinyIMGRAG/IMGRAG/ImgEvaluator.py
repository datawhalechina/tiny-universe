#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ImgEvaluator.py
@Time    :   2025/04/29 19:34:11
@Author  :   Cecilll
@Version :   1.0
@Desc    :   Image Generation and Retrieval Pipeline with careful GPU memory management
'''

import os
import base64
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_qwen_vlm(pretrained_model="./model/Qwen/Qwen2.5-VL-3B-Instruct"):

    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model, torch_dtype="auto", device_map="auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(pretrained_model, min_pixels=min_pixels,
                                              max_pixels=max_pixels)
    return model,processor

def run_qwen_vl(image_path,prompt,model,processor):

    # 编码图片
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_image}",
                },
                {"type": "text", "text": f"Please identify the different between the image and the description of the {prompt}, and output in the format 'The different conception is a XX'. If no inconsistent content are found, return <Content matches>."},
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
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.01)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

if __name__ == "__main__":

    prompt = "The brown bear is giving a lecture on the platform."
    img_path = "../datasets/results/output.png"

    vl_model,vl_processor = load_qwen_vlm(pretrained_model="../model/Qwen/Qwen2.5-VL-3B-Instruct")

    vl_prompt = run_qwen_vl(img_path,prompt,vl_model,vl_processor)

    print(vl_prompt)