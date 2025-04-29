#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   RewritePrompt.py
@Time    :   2025/04/29 19:34:20
@Author  :   Cecilll
@Version :   1.0
@Desc    :   Image Generation and Retrieval Pipeline with careful GPU memory management
'''

from modelscope import AutoModelForCausalLM, AutoTokenizer

def load_qwen_llm(model_name = "./model/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model,tokenizer

def run_qwen_llm(prompt,model,tokenizer):

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.01
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":

    prompt = "The brown bear is giving a lecture on the platform."
    vl_prompt = "A brown bear."

    llm_model,llm_tokenizer = load_qwen_llm(model_name = "../model/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    llm_prompt = run_qwen_llm(f"If you want to change '{vl_prompt}' to '{prompt}', which specific concepts need to be adjusted? These concepts should be used to generate images. Please output a maximum of 3 concepts, and remember to only output the concepts without any additional information.",llm_model,llm_tokenizer)
    print(llm_prompt)

    llm_prompt = f"Please separate the concepts in '{llm_prompt}' with an '&'."
    print(llm_prompt)
    response = run_qwen_llm(llm_prompt,llm_model,llm_tokenizer)
    print(response)

    for conception_text in response.split('&'):
        print(conception_text)