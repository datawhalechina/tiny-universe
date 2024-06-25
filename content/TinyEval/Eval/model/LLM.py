import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, List, Optional, Tuple, Union
import torch
import os
from tqdm import tqdm


class BaseLLM:
    def __init__(self, path: str, model_name: str, adapter_path: str) -> None:
        self.path = path
        self.model_name = model_name
        self.adapter_path = adapter_path

    def build_chat(self, tokenizer, prompt, model_name):
        pass

    def load_model_and_tokenizer(self, path, model_name, device):
        pass

    def post_process(self, response, model_name):
        pass

    def get_pred(self, data: list, max_length: int, max_gen: int, prompt_format: str, device, out_path: str):
        pass


class internlm2Chat(BaseLLM):
    def __init__(self, path: str, model_name: str = '', adapter_path: str = '') -> None:
        super().__init__(path, model_name, adapter_path)  # 调用父类初始化函数并传入参数
        
    def build_chat(self, prompt):
        prompt = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        return prompt
    
    def post_process(self, response):
        response = response.split("<|im_end|>")[0]
        return response

    def load_model_and_tokenizer(self, path, device, adapter_path):
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if adapter_path:
            # print(adapter_path)
            model = PeftModel.from_pretrained(model, model_id=adapter_path)
        model = model.eval()
        return model, tokenizer
    
    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        model, tokenizer = self.load_model_and_tokenizer(self.path, device, self.adapter_path)
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断,因为两头有关键信息.
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            prompt = self.build_chat(prompt)

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]  # 表示喂进去的tokens的长度
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=1.0,
                eos_token_id=eos_token_id,
            )[0]
            
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = self.post_process(pred)
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')


class Qwen2Chat(BaseLLM):
    def __init__(self, path: str, model_name: str = '', adapter_path: str = '') -> None:
        super().__init__(path, model_name, adapter_path)  # 调用父类初始化函数并传入参数
        
    def build_chat(self, prompt, instruct=None):
        if instruct is None:
            instruct = 'You are a helpful assistant.'
        prompt = f'<|im_start|>system\n{instruct}<im_end>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        return prompt
    

    def load_model_and_tokenizer(self, path, device, adapter_path):
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # adapter_path = ''
        if adapter_path:
            model = PeftModel.from_pretrained(model, model_id=adapter_path)
            print(f"adapter loaded in {adapter_path}")
        model = model.eval()
        return model, tokenizer
    
    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        model, tokenizer = self.load_model_and_tokenizer(self.path, device, self.adapter_path)
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断,因为两头有关键信息.
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            prompts = self.build_chat(prompt, json_obj.get('instruction', None))
            inputs = tokenizer(prompts, truncation=False, return_tensors="pt").to(device)

            output = model.generate(
                            inputs.input_ids,
                            do_sample=True,
                            temperature=1.0,
                            max_new_tokens=max_gen,
                            top_p=0.8
                            )
            
            pred = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)]
            pred = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["output"], "all_classes": json_obj.get("all_classes", None), "length": json_obj.get("length", None)}, f, ensure_ascii=False)
                f.write('\n')