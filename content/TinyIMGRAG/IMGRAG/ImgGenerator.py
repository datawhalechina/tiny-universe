#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ImgGenerator.py
@Time    :   2025/04/29 19:32:02
@Author  :   Cecilll
@Version :   1.0
@Desc    :   Image Generation and Retrieval Pipeline with careful GPU memory management
'''

from PIL import Image
from diffusers import AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection
import torch


class SDXLGenerator:
    def __init__(self, prompt, output_path, steps=50, seed=0,
                 use_image_guidance=False, image_path=None, ip_scale=0.5,
                 sd_path="./model/stabilityai/stable-diffusion-xl-base-1.0",
                 adapter_path="./model/h94/IP-Adapter"):
        self.prompt = prompt
        self.output_path = output_path
        self.steps = steps
        self.seed = seed
        self.use_image_guidance = use_image_guidance
        self.image_path = image_path
        self.ip_scale = ip_scale

        # 设备设置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 初始化管道
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            sd_path,
            torch_dtype=self.torch_dtype
        ).to(self.device)

        # 如果需要使用图像引导，加载图像编码器和IP-Adapter
        if self.use_image_guidance:
            # 初始化图像编码器
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                adapter_path,
                subfolder="models/image_encoder",
                torch_dtype=self.torch_dtype
            ).to(self.device)

            # 将图像编码器设置到管道中
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                sd_path,
                image_encoder=self.image_encoder,
                torch_dtype=self.torch_dtype
            ).to(self.device)

            # 加载IP-Adapter
            self.pipe.load_ip_adapter(
                adapter_path,
                subfolder="sdxl_models",
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
            )
            self.pipe.set_ip_adapter_scale(self.ip_scale)

    def generate_image(self):
        # 准备生成参数
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # 执行生成
        print("开始生成...")
        if self.use_image_guidance:
            # 加载参考图像
            ref_image = Image.open(self.image_path)

            # 使用图像和文本生成
            result = self.pipe(
                prompt=self.prompt,
                ip_adapter_image=ref_image,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                num_inference_steps=self.steps,
                generator=generator,
            )
        else:
            # 只使用文本生成
            result = self.pipe(
                prompt=self.prompt,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                num_inference_steps=self.steps,
                generator=generator,
            )

        # 保存结果
        result.images[0].save(self.output_path)
        print(f"生成完成，结果已保存至: {self.output_path}")
        return self.output_path


# 使用示例
if __name__ == "__main__":
    # 示例 1：只用文本生成图片
    generator = SDXLGenerator(
        prompt="A golden retriever and a cradle",
        output_path="output.png",
        steps=50,
        seed=0,
        use_image_guidance=False,  # 设置为False，表示只用文本生成
        sd_path="../model/stabilityai/stable-diffusion-xl-base-1.0",
        adapter_path="../model/h94/IP-Adapter"
    )
    generator.generate_image()

    # # 示例 2：使用图像和文本生成图片
    # generator = SDXLGenerator(
    #     prompt="A golden retriever and a cradle",
    #     image_path="./datasets/imgs/cradle.jpg",
    #     output_path="output.png",
    #     steps=50,
    #     seed=0,
    #     use_image_guidance=True,  # 设置为True，表示使用图像和文本生成
    #     ip_scale=0.5
    # )
    # generator.generate_image()
