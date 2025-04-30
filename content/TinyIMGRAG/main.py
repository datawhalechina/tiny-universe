#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025/04/29 19:31:39
@Author  :   Cecilll
@Version :   1.0
@Desc    :   Image Generation and Retrieval Pipeline with careful GPU memory management
'''

import os
import torch
from IMGRAG.ImgGenerator import SDXLGenerator
from IMGRAG.ImgEvaluator import load_qwen_vlm, run_qwen_vl
from IMGRAG.RewritePrompt import load_qwen_llm, run_qwen_llm
from IMGRAG.ImgRetrieval import get_clip_similarities


class ImageRAGPipeline:
    def __init__(self, base_output_dir="./datasets/results"):
        self.base_output_dir = base_output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)

        # Initialize all models as None (lazy loading)
        self.vl_model = None
        self.vl_processor = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.clip_model = None

    def _initialize_vlm(self):
        """Lazy loading of vision-language model with cleanup of previous models"""
        self._cleanup_models()  # Clean up any existing models first
        if self.vl_model is None:
            self.vl_model, self.vl_processor = load_qwen_vlm()
            print("VLM loaded on GPU")

    def _initialize_llm(self):
        """Lazy loading of language model with cleanup of previous models"""
        self._cleanup_models()  # Clean up any existing models first
        if self.llm_model is None:
            self.llm_model, self.llm_tokenizer = load_qwen_llm()
            print("LLM loaded on GPU")

    def _initialize_clip(self):
        """Lazy loading of CLIP model with cleanup of previous models"""
        self._cleanup_models()  # Clean up any existing models first
        if self.clip_model is None:
            # Assuming get_clip_similarities handles its own model loading
            # We'll just track that we're using CLIP now
            print("CLIP model will be loaded when needed")

    def _cleanup_models(self):
        """Clean up all models to free GPU memory"""
        if self.vl_model is not None:
            del self.vl_model
            self.vl_model = None
            print("VLM removed from GPU")

        if self.vl_processor is not None:
            del self.vl_processor
            self.vl_processor = None

        if self.llm_model is not None:
            del self.llm_model
            self.llm_model = None
            print("LLM removed from GPU")

        if self.llm_tokenizer is not None:
            del self.llm_tokenizer
            self.llm_tokenizer = None

        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
            print("CLIP model removed from GPU")

        torch.cuda.empty_cache()
        print("GPU cache cleared")

    def generate_initial_image(self, prompt, seed=0, steps=50):
        """Generate initial image using text prompt only"""
        self._cleanup_models()  # Clean up before image generation
        output_path = os.path.join(self.base_output_dir, "initial_image.png")
        generator = SDXLGenerator(
            prompt=prompt,
            output_path=output_path,
            steps=steps,
            seed=seed,
            use_image_guidance=False
        )
        image_path = generator.generate_image()
        self._cleanup_models()  # Clean up after image generation
        return image_path

    def evaluate_image_content(self, image_path, prompt):
        """Evaluate if image matches the prompt using VLM"""
        try:
            self._initialize_vlm()
            evaluation = run_qwen_vl(image_path, prompt, self.vl_model, self.vl_processor)
            return evaluation
        finally:
            self._cleanup_models()

    def analyze_mismatch_concepts(self, vlm_feedback):
        """Analyze mismatched concepts using LLM"""
        try:
            self._initialize_llm()
            analysis_prompt = (
                f"Analyze the most important inconsistent concepts from '{vlm_feedback}' "
                "and describe the concepts using a noun or a noun modified by an adjective or adverb, "
                "the format like 'A dog' or 'A oil-painting style' or 'A car & A running man' "
                "but do not use 'A different conception'. I would like to use this conception to search "
                "for related images. Please write a prompt for me based on this request. "
                "Only output the prompt."
            )
            return run_qwen_llm(analysis_prompt, self.llm_model, self.llm_tokenizer)
        finally:
            self._cleanup_models()

    def validate_concepts(self, concept_texts):
        """Validate which concepts are suitable for image retrieval"""
        try:
            self._initialize_llm()
            valid_concepts = []
            for concept in concept_texts:
                validation_prompt = (
                    f"If '{concept.strip()}' is a concept that can be described with an image, "
                    "only output True; otherwise, only output False."
                )
                is_valid = run_qwen_llm(validation_prompt, self.llm_model, self.llm_tokenizer)
                if is_valid == "True":
                    valid_concepts.append(concept)
            return valid_concepts
        finally:
            self._cleanup_models()

    def retrieve_reference_images(self, concepts, image_dir="./datasets/imgs"):
        """Retrieve reference images using CLIP similarity"""
        if not concepts:
            return []

        try:
            self._initialize_clip()
            # Get all images in directory
            image_paths = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
            ]

            # For simplicity, we'll just use the first concept
            top_image_paths, _ = get_clip_similarities(
                prompts=concepts[0],
                image_paths=image_paths,
                model_path="./model/ViT-B-32.pt"
            )

            return top_image_paths[:1]  # Return just the top match
        finally:
            self._cleanup_models()

    def generate_enhanced_image(self, prompt, reference_image_path, seed=0, steps=50, ip_scale=0.5):
        """Generate enhanced image using both text and reference image"""
        self._cleanup_models()  # Clean up before image generation
        output_path = os.path.join(self.base_output_dir, "enhanced_image.png")
        generator = SDXLGenerator(
            prompt=prompt,
            image_path=reference_image_path,
            output_path=output_path,
            steps=steps,
            seed=seed,
            use_image_guidance=True,
            ip_scale=ip_scale
        )
        image_path = generator.generate_image()
        self._cleanup_models()  # Clean up after image generation
        return image_path

    def run_pipeline(self, prompt):
        """Run the complete image generation and enhancement pipeline"""
        print("Starting image generation pipeline...")

        try:
            # Step 1: Generate initial image
            print("Generating initial image...")
            initial_image_path = self.generate_initial_image(prompt)
            print(f"Initial image saved to: {initial_image_path}")

            # Step 2: Evaluate image content
            print("Evaluating image content...")
            evaluation = self.evaluate_image_content(initial_image_path, prompt)
            print(f"Evaluation result: {evaluation}")

            if evaluation == "<Content matches>":
                print("Image content matches prompt perfectly.")
                return initial_image_path

            # Step 3: Analyze mismatched concepts
            print("Analyzing mismatched concepts...")
            concept_prompt = self.analyze_mismatch_concepts(evaluation)
            print(f"Identified concepts: {concept_prompt}")

            # Step 4: Validate concepts
            concept_texts = [t.strip() for t in concept_prompt.split('&')]
            valid_concepts = self.validate_concepts(concept_texts)
            print(f"Valid concepts for retrieval: {valid_concepts}")

            if not valid_concepts:
                print("No valid concepts found for retrieval.")
                return initial_image_path

            # Step 5: Retrieve reference images
            print("Retrieving reference images...")
            reference_images = self.retrieve_reference_images(valid_concepts)

            if not reference_images:
                print("No suitable reference images found.")
                return initial_image_path

            # Step 6: Generate enhanced image
            print("Generating enhanced image...")
            enhanced_image_path = self.generate_enhanced_image(
                prompt,
                reference_images[0]
            )
            print(f"Enhanced image saved to: {enhanced_image_path}")

            return enhanced_image_path

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise
        finally:
            # Ensure all models are cleaned up at the end
            self._cleanup_models()


if __name__ == '__main__':
    # Example prompts
    prompts = [
        "A beautiful retriever and a cradle.",
        # "A car"
    ]

    pipeline = ImageRAGPipeline()

    for prompt in prompts:
        print(f"\nProcessing prompt: '{prompt}'")
        try:
            final_image = pipeline.run_pipeline(prompt)
            print(f"Final image for this prompt: {final_image}")
        except Exception as e:
            print(f"Failed to process prompt '{prompt}': {str(e)}")