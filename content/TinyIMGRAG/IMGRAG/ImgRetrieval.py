#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ImgRetrieval.py
@Time    :   2025/04/29 19:31:39
@Author  :   Cecilll
@Version :   1.0
@Desc    :   Image Generation and Retrieval Pipeline with careful GPU memory management
'''

import os
import torch
import clip
import numpy as np
from PIL import Image


def get_clip_similarities(prompts, image_paths, embeddings_path="./datasets/vector_bases", bs=2, k=5, device='cuda:0',
                          model_path="./model/ViT-B-32.pt"):
    """
    Calculate similarity between text prompts and images using CLIP model.

    Args:
        prompts: List of text prompts to compare against images
        image_paths: List of paths to images
        embeddings_path: Directory to save/load precomputed image embeddings
        bs: Batch size for processing images
        k: Number of top similar images to return
        device: Device to run computations on
        model_path: Path to CLIP model weights

    Returns:
        Tuple of (top image paths, top similarity scores) sorted by similarity
    """
    # Load CLIP model and preprocessing
    model, preprocess = clip.load(model_path, device=device)
    text_tokens = clip.tokenize(prompts).to(device)

    # Initialize result containers
    all_scores = []
    all_paths = []
    all_embeddings = torch.empty((0, 512)).to(device)

    # Process in batches
    with torch.no_grad():
        # Get text features once
        text_features = model.encode_text(text_tokens)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

        # Process images in batches
        for batch_start in range(0, len(image_paths), bs):
            batch_end = min(batch_start + bs, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            # Try to load precomputed embeddings or compute new ones
            embeddings, valid_paths = _get_image_embeddings(
                batch_paths, batch_start,
                model, preprocess, device,
                embeddings_path
            )

            if embeddings is None:
                continue

            # Calculate similarities
            batch_scores = torch.matmul(text_features, embeddings.T)
            batch_scores = batch_scores.cpu().numpy().squeeze()

            # Update running totals
            if batch_scores.ndim == 0:  # 如果是标量
                all_scores.append(batch_scores.item())  # 用 .item() 获取标量值
            else:
                all_scores.extend(batch_scores.tolist())  # 如果是多维数组，转换为列表

            # all_scores.extend(batch_scores)
            all_paths.extend(valid_paths)
            all_embeddings = torch.cat([all_embeddings, embeddings])

            # Keep only top k results
            if len(all_scores) > k:
                top_indices = np.argsort(all_scores)[-k:]
                all_scores = [all_scores[i] for i in top_indices]
                all_paths = [all_paths[i] for i in top_indices]
                all_embeddings = all_embeddings[top_indices]

    # Return sorted results (highest first)
    sorted_indices = np.argsort(all_scores)[::-1]
    return [all_paths[i] for i in sorted_indices], [all_scores[i] for i in sorted_indices]


def _get_image_embeddings(image_paths, batch_idx, model, preprocess, device, embeddings_path):
    """Helper to get embeddings either from cache or by computing them"""
    cache_file = os.path.join(embeddings_path, f"clip_embeddings_b{batch_idx}.pt")

    # Try loading from cache
    if os.path.exists(cache_file):
        cached = torch.load(cache_file, map_location=device)
        return cached['normalized_clip_embeddings'], cached['paths']

    # Compute new embeddings
    images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = preprocess(Image.open(path)).unsqueeze(0).to(device)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Couldn't read {path}: {str(e)}")
            continue

    if not images:
        return None, None

    images = torch.cat(images)
    features = model.encode_image(images)
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    # Save to cache if requested
    if embeddings_path:
        os.makedirs(embeddings_path, exist_ok=True)
        torch.save({
            'normalized_clip_embeddings': features,
            'paths': valid_paths
        }, cache_file)

    return features, valid_paths

if __name__ == "__main__":

    clip_texts = ['a cradle']
    for clip_text in clip_texts:
        print(clip_text)

        src = "../datasets/imgs"
        files = os.listdir(src)

        file_paths = []
        for f in files:
            file_path = os.path.join(src, f)
            file_paths.append(file_path)

        top_text_im_paths, top_text_im_scores = get_clip_similarities(prompts=clip_text, image_paths=file_paths, model_path="../model/ViT-B-32.pt")

        print(top_text_im_paths, top_text_im_scores)
        print('----------------------')
