# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
import gc
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset, load_dataset_builder
from datasets.distributed import split_dataset_by_node
from einops import rearrange
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

# from common import rank_print, load_model, get_standard_transform, collate
#
# try:
#    import wandb
# except ImportError:
#    wandb = None


LAYER_STATS = dict()


@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    """
    Computes the RankMe (http://arxiv.org/abs/2210.02885) and LiDAR (http://arxiv.org/abs/2312.04000)
    estimates of the rank of the produced embeddings. While RADIO doesn't train in a multi-view setting
    which is an assumption of LiDAR, the metric does integrate an important concept of the invariance of the
    summary features to different view/augmentations of the same image.
    """

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    cv2.setNumThreads(1)

    device = torch.device("cuda", local_rank)
    parser = argparse.ArgumentParser(description="Compute SSL embedding rank estimates")
    parser.add_argument("-v", "--model-version", default="radio_v2", help="Which radio model to load.")
    parser.add_argument("-d", "--dataset", default="imagenet-1k", help="The name of the dataset to classify")
    parser.add_argument("--split", default="validation", help="The dataset split to use.")
    parser.add_argument("-n", default=10, type=int, help="The number of samples to load")
    parser.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=int,
        default=None,
        help="The input image resolution."
        " If one value is specified, the shortest dimension is resized to this."
        " If two, the image is center cropped."
        " If not specified, center cropped 378px is used."
        " Default: The RADIO model's preferred resolution.",
    )
    parser.add_argument(
        "--resize-multiple",
        type=int,
        default=None,
        help="Resize images with dimensions a multiple of this value."
        " This should be equal to the patch size of a ViT (e.g. RADIOv1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="The batch size. If the input is variable sized, then this argument becomes a maximum.",
    )
    parser.add_argument("--workers", default=8, type=int, help="Number of loader workers to use")
    parser.add_argument(
        "--vitdet-window-size", default=None, type=int, help="Enable ViTDet at the specific window size"
    )
    parser.add_argument("--output-dir", default="vis_denoise", type=str)
    parser.add_argument("--adaptor-name", default=None, type=str, help="Generate features from a teacher adaptor")

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    rank_print("Loading model...")
    model, preprocessor, info = load_model(
        args.model_version, vitdet_window_size=args.vitdet_window_size, adaptor_name=args.adaptor_name
    )
    model.to(device=device).eval()
    if isinstance(preprocessor, nn.Module):
        preprocessor.to(device).eval()
    rank_print("Done")

    rank_print("Loading dataset...")
    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    patch_size = model.patch_size

    if args.resize_multiple is None:
        args.resize_multiple = getattr(model, "min_resolution_step", model.patch_size)

    transform = get_standard_transform(args.resolution, args.resize_multiple)
    dataset = ds_builder.as_dataset(split=args.split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.map(
        lambda ex: dict(image=transform(ex["image"]), label=torch.as_tensor(ex["label"], dtype=torch.int64))
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=args.workers > 0,
        drop_last=False,
    )
    rank_print("Done")
    rank_print(f"Description: {ds_builder.info.description}")

    dirs = dict(
        orig=os.path.join(args.output_dir, "orig"),
        viz=os.path.join(args.output_dir, "viz"),
        sbs=os.path.join(args.output_dir, "sbs"),
    )

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    ctr = 0
    for batches in loader:
        if ctr >= args.n:
            break

        for images, _ in batches:
            images = images.to(device=device, non_blocking=True)

            all_feat = []
            with torch.autocast(device.type, dtype=torch.bfloat16):
                p_images = preprocessor(images)

                output = model(p_images)
                if args.adaptor_name:
                    all_feat = [
                        output["backbone"].features,
                        output[args.adaptor_name].features,
                    ]
                else:
                    all_feat = [output[1]]

            all_feat = torch.stack(all_feat, dim=1)

            num_rows = images.shape[-2] // patch_size
            num_cols = images.shape[-1] // patch_size

            all_feat = rearrange(all_feat, "b m (h w) c -> b m h w c", h=num_rows, w=num_cols).float()

            for i, feats in enumerate(all_feat):
                colored = []
                for features in feats:
                    color = get_pca_map(features, images.shape[-2:])
                    colored.append(color)

                orig = cv2.cvtColor(images[i].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

                cv2.imwrite(f'{dirs["orig"]}/vis_{ctr}.jpg', orig * 255)
                cv2.imwrite(f'{dirs["viz"]}/vis_{ctr}.jpg', colored[-1] * 255)

                op = np.concatenate([orig] + colored, axis=1) * 255

                cv2.imwrite(f'{dirs["sbs"]}/vis_{ctr}.jpg', op)
                ctr += 1


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="bicubic",
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    feature_map = feature_map.float()
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(feature_map.reshape(-1, feature_map.shape[-1]))
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.clamp(0, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


def get_scale_map(
    scalar_map: torch.Tensor,
    img_size,
    interpolation="nearest",
):
    """
    scalar_map: (1, h, w, C) is the feature map of a single image.
    """
    if scalar_map.shape[0] != 1:
        scalar_map = scalar_map[None]
    scalar_map = (scalar_map - scalar_map.min()) / (scalar_map.max() - scalar_map.min() + 1e-6)
    scalar_map = F.interpolate(
        scalar_map.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    # cmap = plt.get_cmap("viridis")
    # scalar_map = cmap(scalar_map)[..., :3]
    # make it 3 channels
    scalar_map = torch.cat([scalar_map] * 3, dim=-1)
    scalar_map = scalar_map.cpu().numpy().squeeze(0)
    return scalar_map


def get_similarity_map(features: torch.Tensor, img_size=(224, 224)):
    """
    compute the similarity map of the central patch to the rest of the image
    """
    assert len(features.shape) == 4, "features should be (1, C, H, W)"
    H, W, C = features.shape[1:]
    center_patch_feature = features[0, H // 2, W // 2, :]
    center_patch_feature_normalized = center_patch_feature / center_patch_feature.norm()
    center_patch_feature_normalized = center_patch_feature_normalized.unsqueeze(1)
    # Reshape and normalize the entire feature tensor
    features_flat = features.view(-1, C)
    features_normalized = features_flat / features_flat.norm(dim=1, keepdim=True)

    similarity_map_flat = features_normalized @ center_patch_feature_normalized
    # Reshape the flat similarity map back to the spatial dimensions (H, W)
    similarity_map = similarity_map_flat.view(H, W)

    # Normalize the similarity map to be in the range [0, 1] for visualization
    similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())
    # we don't want the center patch to be the most similar
    similarity_map[H // 2, W // 2] = -1.0
    similarity_map = (
        F.interpolate(
            similarity_map.unsqueeze(0).unsqueeze(0),
            size=img_size,
            mode="bilinear",
        )
        .squeeze(0)
        .squeeze(0)
    )

    similarity_map_np = similarity_map.cpu().numpy()
    negative_mask = similarity_map_np < 0

    colormap = plt.get_cmap("turbo")

    # Apply the colormap directly to the normalized similarity map and multiply by 255 to get RGB values
    similarity_map_rgb = colormap(similarity_map_np)[..., :3]
    similarity_map_rgb[negative_mask] = [1.0, 0.0, 0.0]
    return similarity_map_rgb


def get_cluster_map(
    feature_map: torch.Tensor,
    img_size,
    num_clusters=10,
) -> torch.Tensor:
    kmeans = KMeans(n_clusters=num_clusters, distance=CosineSimilarity, verbose=False)
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    labels = kmeans.fit_predict(feature_map.reshape(1, -1, feature_map.shape[-1])).float()
    labels = (
        F.interpolate(labels.reshape(1, *feature_map.shape[:-1]), size=img_size, mode="nearest").squeeze().cpu().numpy()
    ).astype(int)
    cmap = plt.get_cmap("rainbow", num_clusters)
    cluster_map = cmap(labels)[..., :3]
    return cluster_map.reshape(img_size[0], img_size[1], 3)


if __name__ == "__main__":
    rank = 0
    world_size = 1

    # if 'WORLD_SIZE' in os.environ:
    #     dist.init_process_group(backend='nccl')
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()

    main(rank, world_size)


