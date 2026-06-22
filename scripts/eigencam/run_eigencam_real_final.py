"""
TAPIR EigenCAM - Selected layers, all 30 original videos
Correct per-frame reconstruction per layer:
  - backbone (Layer1/2/15): fired in chunks -> concatenate calls -> (T, C, H, W)
  - cost volume (Layer24): (T*n_points, C, H, W) -> reshape (T, n_points,..) -> mean over points
  - refinement (Layer33/34): already (T, C, H, W)
All layers captured in ONE forward pass per video.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

from tapnet.torch import tapir_model
from tapnet import evaluation_datasets

PKL_PATH = "/home/u672153/tapnet/data/tapvid_davis.pkl"
CHECKPOINT_PATH = "/home/u672153/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"
OUTPUT_DIR = "/home/u672153/tapnet/xai_results/selected_layers_sequence"
MAX_FRAMES = 81

LAYERS = {
    "Layer1": "resnet_torch.initial_conv",
    "Layer2": "resnet_torch.block_groups.0.blocks.0.proj_conv",
    "Layer15": "resnet_torch.block_groups.2.blocks.1.conv_0",
    "Layer24": "torch_cost_volume_track_mods.hid3",
    "Layer33": "extra_convs.blocks.4.conv",
    "Layer34": "extra_convs.blocks.4.conv_1",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading model...")
model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.eval().to(device)
print("Model loaded!")


def make_heatmap_from_feature_map(fmap, output_size):
    """fmap shape: [C, H, W] -> EigenCAM heatmap resized to output_size."""
    C, H_feat, W_feat = fmap.shape
    feat_np = fmap.detach().cpu().numpy()
    reshaped = feat_np.reshape(C, -1).T
    reshaped = reshaped - reshaped.mean(axis=0)

    U, S, VT = np.linalg.svd(reshaped, full_matrices=False)
    first_component = np.abs(reshaped @ VT[0, :]).reshape(H_feat, W_feat)

    p_low = np.percentile(first_component, 50)
    p_high = np.percentile(first_component, 99)
    first_component = np.clip(first_component, p_low, p_high)

    hmin, hmax = first_component.min(), first_component.max()
    if hmax > hmin:
        first_component = (first_component - hmin) / (hmax - hmin)
    else:
        first_component = np.zeros_like(first_component)

    H, W = output_size
    return cv2.resize(first_component, (W, H))


def reconstruct_per_frame(calls, T, n_points, label):
    """Map a layer's captured hook calls to a clean (T, C, H, W) per-frame tensor."""
    total_lead = sum(c.shape[0] for c in calls)

    # Case A: backbone processed in chunks (8x10 + 1x1 = 81), or a single (T,...) call.
    if total_lead == T:
        return torch.cat(list(calls), dim=0)  # (T, C, H, W)

    # Case B: cost-volume style (T * n_points, C, H, W) -> average over query points.
    if len(calls) == 1 and calls[0].shape[0] == T * n_points:
        c = calls[0]
        C, H, W = c.shape[1], c.shape[2], c.shape[3]

        return c.reshape(T, n_points, C, H, W).mean(dim=1)  # (T, C, H, W)

    raise ValueError(f"{label}: cannot map to frames, calls={[tuple(c.shape) for c in calls]}")


def compute_eigencam_all_layers(frames_np, query_points_tensor, model, layers, n_points):
    """ONE forward pass over the full real video; capture all layers (all calls)."""
    modules = dict(model.named_modules())
    calls = {label: [] for label in layers}
    hooks = []
    for label, name in layers.items():
        def make_hook(lbl):
            def hook_fn(module, inp, out):
                calls[lbl].append(out.detach().cpu())
            return hook_fn
        hooks.append(modules[name].register_forward_hook(make_hook(label)))

    frames_tensor = torch.tensor(frames_np).float().to(device) / 255 * 2 - 1
    video = frames_tensor.unsqueeze(0)  # [1, T, H, W, C]
    qp = query_points_tensor.float().unsqueeze(0).to(device)

    with torch.no_grad():
        model(video, qp)

    for h in hooks:
        h.remove()

    T = frames_np.shape[0]
    output_size = (frames_np.shape[1], frames_np.shape[2])

    all_heatmaps = {}
    for label in layers:
        feat = reconstruct_per_frame(calls[label], T, n_points, label)  # (T, C, H, W)
        print(f"  [{label}] {len(calls[label])} call(s) -> per-frame feat {tuple(feat.shape)}")
        heatmaps = [make_heatmap_from_feature_map(feat[t], output_size) for t in range(T)]
        all_heatmaps[label] = np.array(heatmaps)
    return all_heatmaps


print("Loading dataset...")
davis_dataset = evaluation_datasets.create_davis_dataset(PKL_PATH, query_mode="first")

for video_idx, sample in enumerate(davis_dataset):
    sample = sample["davis"]
    print(f"\n=== Video {video_idx}/29 ===")

    frames = np.round((sample["video"][0] + 1) / 2 * 255).astype(np.uint8)
    query_points = sample["query_points"][0]
    query_points_tensor = torch.tensor(query_points).to(device)
    n_points = len(query_points)

    if frames.shape[0] > MAX_FRAMES:
        indices = np.linspace(0, frames.shape[0] - 1, MAX_FRAMES, dtype=int)
        frames = frames[indices]

    print(f"Frames: {frames.shape[0]}, Query points: {n_points}")

    video_dir = os.path.join(OUTPUT_DIR, f"video_{video_idx:02d}")
    os.makedirs(video_dir, exist_ok=True)

    num_viz = min(6, frames.shape[0])
    viz_indices = [int(i) for i in np.linspace(0, frames.shape[0] - 1, num_viz)]

    all_heatmaps = compute_eigencam_all_layers(frames, query_points_tensor, model, LAYERS, n_points)

    n_rows = 1 + len(LAYERS)
    fig, axes = plt.subplots(n_rows, num_viz, figsize=(4 * num_viz, 4 * n_rows), squeeze=False)
    fig.suptitle(f"TAPIR EigenCAM Sequence — Video {video_idx}", fontsize=11)

    for col, fidx in enumerate(viz_indices):
        axes[0, col].imshow(frames[fidx])
        axes[0, col].set_title(f"Frame {fidx}", fontsize=8)
        axes[0, col].axis("off")

        for row, (layer_label, heatmaps) in enumerate(all_heatmaps.items(), start=1):
            axes[row, col].imshow(frames[fidx])
            axes[row, col].imshow(heatmaps[fidx], alpha=0.6, cmap="jet")
            axes[row, col].axis("off")

    row_labels = ["Original"] + list(LAYERS.keys())
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=7, fontweight="bold")

    plt.tight_layout()
    png_path = os.path.join(video_dir, f"eigencam_sequence_video{video_idx:02d}.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved: {png_path}")

    np.save(os.path.join(video_dir, f"heatmaps_sequence_video{video_idx:02d}.npy"), all_heatmaps)

print("\nDone!")
