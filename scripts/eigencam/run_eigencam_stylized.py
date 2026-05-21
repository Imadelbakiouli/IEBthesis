"""
TAPIR EigenCAM - Selected layers, all 30 stylized videos
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

STYLIZED_DIR    = "/home/u672153/StyleMaster/stylemaster-wan/results_all30_v2/style_controlnet"
PKL_PATH        = "/home/u672153/tapnet/data/tapvid_davis.pkl"
CHECKPOINT_PATH = "/home/u672153/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"
OUTPUT_DIR      = "/home/u672153/tapnet/xai_results/selected_layers_stylized"
MAX_FRAMES      = 81

LAYERS = {
    "Layer1":  "resnet_torch.initial_conv",
    "Layer2":  "resnet_torch.block_groups.0.blocks.0.proj_conv",
    "Layer15": "resnet_torch.block_groups.2.blocks.1.conv_0",
    "Layer24": "torch_cost_volume_track_mods.hid3",
    "Layer33": "extra_convs.blocks.4.conv",
    "Layer34": "extra_convs.blocks.4.conv_1",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading model...")
model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.eval().to(device)
print("Model loaded!")


def load_mp4_frames(video_path, target_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)


def compute_eigencam_single_frame(frame_np, query_points_tensor, model, target_layer_name):
    target_layer = dict(model.named_modules())[target_layer_name]
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    hook = target_layer.register_forward_hook(hook_fn)

    frame_tensor = torch.tensor(frame_np).float()
    frame_tensor = frame_tensor / 255 * 2 - 1
    video = frame_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1).to(device)
    qp = query_points_tensor.float().unsqueeze(0)

    with torch.no_grad():
        model(video, qp)

    hook.remove()

    feat = activations[0][0]
    C, H_feat, W_feat = feat.shape

    feat_np = feat.numpy()
    reshaped = feat_np.reshape(C, -1).T
    reshaped = reshaped - reshaped.mean(axis=0)
    U, S, VT = np.linalg.svd(reshaped, full_matrices=True)
    first_component = np.abs(reshaped @ VT[0, :]).reshape(H_feat, W_feat)

    p_low = np.percentile(first_component, 50)
    p_high = np.percentile(first_component, 99)
    first_component = np.clip(first_component, p_low, p_high)

    hmin, hmax = first_component.min(), first_component.max()
    if hmax > hmin:
        first_component = (first_component - hmin) / (hmax - hmin)
    else:
        first_component = np.zeros_like(first_component)

    H, W = frame_np.shape[0], frame_np.shape[1]
    return cv2.resize(first_component, (W, H))


print("Loading DAVIS dataset for query points...")
davis_dataset = evaluation_datasets.create_davis_dataset(PKL_PATH, query_mode='first')
davis_samples = list(davis_dataset)

for video_idx in range(30):
    print(f"\n=== Video {video_idx}/29 ===")

    # Query points en frame count van originele DAVIS
    sample = davis_samples[video_idx]['davis']
    query_points = sample['query_points'][0]
    query_points_tensor = torch.tensor(query_points).to(device)

    orig_frames = np.round((sample['video'][0] + 1) / 2 * 255).astype(np.uint8)
    n_orig_frames = min(orig_frames.shape[0], MAX_FRAMES)

    # Frames laden van gestyleerde MP4
    video_path = os.path.join(STYLIZED_DIR, f"video{video_idx}.mp4")
    frames = load_mp4_frames(video_path)
    print(f"Loaded {frames.shape[0]} frames from MP4")

    # Match met originele frame count
    indices = np.linspace(0, frames.shape[0] - 1, n_orig_frames, dtype=int)
    frames = frames[indices]
    print(f"Frames: {frames.shape[0]}, Query points: {len(query_points)}")

    video_dir = os.path.join(OUTPUT_DIR, f'video_{video_idx:02d}')
    os.makedirs(video_dir, exist_ok=True)

    num_viz = min(6, frames.shape[0])
    viz_indices = [int(i) for i in np.linspace(0, frames.shape[0] - 1, num_viz)]

    all_heatmaps = {}
    for layer_label, layer_name in LAYERS.items():
        print(f"  Layer: {layer_label}")
        heatmaps = []
        for t in range(frames.shape[0]):
            heatmaps.append(compute_eigencam_single_frame(
                frames[t], query_points_tensor, model, layer_name
            ))
        all_heatmaps[layer_label] = np.array(heatmaps)

    n_rows = 1 + len(LAYERS)
    fig, axes = plt.subplots(n_rows, num_viz, figsize=(4 * num_viz, 4 * n_rows), squeeze=False)
    fig.suptitle(f'TAPIR EigenCAM Stylized — Video {video_idx} — Selected Layers', fontsize=11)

    for col, fidx in enumerate(viz_indices):
        axes[0, col].imshow(frames[fidx])
        axes[0, col].set_title(f'Frame {fidx}', fontsize=8)
        axes[0, col].axis('off')

        for row, (layer_label, heatmaps) in enumerate(all_heatmaps.items(), start=1):
            axes[row, col].imshow(frames[fidx])
            axes[row, col].imshow(heatmaps[fidx], alpha=0.6, cmap='jet')
            axes[row, col].axis('off')

    row_labels = ['Stylized'] + list(LAYERS.keys())
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=7, fontweight='bold')

    plt.tight_layout()
    png_path = os.path.join(video_dir, f'eigencam_stylized_video{video_idx:02d}.png')
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  PNG saved: {png_path}")

    np.save(os.path.join(video_dir, f'heatmaps_stylized_video{video_idx:02d}.npy'), all_heatmaps)

print("\nDone!")
