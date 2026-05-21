"""
TAPIR Metrics - Stylized videos
"""

import numpy as np
import torch
import torch.nn.functional as F
import jax
import json
import os
import cv2

from tapnet.torch import tapir_model
from tapnet import evaluation_datasets

STYLIZED_DIR    = "/home/u672153/StyleMaster/stylemaster-wan/results_all30_v2/style_controlnet"
PKL_PATH        = "/home/u672153/tapnet/data/tapvid_davis.pkl"
CHECKPOINT_PATH = "/home/u672153/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"
OUTPUT_DIR      = "/home/u672153/tapnet/metrics"
MAX_FRAMES      = 81

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading model...")
model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.eval().to(device)
torch.set_grad_enabled(False)
print("Model loaded!")


def preprocess_frames(frames):
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames

def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles

def inference(frames, query_points, model):
    frames = preprocess_frames(frames)
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]
    outputs = model(frames, query_points)
    tracks, occlusions, expected_dist = (
        outputs['tracks'][0],
        outputs['occlusion'][0],
        outputs['expected_dist'][0],
    )
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles

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


print("Loading DAVIS dataset for query points and ground truth...")
davis_dataset = evaluation_datasets.create_davis_dataset(PKL_PATH, query_mode='first')
davis_samples = list(davis_dataset)

all_metrics = []
summed_scalars = None

for video_idx in range(30):
    print(f"\nVideo {video_idx}/29")

    # Ground truth van originele DAVIS
    sample = davis_samples[video_idx]['davis']
    query_points = sample['query_points'][0]
    query_points_t = torch.tensor(query_points).to(device)

    # Bepaal originele frame count
    orig_frames = np.round((sample['video'][0] + 1) / 2 * 255).astype(np.uint8)
    n_orig_frames = min(orig_frames.shape[0], MAX_FRAMES)

    # Ground truth afkappen naar zelfde aantal frames
    if orig_frames.shape[0] > MAX_FRAMES:
        indices = np.linspace(0, orig_frames.shape[0] - 1, MAX_FRAMES, dtype=int)
        sample['target_points'] = sample['target_points'][:, :, indices, :]
        sample['occluded'] = sample['occluded'][:, :, indices]

    # Stylized frames laden
    video_path = os.path.join(STYLIZED_DIR, f"video{video_idx}.mp4")
    frames = load_mp4_frames(video_path)

    # Match met originele frame count
    indices = np.linspace(0, frames.shape[0] - 1, n_orig_frames, dtype=int)
    frames = frames[indices]
    print(f"Frames: {frames.shape[0]}, Query points: {len(query_points)}")

    frames_t = torch.tensor(frames).to(device)

    tracks, visibles = inference(frames_t, query_points_t, model)

    tracks = tracks.cpu().detach().numpy()
    visibles = visibles.cpu().detach().numpy()
    query_points_np = query_points_t.cpu().detach().numpy()
    occluded = ~visibles

    scalars = evaluation_datasets.compute_tapvid_metrics(
        query_points_np[None],
        sample['occluded'],
        sample['target_points'],
        occluded[None],
        tracks[None],
        query_mode='first',
    )
    scalars = jax.tree.map(lambda x: np.array(np.sum(x, axis=0)), scalars)
    print(f"  {scalars}")

    video_metrics = {k: float(v) for k, v in scalars.items()}
    video_metrics['video_idx'] = video_idx
    all_metrics.append(video_metrics)

    if summed_scalars is None:
        summed_scalars = scalars
    else:
        summed_scalars = jax.tree.map(np.add, summed_scalars, scalars)

    num_samples = video_idx + 1
    mean_scalars = jax.tree.map(lambda x: x / num_samples, summed_scalars)
    print(f"  Mean so far: {mean_scalars}")

with open(os.path.join(OUTPUT_DIR, 'metrics_stylized.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

mean_final = {k: float(v) for k, v in mean_scalars.items()}
with open(os.path.join(OUTPUT_DIR, 'metrics_stylized_mean.json'), 'w') as f:
    json.dump(mean_final, f, indent=2)

print(f"\nFinal mean metrics: {mean_final}")
print("Done!")
