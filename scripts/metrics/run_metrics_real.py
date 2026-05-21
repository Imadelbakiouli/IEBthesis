"""
TAPIR Metrics - Original DAVIS videos
"""

import numpy as np
import torch
import torch.nn.functional as F
import jax
import json
import os

from tapnet.torch import tapir_model
from tapnet import evaluation_datasets

PKL_PATH        = "/home/u672153/tapnet/data/tapvid_davis.pkl"
CHECKPOINT_PATH = "/home/u672153/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"
OUTPUT_DIR      = "/home/u672153/tapnet/metrics"

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


print("Loading dataset...")
davis_dataset = evaluation_datasets.create_davis_dataset(PKL_PATH, query_mode='first')

all_metrics = []
summed_scalars = None

for sample_idx, sample in enumerate(davis_dataset):
    sample = sample['davis']
    print(f"\nVideo {sample_idx}/29")

    frames = np.round((sample['video'][0] + 1) / 2 * 255).astype(np.uint8)
    query_points = sample['query_points'][0]

    # Max 81 frames — ook ground truth afkappen
    if frames.shape[0] > 81:
        indices = np.linspace(0, frames.shape[0] - 1, 81, dtype=int)
        frames = frames[indices]
        sample['target_points'] = sample['target_points'][:, :, indices, :]
        sample['occluded'] = sample['occluded'][:, :, indices]

    frames_t = torch.tensor(frames).to(device)
    query_points_t = torch.tensor(query_points).to(device)

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
    video_metrics['video_idx'] = sample_idx
    all_metrics.append(video_metrics)

    if summed_scalars is None:
        summed_scalars = scalars
    else:
        summed_scalars = jax.tree.map(np.add, summed_scalars, scalars)

    num_samples = sample_idx + 1
    mean_scalars = jax.tree.map(lambda x: x / num_samples, summed_scalars)
    print(f"  Mean so far: {mean_scalars}")

with open(os.path.join(OUTPUT_DIR, 'metrics_real.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

mean_final = {k: float(v) for k, v in mean_scalars.items()}
with open(os.path.join(OUTPUT_DIR, 'metrics_real_mean.json'), 'w') as f:
    json.dump(mean_final, f, indent=2)

print(f"\nFinal mean metrics: {mean_final}")
print("Done!")
