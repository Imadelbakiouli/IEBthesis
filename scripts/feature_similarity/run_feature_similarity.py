"""
Feature Similarity Analysis - All 30 videos, 3 layers
Real vs Stylized DAVIS videos
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
import os
import matplotlib.pyplot as plt
from scipy import stats
from tapnet.torch import tapir_model
from tapnet import evaluation_datasets

PKL_PATH        = "/home/u672153/tapnet/data/tapvid_davis.pkl"
CHECKPOINT_PATH = "/home/u672153/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"
STYLIZED_DIR    = "/home/u672153/stylized_videos_finalv2"
METRICS_REAL    = "/home/u672153/tapnet/metrics/metrics_real.json"
METRICS_STYLIZED= "/home/u672153/tapnet/metrics/metrics_stylized.json"
OUTPUT_DIR      = "/home/u672153/tapnet/feature_similarity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading model...")
model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.eval().to(device)
torch.set_grad_enabled(False)
print("Model loaded!")

# Multiple hooks
features_cache = {}

def make_hook(name):
    def hook_fn(module, input, output):
        features_cache[name] = output.detach()
    return hook_fn

layers = {
    'backbone': model.resnet_torch.block_groups[3].blocks[1].conv_1,
    'cost_volume': model.torch_cost_volume_track_mods.hid3,
    'refinement': model.extra_convs.blocks[4].conv_1,
}

for name, layer in layers.items():
    layer.register_forward_hook(make_hook(name))

def preprocess_frames(frames):
    frames = torch.tensor(frames).float().to(device)
    frames = frames / 255 * 2 - 1
    return frames

def load_stylized_video(video_idx, target_frames):
    path = f"{STYLIZED_DIR}/video{video_idx}.mp4"
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = frames[indices]
    elif len(frames) < target_frames:
        pad = np.repeat(frames[-1:], target_frames - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames

def get_all_features(frames, query_points):
    features_cache.clear()
    frames_t = preprocess_frames(frames).unsqueeze(0)
    query_t = torch.tensor(query_points).float().unsqueeze(0).to(device)
    _ = model(frames_t, query_t)
    return {k: v for k, v in features_cache.items()}

def sample_point_features(feat_map, query_points):
    if feat_map.dim() == 4:
        feat_map = feat_map[0:1]
    N = query_points.shape[0]
    all_features = []
    for i in range(N):
        y_px = query_points[i, 1]
        x_px = query_points[i, 2]
        y_norm = (y_px / 256) * 2 - 1
        x_norm = (x_px / 256) * 2 - 1
        grid = torch.tensor([[[[x_norm, y_norm]]]]).float().to(device)
        sampled = F.grid_sample(feat_map, grid, align_corners=True)
        sampled = sampled.squeeze().cpu().numpy()
        if sampled.ndim == 0:
            sampled = sampled.reshape(1)
        all_features.append(sampled)
    return np.stack(all_features)

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))

# Load metrics
with open(METRICS_REAL) as f:
    metrics_real = json.load(f)
with open(METRICS_STYLIZED) as f:
    metrics_stylized = json.load(f)

aj_drops = [float(metrics_real[i]['average_jaccard']) - float(metrics_stylized[i]['average_jaccard'])
            for i in range(30)]
aj_real  = [float(metrics_real[i]['average_jaccard']) for i in range(30)]
aj_stylized = [float(metrics_stylized[i]['average_jaccard']) for i in range(30)]

# Load dataset
print("Loading dataset...")
davis_dataset = evaluation_datasets.create_davis_dataset(PKL_PATH, query_mode='first')

all_results = []

for sample_idx, sample in enumerate(davis_dataset):
    sample = sample['davis']
    print(f"\nVideo {sample_idx}/29")

    frames_real = np.round((sample['video'][0] + 1) / 2 * 255).astype(np.uint8)
    query_points = sample['query_points'][0]

    if frames_real.shape[0] > 81:
        indices = np.linspace(0, frames_real.shape[0] - 1, 81, dtype=int)
        frames_real = frames_real[indices]

    T = frames_real.shape[0]
    frames_stylized = load_stylized_video(sample_idx, T)

    feats_real = get_all_features(frames_real, query_points)
    feats_stylized = get_all_features(frames_stylized, query_points)

    video_result = {
        'video_idx': sample_idx,
        'aj_real': aj_real[sample_idx],
        'aj_stylized': aj_stylized[sample_idx],
        'aj_drop': aj_drops[sample_idx],
        'similarities': {}
    }

    for layer_name in layers.keys():
        if layer_name not in feats_real or layer_name not in feats_stylized:
            print(f"  {layer_name}: hook did not fire")
            continue
        pts_r = sample_point_features(feats_real[layer_name], query_points)
        pts_s = sample_point_features(feats_stylized[layer_name], query_points)
        sim = cosine_similarity(pts_r, pts_s)
        video_result['similarities'][layer_name] = sim
        print(f"  {layer_name}: {sim:.4f}")

    print(f"  AJ drop: {aj_drops[sample_idx]:.3f}")
    all_results.append(video_result)

# Summary stats
print("\n--- Summary ---")
layer_names = list(layers.keys())
for layer_name in layer_names:
    sims = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    print(f"{layer_name}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}, median={np.median(sims):.4f}")

# Correlation
print("\n--- Correlations with AJ drop ---")
for layer_name in layer_names:
    sims = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    drops = [r['aj_drop'] for r in all_results if layer_name in r['similarities']]
    r_val, p_val = stats.pearsonr(sims, drops)
    print(f"{layer_name}: Pearson r={r_val:.3f}, p={p_val:.4f}")

# Save JSON
summary = {}
for layer_name in layer_names:
    sims = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    summary[layer_name] = {
        'mean': float(np.mean(sims)),
        'std': float(np.std(sims)),
        'median': float(np.median(sims))
    }

output = {
    'per_video': all_results,
    'summary': summary
}
with open(os.path.join(OUTPUT_DIR, 'feature_similarity_results.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved!")

# Plot 1: Bar chart mean similarity per layer
fig, ax = plt.subplots(figsize=(7, 5))
means = [summary[l]['mean'] for l in layer_names]
stds  = [summary[l]['std'] for l in layer_names]
colors = ['#4878CF', '#6ACC65', '#D65F5F']
bars = ax.bar(layer_names, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
ax.set_ylabel('Mean Cosine Similarity', fontsize=12)
ax.set_xlabel('Network Layer', fontsize=12)
ax.set_title('Feature Similarity (Real vs Stylized) per Layer', fontsize=13)
ax.set_ylim(0, 1.05)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_per_layer.png'), dpi=150)
print("Bar chart saved!")

# Plot 2: Scatter similarity vs AJ drop (cost volume)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, layer_name, color in zip(axes, layer_names, colors):
    sims  = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    drops = [r['aj_drop'] for r in all_results if layer_name in r['similarities']]
    r_val, p_val = stats.pearsonr(sims, drops)
    ax.scatter(sims, drops, color=color, s=60, alpha=0.8)
    for i, (s, d) in enumerate(zip(sims, drops)):
        ax.annotate(str(i), (s, d), fontsize=7, ha='right')
    ax.set_xlabel('Cosine Similarity', fontsize=11)
    ax.set_ylabel('AJ Drop', fontsize=11)
    ax.set_title(f'{layer_name}\nr={r_val:.3f}, p={p_val:.4f}', fontsize=11)
plt.suptitle('Feature Similarity vs Performance Drop per Layer', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_vs_drop.png'), dpi=150)
print("Scatter plot saved!")
print("\nDone!")
