"""
Feature Similarity - ALL FRAMES, all 30 videos, 3 layers.
Each non-occluded query point is sampled at its GROUND-TRUTH position in EVERY
frame, in both real and stylized, and cosine similarity is averaged over all
(frame, point) pairs.
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
STYLIZED_DIR    = "/home/u672153/StyleMaster/stylemaster-wan/results_all30_v2/style_controlnet"
METRICS_REAL    = "/home/u672153/tapnet/metrics/metrics_real.json"
METRICS_STYLIZED= "/home/u672153/tapnet/metrics/metrics_stylized.json"
OUTPUT_DIR      = "/home/u672153/tapnet/feature_similarity"
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

# Hooks that collect ALL calls (backbone fires in chunks)
calls_cache = {}
def make_hook(name):
    def hook_fn(module, inp, out):
        calls_cache[name].append(out.detach().cpu())
    return hook_fn

layers = {
    'backbone':    model.resnet_torch.block_groups[3].blocks[1].conv_1,
    'cost_volume': model.torch_cost_volume_track_mods.hid3,
    'refinement':  model.extra_convs.blocks[4].conv_1,
}
for name, layer in layers.items():
    layer.register_forward_hook(make_hook(name))


def preprocess_frames(frames):
    frames = torch.tensor(frames).float().to(device)
    return frames / 255 * 2 - 1


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
        idx = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = frames[idx]
    elif len(frames) < target_frames:
        pad = np.repeat(frames[-1:], target_frames - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames


def get_all_calls(frames, query_points):
    for k in layers:
        calls_cache[k] = []
    frames_t = preprocess_frames(frames).unsqueeze(0)
    query_t = torch.tensor(query_points).float().unsqueeze(0).to(device)
    _ = model(frames_t, query_t)
    return {k: list(v) for k, v in calls_cache.items()}


def reconstruct_per_frame(calls, T, n_points, label):
    """-> schone (T, C, H, W) per laag, zelfde logica als de EigenCAM-fix."""
    total_lead = sum(c.shape[0] for c in calls)
    if total_lead == T:
        return torch.cat(list(calls), dim=0)
    if len(calls) == 1 and calls[0].shape[0] == T * n_points:
        c = calls[0]
        C, H, W = c.shape[1], c.shape[2], c.shape[3]
        return c.reshape(T, n_points, C, H, W).mean(dim=1)
    raise ValueError(f"{label}: kan niet naar frames mappen, calls={[tuple(c.shape) for c in calls]}")


def sample_at(feat_chw, x_px, y_px):
    """feat_chw: (C, H, W) van EEN frame -> feature-vector (C,) op (x_px, y_px)."""
    x_norm = (x_px / 256) * 2 - 1
    y_norm = (y_px / 256) * 2 - 1
    grid = torch.tensor([[[[x_norm, y_norm]]]]).float().to(device)
    fm = feat_chw.unsqueeze(0).to(device).float()      # (1, C, H, W)
    s = F.grid_sample(fm, grid, align_corners=True)    # (1, C, 1, 1)
    return s.squeeze().cpu().numpy().reshape(-1)


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


with open(METRICS_REAL) as f:
    metrics_real = json.load(f)
with open(METRICS_STYLIZED) as f:
    metrics_stylized = json.load(f)
aj_drops = [float(metrics_real[i]['average_jaccard']) - float(metrics_stylized[i]['average_jaccard']) for i in range(30)]
aj_real  = [float(metrics_real[i]['average_jaccard']) for i in range(30)]
aj_stylized = [float(metrics_stylized[i]['average_jaccard']) for i in range(30)]

print("Loading dataset...")
davis_dataset = evaluation_datasets.create_davis_dataset(PKL_PATH, query_mode='first')

all_results = []
checked = False

for sample_idx, sample in enumerate(davis_dataset):
    sample = sample['davis']
    print(f"\nVideo {sample_idx}/29")

    frames_real = np.round((sample['video'][0] + 1) / 2 * 255).astype(np.uint8)
    query_points = sample['query_points'][0]          # (N, 3) = (t, y, x)
    target_points = sample['target_points'][0]        # (N, T, 2) = (x, y)
    occluded = sample['occluded'][0]                  # (N, T) bool
    N = query_points.shape[0]

# frames + GT truncate on the same indices (alignment)
    if frames_real.shape[0] > MAX_FRAMES:
        idx = np.linspace(0, frames_real.shape[0] - 1, MAX_FRAMES, dtype=int)
        frames_real = frames_real[idx]
        target_points = target_points[:, idx, :]
        occluded = occluded[:, idx]
    T = frames_real.shape[0]
    frames_stylized = load_stylized_video(sample_idx, T)

    if not checked:
        gt_x0, gt_y0 = float(target_points[0, 0, 0]), float(target_points[0, 0, 1])
        q_y0, q_x0 = float(query_points[0, 1]), float(query_points[0, 2])
        print(f"  [check] target frame0=(x={gt_x0:.1f}, y={gt_y0:.1f}) vs query=(x={q_x0:.1f}, y={q_y0:.1f})")
        if abs(gt_x0 - q_x0) > 2 or abs(gt_y0 - q_y0) > 2:
            print("  [check] WAARSCHUWING: komt niet overeen -> coordinaten-volgorde checken!")
        else:
            print("  [check] OK: coordinaten-volgorde (x,y) klopt.")
        checked = True

    calls_real = get_all_calls(frames_real, query_points)
    calls_styl = get_all_calls(frames_stylized, query_points)

    video_result = {'video_idx': sample_idx, 'aj_real': aj_real[sample_idx],
                    'aj_stylized': aj_stylized[sample_idx], 'aj_drop': aj_drops[sample_idx],
                    'similarities': {}}

    for layer_name in layers:
        real_pf = reconstruct_per_frame(calls_real[layer_name], T, N, layer_name)  # (T,C,H,W)
        styl_pf = reconstruct_per_frame(calls_styl[layer_name], T, N, layer_name)

        a_list, b_list = [], []
        for t in range(T):
            for i in range(N):
                if bool(occluded[i, t]):
                    continue
                x_px = float(target_points[i, t, 0])
                y_px = float(target_points[i, t, 1])
                a_list.append(sample_at(real_pf[t], x_px, y_px))
                b_list.append(sample_at(styl_pf[t], x_px, y_px))

        if len(a_list) == 0:
            print(f"  {layer_name}: geen zichtbare punten")
            continue
        a = np.stack(a_list); b = np.stack(b_list)
        sim = cosine_similarity(a, b)
        video_result['similarities'][layer_name] = sim
        print(f"  {layer_name}: {sim:.4f}  ({len(a_list)} frame-point paren)")

    print(f"  AJ drop: {aj_drops[sample_idx]:.3f}")
    all_results.append(video_result)

# Summary
print("\n--- Summary ---")
layer_names = list(layers.keys())
for layer_name in layer_names:
    sims = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    print(f"{layer_name}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}, median={np.median(sims):.4f}")

print("\n--- Correlations with AJ drop ---")
for layer_name in layer_names:
    sims = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    drops = [r['aj_drop'] for r in all_results if layer_name in r['similarities']]
    r_val, p_val = stats.pearsonr(sims, drops)
    print(f"{layer_name}: Pearson r={r_val:.3f}, p={p_val:.4f}")

summary = {}
for layer_name in layer_names:
    sims = [r['similarities'][layer_name] for r in all_results if layer_name in r['similarities']]
    summary[layer_name] = {'mean': float(np.mean(sims)), 'std': float(np.std(sims)), 'median': float(np.median(sims))}

with open(os.path.join(OUTPUT_DIR, 'feature_similarity_results_allframes.json'), 'w') as f:
    json.dump({'per_video': all_results, 'summary': summary}, f, indent=2)
print("\nResults saved!")

# Bar chart
fig, ax = plt.subplots(figsize=(7, 5))
means = [summary[l]['mean'] for l in layer_names]
stds  = [summary[l]['std'] for l in layer_names]
colors = ['#4878CF', '#6ACC65', '#D65F5F']
bars = ax.bar(layer_names, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
ax.set_ylabel('Mean Cosine Similarity', fontsize=12)
ax.set_xlabel('Network Layer', fontsize=12)
ax.set_title('Feature Similarity (Real vs Stylized) per Layer - All Frames', fontsize=12)
ax.set_ylim(0, 1.05)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{mean:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_per_layer_allframes.png'), dpi=150)
plt.close()
print("Bar chart saved!")

# Scatter similarity vs AJ drop
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
plt.suptitle('Feature Similarity vs Performance Drop per Layer - All Frames', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_vs_drop_allframes.png'), dpi=150)
plt.close()
print("Scatter plot saved!")
print("\nDone!")
