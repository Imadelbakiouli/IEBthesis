import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

with open('data/tapvid_davis.pkl', 'rb') as f:
    data = pickle.load(f)
keys = list(data.keys())

layers = ['Layer1', 'Layer2', 'Layer15', 'Layer24', 'Layer33', 'Layer34']
output_dir = 'feature_similarity/diff_grids'
os.makedirs(output_dir, exist_ok=True)

for vid_idx in range(30):
    real_path = f'xai_results/selected_layers/video_{vid_idx:02d}/heatmaps_video{vid_idx:02d}.npy'
    styl_path = f'xai_results/selected_layers_stylized/video_{vid_idx:02d}/heatmaps_stylized_video{vid_idx:02d}.npy'
    
    if not os.path.exists(real_path) or not os.path.exists(styl_path):
        print(f"Skipping video {vid_idx}")
        continue
    
    real_h = np.load(real_path, allow_pickle=True).item()
    styl_h = np.load(styl_path, allow_pickle=True).item()
    
    T = real_h['Layer1'].shape[0]
    frame_indices = np.linspace(0, T-1, 6, dtype=int)
    
    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    
    for i, layer in enumerate(layers):
        for j, idx in enumerate(frame_indices):
            r = real_h[layer][idx]
            s = styl_h[layer][idx]
            diff = r - s
            vmax = np.max(np.abs(diff)) + 1e-8
            axes[i, j].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(layer, fontsize=9, rotation=90, labelpad=5)
    
    video_name = keys[vid_idx]
    plt.suptitle(f'EigenCAM Difference (Real - Stylized) - {video_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diff_video{vid_idx:02d}.png', dpi=100)
    plt.close()
    print(f"Video {vid_idx} saved")

print("Done!")
