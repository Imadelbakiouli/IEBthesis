import json
import matplotlib.pyplot as plt
import numpy as np

with open('/home/u672153/tapnet/feature_similarity/feature_similarity_results.json') as f:
    data = json.load(f)

summary = data['summary']
layer_names = ['backbone', 'cost_volume', 'refinement']
means = [summary[l]['mean'] for l in layer_names]
colors = ['#4878CF', '#6ACC65', '#D65F5F']

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(layer_names, means, color=colors, alpha=0.85)
ax.set_ylabel('Mean Cosine Similarity', fontsize=12)
ax.set_xlabel('Network Layer', fontsize=12)
ax.set_title('Feature Similarity (Real vs Stylized) per Layer', fontsize=13)
ax.set_ylim(0, 1.05)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('/home/u672153/tapnet/feature_similarity/similarity_per_layer.png', dpi=150)
print("Done!")
