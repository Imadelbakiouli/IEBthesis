"""
Statistical analysis: paired t-test and Wilcoxon signed-rank test
comparing TAPIR performance on original vs stylized TAP-Vid-DAVIS videos.
"""
import json
from scipy import stats

real = json.load(open('/home/u672153/tapnet/metrics/metrics_real.json'))
stylized = json.load(open('/home/u672153/tapnet/metrics/metrics_stylized.json'))

for metric in ['average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy']:
    r = [v[metric] for v in real]
    s = [v[metric] for v in stylized]
    t, p_t = stats.ttest_rel(r, s)
    w, p_w = stats.wilcoxon(r, s)
    print(f"{metric}:")
    print(f"  paired t-test: t={t:.3f}, p={p_t:.2e}")
    print(f"  wilcoxon:      W={w:.0f}, p={p_w:.2e}")
