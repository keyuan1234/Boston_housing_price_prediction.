"""
Generate 4 feature-relation plots with polynomial fit curves and user predictions in red.
Reads housing-price.txt, model_params.csv, user_pred.csv.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- Load raw dataset ----
raw_features = []
raw_medv = []
with open('housing-price.txt', 'r') as f:
    lines = f.readlines()
for line in lines[1:]:
    parts = line.strip().split()
    if len(parts) >= 14:
        raw_features.append([float(x) for x in parts[:13]])
        raw_medv.append(float(parts[13]))
raw_features = np.array(raw_features)
raw_medv = np.array(raw_medv)

# ---- Load model params ----
params = {}
with open('model_params.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        params[row[0]] = float(row[1])

feat_info = {
    'LSTAT':   {'idx': 12, 'label': 'LSTAT (Low-income %)'},
    'RM':      {'idx': 5,  'label': 'RM (Avg rooms)'},
    'PTRATIO': {'idx': 10, 'label': 'PTRATIO (Pupil-teacher)'},
    'INDUS':   {'idx': 2,  'label': 'INDUS (Non-retail acres)'},
}

# ---- Load user predictions ----
user_points = []
if os.path.exists('user_pred.csv'):
    with open('user_pred.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            user_points.append([float(v) for v in row])

# ---- 2x2 subplots ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (fname, finfo) in enumerate(feat_info.items()):
    ax = axes[i]
    fidx = finfo['idx']
    fmin = params[f'min_{fname}']
    fmax = params[f'max_{fname}']

    x_raw = raw_features[:, fidx]

    # Grey scatter: raw data
    ax.scatter(x_raw, raw_medv, alpha=0.35, s=12, color='#AAAAAA', label='Dataset')

    # Non-linear polynomial fit (degree 2)
    poly_coeffs = np.polyfit(x_raw, raw_medv, deg=2)
    x_fit = np.linspace(fmin, fmax, 200)
    y_fit = np.polyval(poly_coeffs, x_fit)
    ax.plot(x_fit, y_fit, color='#2196F3', linewidth=2, label='Poly fit (deg=2)')

    # User predictions in red
    names_order = ['LSTAT', 'RM', 'PTRATIO', 'INDUS']
    j_idx = names_order.index(fname)
    for up in user_points:
        ax.scatter(up[j_idx], up[4], color='#F44336', s=80, marker='D',
                   edgecolors='#B71C1C', linewidth=1.5, zorder=5)

    if i == 0 and len(user_points) > 0:
        ax.scatter([], [], color='#F44336', s=80, marker='D',
                   edgecolors='#B71C1C', linewidth=1.5,
                   label=f'User pred ({len(user_points)} pts)')

    ax.set_xlabel(finfo['label'])
    ax.set_ylabel('MEDV ($k)')
    ax.set_title(f'MEDV vs {fname}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Boston Housing: Feature-Price (Polynomial Fit)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('pred_vs_true.png', dpi=150)
plt.close(fig)
print('Saved pred_vs_true.png')
