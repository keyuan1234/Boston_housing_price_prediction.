"""
Boston Housing - Visualization Script
Reads loss.csv, corr.csv, pred.csv from C program and generates plots.
Usage: python plot_results.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def load_csv(path, skip_header=True):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)
        for row in reader:
            rows.append(row)
    return rows

def plot_loss():
    data = load_csv('loss.csv')
    epochs = [int(r[0]) for r in data]
    mse    = [float(r[1]) for r in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, mse, color='#2196F3', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (normalized)')
    ax.set_title('Training Loss Curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('loss_curve.png', dpi=120)
    plt.close(fig)
    print('Saved loss_curve.png')

def plot_correlation():
    data = load_csv('corr.csv')
    features = [r[0] for r in data]
    corr_vals = [float(r[1]) for r in data]

    colors = ['#F44336' if v < 0 else '#2196F3' for v in corr_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(features, corr_vals, color=colors, edgecolor='white')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Pearson r with MEDV')
    ax.set_title('Feature Correlation with House Price (MEDV)')
    ax.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, corr_vals):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.3f}', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=7)

    fig.tight_layout()
    fig.savefig('corr_bar.png', dpi=120)
    plt.close(fig)
    print('Saved corr_bar.png')

def plot_pred_vs_true():
    data = load_csv('pred.csv')
    true_k = [float(r[2]) for r in data]
    pred_k = [float(r[3]) for r in data]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_k, pred_k, alpha=0.6, s=30, color='#673AB7',
               edgecolors='white', linewidth=0.3)

    mn = min(min(true_k), min(pred_k))
    mx = max(max(true_k), max(pred_k))
    ax.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1,
            label='Perfect fit')

    ax.set_xlabel('True MEDV ($k)')
    ax.set_ylabel('Predicted MEDV ($k)')
    ax.set_title('Predicted vs True Values (Test Set)')

    errors = np.array(pred_k) - np.array(true_k)
    rmse_val = np.sqrt(np.mean(errors ** 2))
    r2 = 1 - np.sum(errors**2) / np.sum((np.array(true_k) - np.mean(true_k))**2)
    ax.text(0.05, 0.92, f'RMSE = ${rmse_val:.2f}k\nR^2 = {r2:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('pred_vs_true.png', dpi=120)
    plt.close(fig)
    print('Saved pred_vs_true.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('Generating plots...')
    plot_loss()
    plot_correlation()
    plot_pred_vs_true()
    print('Done. 3 PNG files generated.')
