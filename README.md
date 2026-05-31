# Boston Housing Price Prediction - User Guide

## Project Overview

Grade-A Boston housing price prediction using C language.
- Dataset: 506 samples, 13 features + 1 target (MEDV)
- Algorithm: Multiple Linear Regression with Gradient Descent
- Feature selection: Top 4 by Pearson correlation
- Metric: RMSE

## Requirements

| Component | Details |
|-----------|---------|
| C Compiler | GCC 8.1+ (MinGW-W64) |
| Python | 3.7+ (visualization only) |
| Python pkgs | matplotlib, numpy |

Install Python dependencies:
```
pip install matplotlib numpy
```

## Files

| File | Purpose |
|------|---------|
| `boston_housing.c` | Source code |
| `housing-price.txt` | Dataset (506 x 14) |
| `plot_results.py` | Visualization script |
| `boston_housing.exe` | Compiled executable |

Generated at runtime:
| File | Content |
|------|---------|
| `loss.csv` | Training loss curve (epoch, mse) |
| `corr.csv` | 13 feature correlations (feature, r) |
| `pred.csv` | Test set predictions (true, pred) |
| `loss_curve.png` | Loss curve chart |
| `corr_bar.png` | Correlation bar chart |
| `pred_vs_true.png` | Predicted vs true scatter plot |

## Compile

```
gcc -O2 -Wall -o boston_housing.exe boston_housing.c -lm
```

Flags:
- `-O2`: optimization level
- `-lm`: link math library

## Run

### 1. Train Model

```
.\boston_housing.exe
```

Program executes:
1. Load `housing-price.txt`
2. Min-Max normalization
3. Compute Pearson correlation for all 13 features
4. Select Top 4: **LSTAT, RM, PTRATIO, INDUS**
5. Gradient descent training (lr=0.01, epochs=10000)
6. Print weights and RMSE
7. Auto-save `loss.csv`, `corr.csv`, `pred.csv`
8. Enter interactive prediction mode

### 2. Interactive Prediction

After training, the program prompts for feature values:

```
Enter values for: LSTAT, RM, PTRATIO, INDUS
> 5.0 6.5 17.0 7.0
Predicted MEDV = $23.15k
```

Enter 4 numbers separated by spaces, press Enter to get prediction.
Type `q` to quit.

### 3. Non-interactive Run (data only)

```
echo q | .\boston_housing.exe
```

## Visualization

```
python plot_results.py
```

Generates 3 PNG charts:

| Chart | Description |
|-------|-------------|
| `loss_curve.png` | MSE vs epoch training curve |
| `corr_bar.png` | Pearson r bar chart (red=negative, blue=positive) |
| `pred_vs_true.png` | True vs predicted scatter plot with RMSE & R^2 |

## Results Summary

### Correlation Ranking (by |r|)

```
LSTAT    -0.7377   Low-income population % (strongest negative)
RM       +0.6954   Average rooms per dwelling (strongest positive)
PTRATIO  -0.5078   Pupil-teacher ratio
INDUS    -0.4837   Non-retail business acres
TAX      -0.4685   Property tax rate
NOX      -0.4273   Nitric oxide concentration
CRIM     -0.3883   Crime rate
RAD      -0.3816   Highway accessibility index
AGE      -0.3770   Old housing proportion
ZN       +0.3604   Residential land proportion
B        +0.3335   Black population metric
DIS      +0.2499   Distance to employment centers
CHAS     +0.1753   Charles River adjacency
```

### Model Weights

```
bias      = +0.364
w_LSTAT   = -0.430    (higher low-income % -> lower price)
w_RM      = +0.553    (more rooms -> higher price)
w_PTRATIO = -0.203    (higher ratio -> lower price)
w_INDUS   = -0.020    (more non-retail -> slightly lower)
```

### Evaluation

| Metric | Train | Test |
|--------|:-----:|:----:|
| RMSE (norm) | 0.117 | 0.112 |
| RMSE ($k) | 10.26 | 10.03 |

## Troubleshooting

**"Cannot open housing-price.txt"**
Put `housing-price.txt` in the same directory as the executable.
If running from `output\`: `copy ..\housing-price.txt .`

**"No module named 'matplotlib'"**
Run: `pip install matplotlib numpy`

**CSV file garbled**
All output is pure ASCII, no encoding issues.
