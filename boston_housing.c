/**
 * Boston Housing Price Prediction - Grade A
 * Pearson correlation -> Top 4 features -> Multiple Linear Regression
 * Interactive prediction mode + CSV export for matplotlib.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SAMPLES 506
#define N_FEATURES  13
#define MAX_LINE    512

typedef struct {
    double features[N_FEATURES];
    double target;
} House;

typedef struct {
    double min;
    double max;
} Scaler;

static House  data[MAX_SAMPLES];
static int    n_samples = 0;
static Scaler feature_scaler[N_FEATURES];
static Scaler target_scaler;
static const char *feat_names[N_FEATURES] = {
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
};

int  parse_data(const char *filename);
void normalize_data(void);
void unscale_target(double val, double *out);
double scale_feature(int feat_idx, double raw);
double dot(const double *w, const double *x, int n);
void grad_desc(const double *X, const double *y, int m, int n,
               double *weights, double lr, int epochs, FILE *log);
double rmse(const double *weights, const double *X, const double *y,
            int rows, int n);
void pearson_corr(int *top_indices, int top_k);
void print_corr_and_top(const int *indices, int k, FILE *corr_fp);
void predict_loop(const double *weights, const int *top_idx, int n);

/* ================================================================ */

int parse_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return -1; }
    char line[MAX_LINE];
    fgets(line, sizeof(line), fp); /* skip header */
    while (fgets(line, sizeof(line), fp)) {
        if (n_samples >= MAX_SAMPLES) break;
        House *h = &data[n_samples];
        if (14 == sscanf(line,
            "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &h->features[0],  &h->features[1],  &h->features[2],
            &h->features[3],  &h->features[4],  &h->features[5],
            &h->features[6],  &h->features[7],  &h->features[8],
            &h->features[9],  &h->features[10], &h->features[11],
            &h->features[12], &h->target))
            n_samples++;
    }
    fclose(fp);
    printf("Samples loaded: %d\n", n_samples);
    return n_samples;
}

void normalize_data(void) {
    for (int j = 0; j < N_FEATURES; j++) {
        feature_scaler[j].min =  1e18;
        feature_scaler[j].max = -1e18;
    }
    target_scaler.min =  1e18;
    target_scaler.max = -1e18;
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            double v = data[i].features[j];
            if (v < feature_scaler[j].min) feature_scaler[j].min = v;
            if (v > feature_scaler[j].max) feature_scaler[j].max = v;
        }
        double t = data[i].target;
        if (t < target_scaler.min) target_scaler.min = t;
        if (t > target_scaler.max) target_scaler.max = t;
    }
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            double r = feature_scaler[j].max - feature_scaler[j].min;
            data[i].features[j] = (r > 1e-12)
                ? (data[i].features[j] - feature_scaler[j].min) / r : 0.0;
        }
        double r = target_scaler.max - target_scaler.min;
        data[i].target = (r > 1e-12)
            ? (data[i].target - target_scaler.min) / r : 0.0;
    }
    printf("Min-Max normalization done\n");
}

void unscale_target(double val, double *out) {
    *out = val * (target_scaler.max - target_scaler.min) + target_scaler.min;
}

double scale_feature(int feat_idx, double raw) {
    double r = feature_scaler[feat_idx].max - feature_scaler[feat_idx].min;
    return (r > 1e-12) ? (raw - feature_scaler[feat_idx].min) / r : 0.0;
}

double dot(const double *w, const double *x, int n) {
    double s = w[0];
    for (int j = 0; j < n; j++) s += w[j + 1] * x[j];
    return s;
}

void grad_desc(const double *X, const double *y, int m, int n,
               double *weights, double lr, int epochs, FILE *log) {
    for (int e = 0; e < epochs; e++) {
        double *grad = (double *)calloc(n + 1, sizeof(double));
        for (int i = 0; i < m; i++) {
            double err = dot(weights, X + i * n, n) - y[i];
            grad[0] += err;
            for (int j = 0; j < n; j++)
                grad[j + 1] += err * X[i * n + j];
        }
        for (int j = 0; j <= n; j++)
            weights[j] -= (lr / m) * grad[j];
        free(grad);
        /* log every 100 epochs */
        if (log && e % 100 == 0) {
            double loss = 0;
            for (int i = 0; i < m; i++) {
                double err = dot(weights, X + i * n, n) - y[i];
                loss += err * err;
            }
            loss /= (2.0 * m);
            fprintf(log, "%d,%.6f\n", e, loss);
        }
    }
}

double rmse(const double *weights, const double *X, const double *y,
            int rows, int n) {
    double sum = 0;
    for (int i = 0; i < rows; i++) {
        double err = dot(weights, X + i * n, n) - y[i];
        sum += err * err;
    }
    return sqrt(sum / rows);
}

void pearson_corr(int *top_indices, int top_k) {
    typedef struct { int idx; double corr; } Pair;
    Pair pairs[N_FEATURES];
    double N = (double)n_samples;
    double sum_y = 0, sum_y2 = 0;
    for (int i = 0; i < n_samples; i++) {
        sum_y  += data[i].target;
        sum_y2 += data[i].target * data[i].target;
    }
    for (int j = 0; j < N_FEATURES; j++) {
        double sum_x = 0, sum_x2 = 0, sum_xy = 0;
        for (int i = 0; i < n_samples; i++) {
            double x = data[i].features[j];
            sum_x  += x;
            sum_x2 += x * x;
            sum_xy += x * data[i].target;
        }
        double num = N * sum_xy - sum_x * sum_y;
        double den = sqrt((N * sum_x2 - sum_x * sum_x)
                        * (N * sum_y2 - sum_y * sum_y));
        pairs[j].idx  = j;
        pairs[j].corr = (den > 1e-12) ? num / den : 0.0;
    }
    for (int i = 0; i < N_FEATURES - 1; i++)
        for (int k = i + 1; k < N_FEATURES; k++)
            if (fabs(pairs[k].corr) > fabs(pairs[i].corr)) {
                Pair t = pairs[i]; pairs[i] = pairs[k]; pairs[k] = t;
            }
    for (int i = 0; i < top_k; i++) top_indices[i] = pairs[i].idx;
}

void print_corr_and_top(const int *indices, int k, FILE *corr_fp) {
    printf("\nPearson correlations with MEDV:\n");
    printf("  %-8s | %8s | Rank\n", "Feature", "r");
    typedef struct { int idx; double corr; } Pair;
    Pair p[N_FEATURES];
    double N = (double)n_samples, sy = 0, sy2 = 0;
    for (int i = 0; i < n_samples; i++) {
        sy  += data[i].target;
        sy2 += data[i].target * data[i].target;
    }
    for (int j = 0; j < N_FEATURES; j++) {
        double sx = 0, sx2 = 0, sxy = 0;
        for (int i = 0; i < n_samples; i++) {
            double x = data[i].features[j];
            sx += x; sx2 += x * x; sxy += x * data[i].target;
        }
        double num = N * sxy - sx * sy;
        double den = sqrt((N * sx2 - sx * sx) * (N * sy2 - sy * sy));
        p[j].corr = (den > 1e-12) ? num / den : 0;
        p[j].idx = j;
    }
    for (int i = 0; i < N_FEATURES - 1; i++)
        for (int k2 = i + 1; k2 < N_FEATURES; k2++)
            if (fabs(p[k2].corr) > fabs(p[i].corr)) {
                Pair t = p[i]; p[i] = p[k2]; p[k2] = t;
            }
    for (int rank = 0; rank < N_FEATURES; rank++) {
        int j = p[rank].idx;
        if (corr_fp)
            fprintf(corr_fp, "%s,%.4f\n", feat_names[j], p[rank].corr);
        printf("  %-8s | %+8.4f | #%d\n", feat_names[j], p[rank].corr, rank + 1);
    }
    printf("\n[Grade A] Selected Top %d features: ", k);
    for (int i = 0; i < k; i++)
        printf("%s%s", feat_names[indices[i]], i < k - 1 ? ", " : "");
    printf("\n");
}

/* ---- Interactive prediction ---- */
void predict_loop(const double *weights, const int *top_idx, int n) {
    printf("\n");
    printf("========================================\n");
    printf("  Interactive Prediction Mode\n");
    printf("  Enter feature values to predict MEDV\n");
    printf("  (type 'q' to quit)\n");
    printf("========================================\n");

    char buf[256];
    while (1) {
        double raw[4], scaled[4];
        printf("\nEnter values for: ");
        for (int j = 0; j < n; j++)
            printf("%s%s", feat_names[top_idx[j]], j < n - 1 ? ", " : "");
        printf("\n> ");

        if (!fgets(buf, sizeof(buf), stdin)) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;

        int got = sscanf(buf, "%lf %lf %lf %lf",
                         &raw[0], &raw[1], &raw[2], &raw[3]);
        if (got < n) {
            printf("Need %d numbers. Try again.\n", n);
            continue;
        }

        for (int j = 0; j < n; j++)
            scaled[j] = scale_feature(top_idx[j], raw[j]);

        double pred_norm = dot(weights, scaled, n);
        double pred_raw;
        unscale_target(pred_norm, &pred_raw);

        printf("Predicted MEDV = $%.2fk\n", pred_raw);
    }
    printf("Exiting prediction mode.\n");
}

/* ================================================================ */

int main(void) {
    srand((unsigned)time(NULL));

    printf("========================================\n");
    printf("  Boston Housing - Grade A Prediction\n");
    printf("  Top-4 features + Multiple Linear Reg.\n");
    printf("========================================\n\n");

    if (parse_data("housing-price.txt") < 0) return 1;
    normalize_data();

    for (int i = n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        House t = data[i]; data[i] = data[j]; data[j] = t;
    }
    int train_sz = (int)(n_samples * 0.8);
    int test_sz  = n_samples - train_sz;

    int top_idx[4];
    pearson_corr(top_idx, 4);
    FILE *corr_fp = fopen("corr.csv", "w");
    if (corr_fp) fprintf(corr_fp, "feature,r\n");
    print_corr_and_top(top_idx, 4, corr_fp);
    if (corr_fp) fclose(corr_fp);
    printf("Correlation data saved to corr.csv\n");

    int sel_n = 4;
    double *train_X = (double *)malloc(train_sz * sel_n * sizeof(double));
    double *train_y = (double *)malloc(train_sz * sizeof(double));
    double *test_X  = (double *)malloc(test_sz  * sel_n * sizeof(double));
    double *test_y  = (double *)malloc(test_sz  * sizeof(double));
    for (int i = 0; i < train_sz; i++) {
        train_y[i] = data[i].target;
        for (int j = 0; j < sel_n; j++)
            train_X[i * sel_n + j] = data[i].features[top_idx[j]];
    }
    for (int i = 0; i < test_sz; i++) {
        test_y[i] = data[train_sz + i].target;
        for (int j = 0; j < sel_n; j++)
            test_X[i * sel_n + j] = data[train_sz + i].features[top_idx[j]];
    }

    double *weights = (double *)calloc(sel_n + 1, sizeof(double));
    printf("\n--- Training (lr=0.01, epochs=10000) ---\n");
    FILE *loss_fp = fopen("loss.csv", "w");
    if (loss_fp) fprintf(loss_fp, "epoch,mse\n");
    grad_desc(train_X, train_y, train_sz, sel_n, weights, 0.01, 10000, loss_fp);
    if (loss_fp) fclose(loss_fp);
    printf("Loss curve data saved to loss.csv\n");

    double train_rmse_norm = rmse(weights, train_X, train_y, train_sz, sel_n);
    double test_rmse_norm  = rmse(weights, test_X,  test_y,  test_sz,  sel_n);
    double train_rmse_raw, test_rmse_raw;
    unscale_target(train_rmse_norm, &train_rmse_raw);
    unscale_target(test_rmse_norm,  &test_rmse_raw);

    printf("\n========================================\n");
    printf("  Grade A Evaluation\n");
    printf("========================================\n");
    printf("  Weights:\n");
    printf("    bias     = %+.6f\n", weights[0]);
    for (int j = 0; j < sel_n; j++)
        printf("    w_%-6s = %+.6f\n", feat_names[top_idx[j]], weights[j + 1]);
    printf("  Train RMSE: %.4f (norm) / $%.2fk\n",
           train_rmse_norm, train_rmse_raw);
    printf("  Test  RMSE: %.4f (norm) / $%.2fk\n",
           test_rmse_norm, test_rmse_raw);
    printf("========================================\n");

    /* write prediction vs true CSV */
    FILE *pred_fp = fopen("pred.csv", "w");
    if (pred_fp) fprintf(pred_fp, "true_norm,pred_norm,true_k,pred_k\n");
    for (int i = 0; i < test_sz; i++) {
        double pred_norm = dot(weights, test_X + i * sel_n, sel_n);
        double pred_raw, true_raw;
        unscale_target(pred_norm, &pred_raw);
        unscale_target(test_y[i], &true_raw);
        if (pred_fp) fprintf(pred_fp, "%.4f,%.4f,%.2f,%.2f\n",
            test_y[i], pred_norm, true_raw, pred_raw);
    }
    if (pred_fp) fclose(pred_fp);
    printf("Prediction data saved to pred.csv\n");

    predict_loop(weights, top_idx, sel_n);

    free(train_X); free(train_y);
    free(test_X);  free(test_y);
    free(weights);
    return 0;
}