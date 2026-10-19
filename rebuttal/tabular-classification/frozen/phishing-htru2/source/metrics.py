"""Validation-selected thresholds and prespecified held-out curve summaries."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score


def primary(dataset, y, scores):
    return float(roc_auc_score(y, scores) if dataset == 'magic' else average_precision_score(y, scores))


def validation_thresholds(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    # Deterministic conservative tie: roc_curve orders thresholds high to low.
    best = int(np.argmax(tpr - fpr))
    available = np.flatnonzero(fpr <= .05)
    low = int(available[np.argmax(tpr[available])])
    # Use the finite boundary just above the validation maximum for the initial
    # all-negative validation rule. Future scores above it can still be positive.
    def finite_threshold(v):
        return float(np.nextafter(np.max(scores), np.inf)) if np.isposinf(v) else float(v)
    return {'balanced': finite_threshold(thresholds[best]), 'low_fpr': finite_threshold(thresholds[low])}


def classification(y, scores, threshold):
    pred = scores >= threshold
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)),
                fpr=float(np.mean(pred[y == 0])), threshold=float(threshold))


def evaluate(y, scores, thresholds, zero=0.):
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    # Keep the ordered empirical path: interpolate to the *lower* end of the
    # next vertical jump. At an exact repeated FPR, np.interp uses its last TPR
    # (the upper endpoint, a right-continuous convention).
    return dict(roc_auc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
                tpr_at_fpr_005=float(np.interp(.05, fpr, tpr)),
                balanced=classification(y, scores, thresholds['balanced']),
                validation_low_fpr=classification(y, scores, thresholds['low_fpr']),
                default=classification(y, scores, zero))
