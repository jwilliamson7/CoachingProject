"""
Evaluation metrics for ordinal classification.

This module provides metrics specifically designed for ordinal classification
problems where the class labels have a natural ordering (e.g., 0 < 1 < 2).
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from typing import Dict, Any, Optional


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Quadratic Weighted Kappa (QWK) score.

    QWK measures agreement between raters (or predictions and ground truth)
    while accounting for chance agreement. Distant errors are penalized more
    heavily than adjacent errors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    kappa : float
        Quadratic weighted kappa score in range [-1, 1].
        1 = perfect agreement, 0 = chance agreement, <0 = worse than chance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = max(y_true.max(), y_pred.max()) + 1

    # Build the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    n_samples = conf_mat.sum()

    if n_samples == 0:
        return 0.0

    # Build the weight matrix (quadratic weights)
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    # Calculate observed and expected matrices
    hist_true = np.sum(conf_mat, axis=1)
    hist_pred = np.sum(conf_mat, axis=0)

    # Expected matrix under independence
    expected = np.outer(hist_true, hist_pred) / n_samples

    # Calculate kappa
    observed_weighted = np.sum(weights * conf_mat)
    expected_weighted = np.sum(weights * expected)

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    kappa = 1 - (observed_weighted / expected_weighted)
    return kappa


def mean_absolute_error_ordinal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error for ordinal predictions.

    For ordinal classification, MAE represents the average distance
    between predicted and true class labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    mae : float
        Mean absolute error between predictions and true labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy allowing for predictions within 1 class of the true label.

    A prediction is considered correct if it matches the true label or is
    exactly one class away (adjacent).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    accuracy : float
        Proportion of predictions within 1 class of true label.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred) <= 1)


def exact_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute exact accuracy (standard classification accuracy).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    accuracy : float
        Proportion of exactly correct predictions.
    """
    return accuracy_score(y_true, y_pred)


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and F1 for each class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    class_names : list, optional
        Names for each class. If None, uses "Class 0", "Class 1", etc.

    Returns
    -------
    metrics : dict
        Nested dictionary with metrics for each class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = max(y_true.max(), y_pred.max()) + 1

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    metrics = {}
    for i, name in enumerate(class_names):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(conf_mat[i, :].sum())
        }

    return metrics


def ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive ordinal classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    y_proba : array-like of shape (n_samples, n_classes), optional
        Predicted class probabilities. If provided, AUROC is computed.

    class_names : list, optional
        Names for each class.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mae': Mean Absolute Error
        - 'qwk': Quadratic Weighted Kappa
        - 'adjacent_accuracy': Accuracy within 1 class
        - 'exact_accuracy': Exact match accuracy
        - 'macro_f1': Macro-averaged F1 score
        - 'auroc': AUROC (if y_proba provided)
        - 'per_class': Per-class precision, recall, F1
        - 'confusion_matrix': Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = max(y_true.max(), y_pred.max()) + 1

    if class_names is None:
        class_names = [
            "Class 0 (1-2 yrs)",
            "Class 1 (3-4 yrs)",
            "Class 2 (5+ yrs)"
        ][:n_classes]

    metrics = {
        'mae': mean_absolute_error_ordinal(y_true, y_pred),
        'qwk': quadratic_weighted_kappa(y_true, y_pred),
        'adjacent_accuracy': adjacent_accuracy(y_true, y_pred),
        'exact_accuracy': exact_accuracy(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'per_class': per_class_metrics(y_true, y_pred, class_names),
        'confusion_matrix': confusion_matrix(
            y_true, y_pred, labels=np.arange(n_classes)
        )
    }

    # Add AUROC if probabilities are provided
    if y_proba is not None:
        try:
            metrics['auroc'] = roc_auc_score(
                y_true, y_proba, average='macro', multi_class='ovr'
            )
        except ValueError:
            # May fail if not all classes present in y_true
            metrics['auroc'] = None

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'r2': R-squared (coefficient of determination)
        - 'correlation': Pearson correlation coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred),
        'correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    }

    return metrics


def format_regression_report(
    metrics: Dict[str, float],
    title: str = "Regression Report"
) -> str:
    """
    Format regression metrics dictionary as a readable string report.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from regression_metrics().

    title : str
        Title for the report.

    Returns
    -------
    report : str
        Formatted string report.
    """
    lines = [
        "=" * 60,
        title.center(60),
        "=" * 60,
        "",
        "REGRESSION METRICS",
        "-" * 40,
        f"Mean Absolute Error (MAE):  {metrics['mae']:.4f}",
        f"Mean Squared Error (MSE):   {metrics['mse']:.6f}",
        f"Root MSE (RMSE):            {metrics['rmse']:.4f}",
        f"R-squared (R²):             {metrics['r2']:.4f}",
        f"Pearson Correlation:        {metrics['correlation']:.4f}",
        "",
        "=" * 60
    ]

    return "\n".join(lines)


def format_metrics_report(
    metrics: Dict[str, Any],
    title: str = "Classification Report"
) -> str:
    """
    Format metrics dictionary as a readable string report.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from ordinal_metrics().

    title : str
        Title for the report.

    Returns
    -------
    report : str
        Formatted string report.
    """
    lines = [
        "=" * 60,
        title.center(60),
        "=" * 60,
        "",
        "ORDINAL METRICS",
        "-" * 40,
        f"Mean Absolute Error (MAE):  {metrics['mae']:.4f}",
        f"Quadratic Weighted Kappa:   {metrics['qwk']:.4f}",
        f"Adjacent Accuracy (+-1):    {metrics['adjacent_accuracy']:.4f}",
        f"Exact Accuracy:             {metrics['exact_accuracy']:.4f}",
        f"Macro F1 Score:             {metrics['macro_f1']:.4f}",
    ]

    if 'auroc' in metrics and metrics['auroc'] is not None:
        lines.append(f"AUROC (macro):              {metrics['auroc']:.4f}")

    lines.extend([
        "",
        "PER-CLASS METRICS",
        "-" * 40,
        f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    ])

    for class_name, class_metrics in metrics['per_class'].items():
        lines.append(
            f"{class_name:<20} {class_metrics['precision']:>10.3f} "
            f"{class_metrics['recall']:>10.3f} {class_metrics['f1']:>10.3f} "
            f"{class_metrics['support']:>10}"
        )

    lines.extend([
        "",
        "CONFUSION MATRIX",
        "-" * 40,
        "             Predicted",
        "          " + "  ".join([f"{i:>5}" for i in range(metrics['confusion_matrix'].shape[1])])
    ])

    for i, row in enumerate(metrics['confusion_matrix']):
        lines.append(f"Actual {i}:  " + "  ".join([f"{val:>5}" for val in row]))

    lines.append("=" * 60)

    return "\n".join(lines)
