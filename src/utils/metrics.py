"""Sequence Labeling Metrics for Bull/Bear Flag Detection.

This module provides specialized metrics for evaluating per-timestep
sequence labeling predictions, beyond standard classification metrics.

Metrics included:
- Standard: Accuracy, F1, Precision, Recall (micro/macro/weighted)
- ROC-AUC: Per-class and macro-averaged
- Coverage Score: IoU-like metric for pattern regions
- Detection Rate: % of patterns at least partially detected
- Boundary Accuracy: Accuracy of pattern start/end positions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


# Label mapping (must match prepare_data_sequence.py)
LABEL_NAMES = {
    0: "None",
    1: "Bearish Normal",
    2: "Bearish Wedge",
    3: "Bearish Pennant",
    4: "Bullish Normal",
    5: "Bullish Wedge",
    6: "Bullish Pennant",
}

NUM_CLASSES = len(LABEL_NAMES)


@dataclass
class SequenceMetrics:
    """Container for all sequence labeling metrics."""
    
    # Standard metrics
    accuracy: float = 0.0
    f1_micro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision_micro: float = 0.0
    precision_macro: float = 0.0
    recall_micro: float = 0.0
    recall_macro: float = 0.0
    
    # Per-class metrics
    f1_per_class: Dict[str, float] = field(default_factory=dict)
    precision_per_class: Dict[str, float] = field(default_factory=dict)
    recall_per_class: Dict[str, float] = field(default_factory=dict)
    
    # ROC-AUC (requires probabilities)
    roc_auc_macro: float = 0.0
    roc_auc_per_class: Dict[str, float] = field(default_factory=dict)
    
    # Pattern-specific metrics
    coverage_score: float = 0.0  # IoU for pattern regions
    detection_rate: float = 0.0  # % of patterns detected
    false_alarm_rate: float = 0.0  # % of false positive patterns
    boundary_accuracy: float = 0.0  # Accuracy within N timesteps of boundaries
    
    # Pattern-level (not timestep-level)
    pattern_precision: float = 0.0  # Of predicted patterns, how many are correct
    pattern_recall: float = 0.0  # Of true patterns, how many are detected
    pattern_f1: float = 0.0
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        result = {
            "accuracy": self.accuracy,
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_micro": self.precision_micro,
            "precision_macro": self.precision_macro,
            "recall_micro": self.recall_micro,
            "recall_macro": self.recall_macro,
            "roc_auc_macro": self.roc_auc_macro,
            "coverage_score": self.coverage_score,
            "detection_rate": self.detection_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "boundary_accuracy": self.boundary_accuracy,
            "pattern_precision": self.pattern_precision,
            "pattern_recall": self.pattern_recall,
            "pattern_f1": self.pattern_f1,
        }
        
        # Add per-class metrics
        for cls_name, value in self.f1_per_class.items():
            result[f"f1_{cls_name.replace(' ', '_').lower()}"] = value
        
        return result
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "SEQUENCE LABELING METRICS",
            "=" * 60,
            "",
            "Standard Metrics:",
            f"  Accuracy:         {self.accuracy:.4f}",
            f"  F1 (micro):       {self.f1_micro:.4f}",
            f"  F1 (macro):       {self.f1_macro:.4f}",
            f"  F1 (weighted):    {self.f1_weighted:.4f}",
            "",
            "Pattern Detection Metrics:",
            f"  Coverage Score:   {self.coverage_score:.4f}",
            f"  Detection Rate:   {self.detection_rate:.4f}",
            f"  False Alarm Rate: {self.false_alarm_rate:.4f}",
            f"  Boundary Acc:     {self.boundary_accuracy:.4f}",
            "",
            "Pattern-Level Metrics:",
            f"  Pattern Precision: {self.pattern_precision:.4f}",
            f"  Pattern Recall:    {self.pattern_recall:.4f}",
            f"  Pattern F1:        {self.pattern_f1:.4f}",
            "",
            f"ROC-AUC (macro):     {self.roc_auc_macro:.4f}",
            "",
            "Per-Class F1:",
        ]
        
        for cls_name, f1 in self.f1_per_class.items():
            lines.append(f"  {cls_name:20s}: {f1:.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels, shape (n_samples,) or (n_samples, seq_len)
        y_pred: Predicted labels, same shape as y_true
        num_classes: Number of classes
        class_names: Optional mapping of class index to name
        
    Returns:
        Dictionary of metrics
    """
    # Use provided class names or default
    if class_names is None:
        class_names = LABEL_NAMES
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    
    # Flatten if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Get present labels (for handling missing classes)
    present_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", labels=present_labels, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=present_labels, zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", labels=present_labels, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", labels=present_labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0),
    }
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    
    for i in range(num_classes):
        name = class_names.get(i, f"Class_{i}")
        metrics[f"f1_{name}"] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        metrics[f"precision_{name}"] = precision_per_class[i] if i < len(precision_per_class) else 0.0
        metrics[f"recall_{name}"] = recall_per_class[i] if i < len(recall_per_class) else 0.0
    
    return metrics


def compute_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """Compute ROC-AUC scores.
    
    Args:
        y_true: Ground truth labels, shape (n_samples,) or flattened
        y_proba: Class probabilities, shape (n_samples, num_classes)
        num_classes: Number of classes
        
    Returns:
        Dictionary with macro and per-class ROC-AUC
    """
    y_true = y_true.flatten()
    
    # One-hot encode for multi-class ROC-AUC
    y_true_onehot = np.zeros((len(y_true), num_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    metrics = {}
    
    try:
        # Macro ROC-AUC
        # Only compute for classes that appear in both y_true and have valid probabilities
        valid_classes = []
        for i in range(num_classes):
            if np.sum(y_true_onehot[:, i]) > 0 and np.sum(y_true_onehot[:, i]) < len(y_true):
                valid_classes.append(i)
        
        if len(valid_classes) > 1:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_true_onehot[:, valid_classes],
                y_proba[:, valid_classes],
                average="macro",
                multi_class="ovr",
            )
        else:
            metrics["roc_auc_macro"] = 0.0
        
        # Per-class ROC-AUC
        for i, name in LABEL_NAMES.items():
            if i in valid_classes:
                try:
                    metrics[f"roc_auc_{name}"] = roc_auc_score(
                        y_true_onehot[:, i], y_proba[:, i]
                    )
                except ValueError:
                    metrics[f"roc_auc_{name}"] = 0.0
            else:
                metrics[f"roc_auc_{name}"] = 0.0
                
    except ValueError:
        # ROC-AUC computation can fail if classes are missing
        metrics["roc_auc_macro"] = 0.0
        for name in LABEL_NAMES.values():
            metrics[f"roc_auc_{name}"] = 0.0
    
    return metrics


def find_pattern_regions(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Find contiguous pattern regions (non-zero labels).
    
    Args:
        labels: 1D array of labels for a single sequence
        
    Returns:
        List of (start, end, label) tuples for each pattern region
    """
    regions = []
    in_region = False
    start = 0
    current_label = 0
    
    for i, label in enumerate(labels):
        if label > 0:  # Pattern (not None)
            if not in_region:
                start = i
                current_label = label
                in_region = True
            elif label != current_label:
                # Different pattern, end current and start new
                regions.append((start, i - 1, current_label))
                start = i
                current_label = label
        else:
            if in_region:
                regions.append((start, i - 1, current_label))
                in_region = False
    
    # Handle region at end
    if in_region:
        regions.append((start, len(labels) - 1, current_label))
    
    return regions


def compute_coverage_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute IoU-like coverage score for pattern regions.
    
    Measures overlap between predicted and true pattern regions,
    ignoring class labels (just pattern vs non-pattern).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Coverage score (0-1), where 1 is perfect overlap
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Convert to binary (pattern vs no pattern)
    true_pattern = y_true > 0
    pred_pattern = y_pred > 0
    
    intersection = np.sum(true_pattern & pred_pattern)
    union = np.sum(true_pattern | pred_pattern)
    
    if union == 0:
        return 1.0  # No patterns in either
    
    return intersection / union


def compute_detection_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_overlap: float = 0.5,
) -> Tuple[float, float]:
    """Compute pattern detection and false alarm rates.
    
    A true pattern is "detected" if at least min_overlap of its timesteps
    are predicted as pattern (any class).
    
    Args:
        y_true: Ground truth labels, shape (n_samples, seq_len) or (seq_len,)
        y_pred: Predicted labels, same shape
        min_overlap: Minimum overlap ratio to count as detected
        
    Returns:
        Tuple of (detection_rate, false_alarm_rate)
    """
    # Handle both batch and single sequence
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    
    total_true_patterns = 0
    detected_patterns = 0
    total_pred_patterns = 0
    false_alarms = 0
    
    for seq_idx in range(y_true.shape[0]):
        true_regions = find_pattern_regions(y_true[seq_idx])
        pred_regions = find_pattern_regions(y_pred[seq_idx])
        
        # Check detection of true patterns
        for start, end, label in true_regions:
            total_true_patterns += 1
            region_len = end - start + 1
            pred_in_region = y_pred[seq_idx, start:end+1] > 0
            overlap = np.sum(pred_in_region) / region_len
            if overlap >= min_overlap:
                detected_patterns += 1
        
        # Check false alarms (predicted patterns with no true pattern)
        for start, end, label in pred_regions:
            total_pred_patterns += 1
            true_in_region = y_true[seq_idx, start:end+1] > 0
            if np.sum(true_in_region) == 0:
                false_alarms += 1
    
    detection_rate = detected_patterns / total_true_patterns if total_true_patterns > 0 else 1.0
    false_alarm_rate = false_alarms / total_pred_patterns if total_pred_patterns > 0 else 0.0
    
    return detection_rate, false_alarm_rate


def compute_boundary_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 3,
) -> float:
    """Compute accuracy of pattern boundary detection.
    
    Measures how accurately the model predicts pattern start/end positions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        tolerance: Number of timesteps tolerance for boundary matching
        
    Returns:
        Boundary accuracy (0-1)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    
    total_boundaries = 0
    matched_boundaries = 0
    
    for seq_idx in range(y_true.shape[0]):
        true_regions = find_pattern_regions(y_true[seq_idx])
        pred_regions = find_pattern_regions(y_pred[seq_idx])
        
        # Extract boundary positions
        true_boundaries = set()
        for start, end, _ in true_regions:
            true_boundaries.add(("start", start))
            true_boundaries.add(("end", end))
        
        pred_boundaries = []
        for start, end, _ in pred_regions:
            pred_boundaries.append(("start", start))
            pred_boundaries.append(("end", end))
        
        total_boundaries += len(true_boundaries)
        
        # Match predicted boundaries to true boundaries
        for btype, bpos in pred_boundaries:
            for true_btype, true_bpos in true_boundaries:
                if btype == true_btype and abs(bpos - true_bpos) <= tolerance:
                    matched_boundaries += 1
                    break
    
    if total_boundaries == 0:
        return 1.0
    
    return matched_boundaries / total_boundaries


def compute_pattern_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute pattern-level precision, recall, and F1.
    
    Treats each contiguous pattern region as a single "object" and
    computes object-level detection metrics (similar to object detection).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        iou_threshold: IoU threshold for matching patterns
        
    Returns:
        Dict with pattern_precision, pattern_recall, pattern_f1
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    
    total_true = 0
    total_pred = 0
    true_positives = 0
    
    for seq_idx in range(y_true.shape[0]):
        true_regions = find_pattern_regions(y_true[seq_idx])
        pred_regions = find_pattern_regions(y_pred[seq_idx])
        
        total_true += len(true_regions)
        total_pred += len(pred_regions)
        
        # Match predictions to ground truth
        matched_true = set()
        for pred_start, pred_end, pred_label in pred_regions:
            best_iou = 0
            best_match = None
            
            for true_idx, (true_start, true_end, true_label) in enumerate(true_regions):
                if true_idx in matched_true:
                    continue
                
                # Compute IoU
                inter_start = max(pred_start, true_start)
                inter_end = min(pred_end, true_end)
                
                if inter_end >= inter_start:
                    intersection = inter_end - inter_start + 1
                    union = (pred_end - pred_start + 1) + (true_end - true_start + 1) - intersection
                    iou = intersection / union
                    
                    # Also check class match
                    if iou > best_iou and pred_label == true_label:
                        best_iou = iou
                        best_match = true_idx
            
            if best_iou >= iou_threshold and best_match is not None:
                true_positives += 1
                matched_true.add(best_match)
    
    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "pattern_precision": precision,
        "pattern_recall": recall,
        "pattern_f1": f1,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    num_classes: int = NUM_CLASSES,
    class_names: Optional[Dict[int, str]] = None,
) -> SequenceMetrics:
    """Compute all sequence labeling metrics.
    
    Args:
        y_true: Ground truth labels, shape (n_samples, seq_len)
        y_pred: Predicted labels, same shape
        y_proba: Optional class probabilities, shape (n_samples, seq_len, num_classes)
        num_classes: Number of classes
        class_names: Optional mapping of class index to name. If None, uses LABEL_NAMES.
        
    Returns:
        SequenceMetrics object with all computed metrics
    """
    # Use provided class names or default to LABEL_NAMES
    if class_names is None:
        class_names = LABEL_NAMES
    else:
        # Convert list to dict if needed
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
    
    metrics = SequenceMetrics()
    
    # Flatten for standard metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Standard metrics - pass class_names for proper per-class naming
    std_metrics = compute_standard_metrics(y_true_flat, y_pred_flat, num_classes, class_names)
    metrics.accuracy = std_metrics["accuracy"]
    metrics.f1_micro = std_metrics["f1_micro"]
    metrics.f1_macro = std_metrics["f1_macro"]
    metrics.f1_weighted = std_metrics["f1_weighted"]
    metrics.precision_micro = std_metrics["precision_micro"]
    metrics.precision_macro = std_metrics["precision_macro"]
    metrics.recall_micro = std_metrics["recall_micro"]
    metrics.recall_macro = std_metrics["recall_macro"]
    
    # Per-class metrics - only for classes that exist
    for i in range(num_classes):
        name = class_names.get(i, f"Class_{i}")
        metrics.f1_per_class[name] = std_metrics.get(f"f1_{name}", 0.0)
        metrics.precision_per_class[name] = std_metrics.get(f"precision_{name}", 0.0)
        metrics.recall_per_class[name] = std_metrics.get(f"recall_{name}", 0.0)
    
    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        y_proba_flat = y_proba.reshape(-1, num_classes)
        roc_metrics = compute_roc_auc(y_true_flat, y_proba_flat, num_classes)
        metrics.roc_auc_macro = roc_metrics["roc_auc_macro"]
        for i in range(num_classes):
            name = class_names.get(i, f"Class_{i}")
            metrics.roc_auc_per_class[name] = roc_metrics.get(f"roc_auc_{name}", 0.0)
    
    # Pattern-specific metrics
    metrics.coverage_score = compute_coverage_score(y_true, y_pred)
    
    detection_rate, false_alarm_rate = compute_detection_rate(y_true, y_pred)
    metrics.detection_rate = detection_rate
    metrics.false_alarm_rate = false_alarm_rate
    
    metrics.boundary_accuracy = compute_boundary_accuracy(y_true, y_pred)
    
    # Pattern-level metrics
    pattern_metrics = compute_pattern_level_metrics(y_true, y_pred)
    metrics.pattern_precision = pattern_metrics["pattern_precision"]
    metrics.pattern_recall = pattern_metrics["pattern_recall"]
    metrics.pattern_f1 = pattern_metrics["pattern_f1"]
    
    # Confusion matrix
    metrics.confusion_matrix = confusion_matrix(
        y_true_flat, y_pred_flat, labels=range(num_classes)
    )
    
    return metrics


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create sample data
    n_samples, seq_len = 10, 256
    y_true = np.zeros((n_samples, seq_len), dtype=np.int64)
    y_pred = np.zeros((n_samples, seq_len), dtype=np.int64)
    
    # Add some patterns
    for i in range(n_samples):
        # True patterns
        start = np.random.randint(10, 100)
        length = np.random.randint(20, 50)
        label = np.random.randint(1, 7)
        y_true[i, start:start+length] = label
        
        # Predicted patterns (with some noise)
        pred_start = start + np.random.randint(-5, 5)
        pred_length = length + np.random.randint(-10, 10)
        pred_label = label if np.random.random() > 0.3 else np.random.randint(1, 7)
        y_pred[i, max(0, pred_start):min(seq_len, pred_start+pred_length)] = pred_label
    
    # Compute metrics
    y_proba = np.random.rand(n_samples, seq_len, NUM_CLASSES)
    y_proba = y_proba / y_proba.sum(axis=-1, keepdims=True)  # Normalize
    
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    print(metrics.summary())
