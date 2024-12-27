import torch
import numpy as np
from typing import List, Dict, Tuple

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) between two boxes."""
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    x1_2, y1_2, x2_2, y2_2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    
    # Calculate intersection area
    x1_i = torch.maximum(x1_1, x1_2)
    y1_i = torch.maximum(y1_1, y1_2)
    x2_i = torch.minimum(x2_1, x2_2)
    y2_i = torch.minimum(y2_1, y2_2)
    
    w_i = torch.maximum(torch.zeros_like(x2_i), x2_i - x1_i)
    h_i = torch.maximum(torch.zeros_like(y2_i), y2_i - y1_i)
    intersection = w_i * h_i
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def calculate_map(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_classes: torch.Tensor,
    true_boxes: torch.Tensor,
    true_classes: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """Calculate mean Average Precision (mAP) for object detection."""
    # Initialize metrics
    aps = []
    class_aps = {}
    
    # Calculate AP for each class
    for class_id in range(num_classes):
        # Get predictions and ground truth for this class
        class_mask_pred = pred_classes == class_id
        class_mask_true = true_classes == class_id
        
        if not torch.any(class_mask_true):
            continue
        
        # Get boxes and scores for this class
        class_boxes_pred = pred_boxes[class_mask_pred]
        class_scores_pred = pred_scores[class_mask_pred]
        class_boxes_true = true_boxes[class_mask_true]
        
        if len(class_boxes_pred) == 0:
            aps.append(0)
            class_aps[str(class_id)] = 0
            continue
        
        # Sort predictions by confidence
        sorted_indices = torch.argsort(class_scores_pred, descending=True)
        class_boxes_pred = class_boxes_pred[sorted_indices]
        class_scores_pred = class_scores_pred[sorted_indices]
        
        # Calculate precision and recall
        num_true = len(class_boxes_true)
        num_pred = len(class_boxes_pred)
        
        tp = torch.zeros(num_pred)
        fp = torch.zeros(num_pred)
        matched = torch.zeros(num_true, dtype=torch.bool)
        
        # Match predictions to ground truth
        for pred_idx in range(num_pred):
            pred_box = class_boxes_pred[pred_idx]
            
            # Calculate IoU with all unmatched ground truth boxes
            ious = calculate_iou(
                pred_box.unsqueeze(0),
                class_boxes_true[~matched]
            )
            
            if len(ious) > 0:
                max_iou, max_idx = torch.max(ious, dim=0)
                if max_iou >= iou_threshold:
                    tp[pred_idx] = 1
                    matched[torch.where(~matched)[0][max_idx]] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Calculate precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / (num_true + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if torch.any(recalls >= t):
                ap += torch.max(precisions[recalls >= t])
        ap = ap / 11
        
        aps.append(ap.item())
        class_aps[str(class_id)] = ap.item()
    
    # Calculate mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0
    
    return mAP, class_aps

def calculate_detection_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    num_classes: int
) -> Dict[str, float]:
    """Calculate all detection metrics."""
    metrics = {}
    
    # Calculate mAP
    mAP, class_aps = calculate_map(
        predictions['boxes'],
        predictions['confidence'],
        predictions['class_logits'].argmax(dim=-1),
        targets['boxes'],
        targets['classes'],
        num_classes
    )
    
    metrics['mAP'] = mAP
    for class_id, ap in class_aps.items():
        metrics[f'AP_class_{class_id}'] = ap
    
    # Calculate average confidence score
    metrics['avg_confidence'] = predictions['confidence'].mean().item()
    
    # Calculate average IoU with ground truth
    pred_boxes = predictions['boxes']
    true_boxes = targets['boxes']
    
    if len(pred_boxes) > 0 and len(true_boxes) > 0:
        ious = calculate_iou(pred_boxes, true_boxes)
        metrics['avg_iou'] = ious.mean().item()
    else:
        metrics['avg_iou'] = 0.0
    
    return metrics 