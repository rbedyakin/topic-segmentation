from torchmetrics import Metric
import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional, Union, Callable, Any
from torchmetrics.utilities.compute import _safe_divide
from collections import Counter


def precision_recall_f1(
        pred_sum: torch.Tensor, tp_sum: torch.Tensor, true_sum: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision, recall, f1.

        Args:
            pred_sum: Tensor of predicted positives per type.
            tp_sum: Tensor of true positives per type.
            true_sum: Tensor of actual positives per type.
        
        Returns:
            precision: Tensor of precision per type.
            recall: Tensor of recall per type. 
            f1: Tensor of f1 per type.
        
        Note: Metric value is substituted as 0 when encountering zero division."""

    precision = _safe_divide(num=tp_sum, denom=pred_sum, zero_division=0.0)
    recall = _safe_divide(num=tp_sum, denom=true_sum, zero_division=0.0)
    f1 = _safe_divide(num=2 * tp_sum,
                      denom=pred_sum + true_sum,
                      zero_division=0.0)

    return precision, recall, f1


def update_pk(ref: List[int],
              hyp: List[int],
              window_size: Union[int, str] = 'auto') -> float:
    """Compute the Pk metric for a pair of segmentations. 

        A segmentation is any sequence of zeros or ones, 
        where 1 the specified boundary value is used to mark
        the edge of a segmentation.

        Args:
            ref: the reference segmentation, list of int
            hyp: the segmentation to evaluate, list of int
            window_size: window size, 'auto' or int
        
        Returns:
            pk: float

        >>> update_pk([0,1,0,0]*100, [1]*400, 2)
        0.50
        >>> update_pk([0,1,0,0]*100, [0]*400, 2)
        0.50
        >>> update_pk([0,1,0,0]*100, [0,1,0,0]*100, 2)
        0.00
    """

    if window_size == 'auto':
        n_segment = ref.count(1) + 1  # 00000100100100
        aver_length_of_segment = len(ref) / n_segment / 2
        k = max(2, int(round(aver_length_of_segment)))
    else:
        k = window_size

    # err = 0
    # for i in range(len(ref) - k + 1):
    #     err += (1 in ref[i:i + k]) is not (1 in hyp[i:i + k])
    # return err / (len(ref) - k + 1.0)

    sum_differences = 0
    # Slide window over and sum the number of varying windows
    measurements = 0
    for i in range(0, len(ref) - k):
        # Create probe windows with k boundaries inside
        window_ref = ref[i:i + k + 1]
        window_hyp = hyp[i:i + k + 1]
        # Probe agreement
        agree_ref = window_ref[0] is window_ref[-1]
        agree_hyp = window_hyp[0] is window_hyp[-1]
        # If the windows agreements agree
        if agree_ref is not agree_hyp:
            sum_differences += 1
        measurements += 1
    # Perform final division
    value = sum_differences / measurements if measurements > 0 else 0
    return value


def update_windowdiff(ref: List[int],
                      hyp: List[int],
                      window_size: Union[int, str] = 'auto',
                      weighted: bool = False) -> float:
    """Compute the windowdiff metric for a pair of segmentations. 

        A segmentation is any sequence of zeros or ones, 
        where 1 the specified boundary value is used to mark
        the edge of a segmentation.

        Args:
            ref: the reference segmentation, list of int
            hyp: the segmentation to evaluate, list of int
            window_size: window size, 'auto' or int
            weighted: use the weighted variant of windowdiff, boolean
        
        Returns:
            windowdiff: float

        >>> update_windowdiff([0,0,0,1,0,0,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,0,0,1,0], 3)
        0.00
        >>> update_windowdiff([0,0,0,1,0,0,0,0,0,0,1,0], [0,0,0,0,1,0,0,0,0,1,0,0], 3)
        0.30
        >>> update_windowdiff([0,0,0,0,1,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,1,0,0,0,0], 3)
        0.80
    """

    if window_size == 'auto':
        n_segment = ref.count(1) + 1  # 00000100100100
        aver_length_of_segment = len(ref) / n_segment / 2
        k = max(2, int(round(aver_length_of_segment)))
    else:
        k = window_size

    if len(ref) != len(hyp):
        raise ValueError("Segmentations have unequal length")
    if k > len(ref):
        raise ValueError(
            "Window width k should be smaller or equal than segmentation lengths"
        )
    wd = 0
    for i in range(len(ref) - k + 1):
        ndiff = abs(ref[i:i + k].count(1) - hyp[i:i + k].count(1))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (len(ref) - k + 1.0)


class SegmentationEvaluation(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self,
                 stage: Optional[str] = None,
                 window_size: Union[int, str] = 'auto',
                 **kwargs):
        """Init Metric

        Args:
            stage: Optional prefix for keys in output dict
                default: None
        """
        super().__init__(**kwargs)

        self.window_size = window_size

        if stage:
            self.stage = f"{stage}_"
        else:
            self.stage = ''

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("sum_pk",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("sum_windowdiff",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("sum_abs_error",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")

        self.add_state("pred_sum",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("tp_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_sum",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds: List[List[int]], target: List[List[int]]) -> None:
        """Update state with predictions and targets.
        
        Args:
            preds: List of predictions (Estimated targets as returned by a tagger)
            target: List of reference (Ground truth (correct) target values)
        """
        pred_tensor = torch.tensor(preds)
        target_tensor = torch.tensor(target)
        self.pred_sum += pred_tensor.sum()
        self.tp_sum += (target_tensor * pred_tensor).sum()
        self.true_sum += target_tensor.sum()
        self.sum_abs_error += torch.abs(target_tensor - pred_tensor).sum()

        for pred, gt in zip(preds, target):
            self.total += torch.tensor(1)

            self.sum_pk += update_pk(ref=gt,
                                     hyp=pred,
                                     window_size=self.window_size)
            self.sum_windowdiff += update_windowdiff(
                ref=gt, hyp=pred, window_size=self.window_size)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute the final statistics.
        
        Returns:
            'metrics': dict. Summary of the scores."""

        metrics = {}
        pk = _safe_divide(num=self.sum_pk, denom=self.total, zero_division=0.0)
        metrics[f'{self.stage}pk'] = pk

        windowdiff = _safe_divide(num=self.sum_windowdiff,
                                  denom=self.total,
                                  zero_division=0.0)
        metrics[f'{self.stage}windowdiff'] = windowdiff
        metrics[f'{self.stage}mae'] = _safe_divide(num=self.sum_abs_error,
                                                   denom=self.total,
                                                   zero_division=0.0)

        precision, recall, f1 = precision_recall_f1(pred_sum=self.pred_sum,
                                                    tp_sum=self.tp_sum,
                                                    true_sum=self.true_sum)
        metrics[f'{self.stage}overall_precision'] = precision
        metrics[f'{self.stage}overall_recall'] = recall
        metrics[f'{self.stage}overall_f1'] = f1

        metrics[f'{self.stage}total_score'] = torch.tensor(0.5) * f1 \
            + torch.tensor(0.25) * (torch.tensor(1.0) - pk) \
            + torch.tensor(0.25) * (torch.tensor(1.0) - windowdiff)

        return metrics


if __name__ == "__main__":
    SegEval = SegmentationEvaluation(window_size='auto')
    SegEval.update(preds=[[1] * 400], target=[[0, 1, 0, 0] * 100])
    print(SegEval.compute())
