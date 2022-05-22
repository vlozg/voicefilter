from typing import Any, Dict, List, Optional, Tuple, Union

from torch import Tensor

from torchmetrics import ROC

from scipy.optimize import brentq
from scipy.interpolate import interp1d

class EER(ROC):
    """Computes the Equal Error Rate (EER) for verification problem.
    The input can only be binary or multiclass with probabilities.
    Forward accepts
    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass/multilabel) tensor
      with probabilities, where C is the number of classes/labels.
    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels
    """

    def __init__(
        self,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, pos_label=1, **kwargs)

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute the EER base on ROC.
        Returns:
            2-element tuple containing
            eer:
                tensor with equal error rate.
            threshold:
                thresholds used for computing false- and true postive rates
        """
        fpr, tpr, thresholds = super().compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        thresholds = thresholds.cpu().numpy()

        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        
        return eer, thresh