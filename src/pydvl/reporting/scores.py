from typing import Dict, Iterable, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pydvl.utils import Utility, maybe_progress
from pydvl.value.result import ValuationResult


__all__ = ["compute_removal_score"]


def compute_removal_score(
    u: Utility,
    values: ValuationResult,
    percentages: Union[NDArray[np.float_], Iterable[float]],
    *,
    remove_best: bool = False,
    progress: bool = False,
) -> Dict[float, float]:
    r"""Fits model and computes score on the test set after incrementally removing
    a percentage of data points from the training set, based on their values.

    Args:
        u: Utility object with model, data, and scoring function.
        values: Data values of data instances in the training set.
        percentages: Sequence of removal percentages.
        remove_best: If True, removes data points in order of decreasing valuation.
        progress: If True, display a progress bar.

    Returns:
        Dictionary that maps the percentages to their respective scores.
    """
    # Sanity checks
    if np.any([x >= 1.0 or x < 0.0 for x in percentages]):
        raise ValueError("All percentages should be in the range [0.0, 1.0)")

    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values) }, should be equal to the number of data indices, {len(u.data.indices)}"
        )

    scores = {}

    # We sort in descending order if we want to remove the best values
    values.sort(reverse=remove_best)

    for pct in maybe_progress(percentages, display=progress, desc="Removal Scores"):
        n_removal = int(pct * len(u.data))
        indices = values.indices[n_removal:]
        score = u(indices)
        scores[pct] = score
    return scores


def compute_removal_class_imbal_evol(
    u: Utility,
    values: ValuationResult,
    percentages: Union[NDArray[np.float_], Iterable[float]],
    *,
    remove_best: bool = False,
    progress: bool = False,
) -> Dict[float, float]:
    
    # Sanity checks
    if np.any([x >= 1.0 or x < 0.0 for x in percentages]):
        raise ValueError("All percentages should be in the range [0.0, 1.0)")

    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values) }, should be equal to the number of data indices, {len(u.data.indices)}"
        )

    scores = {}
    classes = np.unique(u.data.y_train)
    # We sort in descending order if we want to remove the best values
    values.sort(reverse=remove_best)
    all_classes_bal = []
    for pct in maybe_progress(percentages, display=progress, desc="Removal Scores"):
        n_removal = int(pct * len(u.data))
        indices = values.indices[n_removal:]
        labels = u.data.y_train[indices]
        classes_bal = {c:sum([x==c for x in labels]) for c in classes}
        all_classes_bal.append(classes_bal)
    return pd.DataFrame(all_classes_bal, index = percentages)
