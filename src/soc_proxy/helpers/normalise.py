import pandas as pd
import numpy as np

def normalise_by_method(data, method="mean"):
    """
    Normalizes the input (list, dict, or pandas Series) using one of:
    - method="sum": values sum to 1
    - method="mean": values are scaled by mean
    - method="max": values are scaled by max

    Returns a new object of the same type. Does not modify input in-place.
    """

    if method not in {"sum", "mean", "max"}:
        raise ValueError("Method must be 'sum', 'mean', or 'max'")

    # Convert input to Series
    if isinstance(data, dict):
        s = pd.Series(data)
        output_type = 'dict'
    elif isinstance(data, (list, np.ndarray)):
        s = pd.Series(data)
        output_type = 'list'
    elif isinstance(data, pd.Series):
        s = data.astype(float)
        output_type = 'series'
    else:
        raise TypeError("Input must be a list, dictionary, or pandas Series.")

    # Compute normalization factor
    factor = {
        "sum": s.sum(),
        "mean": s.mean(),
        "max": s.max()
    }[method]

    if factor == 0:
        raise ValueError("Normalization factor is zero — cannot normalize.")

    normed = s / factor

    # Convert back to original type
    if output_type == 'dict':
        return normed.to_dict(),factor
    elif output_type == 'list':
        return normed.tolist(), factor
    elif output_type == 'series':
        return normed, factor
