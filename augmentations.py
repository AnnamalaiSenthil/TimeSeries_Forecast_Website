import pandas as pd
import numpy as np

def convex_hull_method(series, slope_limit=1000000000):
    """
    Apply the Convex Hull Method augmentation as described.
    """
    NO_OF_ENTRIES = len(series)
    new_value = []
    cur = series.iloc[0]
    RELAXATION_TP = 10 # assuming a value for RELAXATION_TP, you can adjust this if needed
    
    for i in range(NO_OF_ENTRIES):
        vmax = cur
        imax = i
        new_value.append(cur)
        slope = -slope_limit
        for j in range(i, min(i + RELAXATION_TP, NO_OF_ENTRIES)):
            if (series.iloc[j] - cur) / (j - i + 1) > slope:
                slope = (series.iloc[j] - cur) / (j - i + 1)
                vmax = series.iloc[j]
                imax = j
        cur += slope
    return pd.Series(new_value, index=series.index)


def rolling_means(series, window_size=5):
    """
    Apply Rolling Means augmentation.
    """
    return series.rolling(window=window_size, min_periods=1).mean()


def peak_multiplier(series, peak_multiplier=1.5, peak_threshold=8, hill_dist=5):
    """
    Modify peaks in the series based on the provided algorithm.
    """
    modified = series.copy()
    is_peak = [False] * len(series)
    
    for i in range(1, len(series)):
        if (series.iloc[i] - series.iloc[i-1]) > peak_threshold:
            is_peak[i] = True
        elif is_peak[i-1] and (series.iloc[i] > series.iloc[i-1]):
            is_peak[i] = True
    
    for i in range(len(series)):
        if is_peak[i]:
            modified.iloc[i] = series.iloc[i] * peak_multiplier
    
    mod = modified.copy()
    for i in range(len(modified)):
        for j in range(1, hill_dist):
            if i + j < len(modified):
                mod.iloc[i] = max(mod.iloc[i], modified.iloc[i+j] * (1 - (j / hill_dist)))
            if i - j >= 0:
                mod.iloc[i] = max(mod.iloc[i], modified.iloc[i-j] * (1 - (j / hill_dist)))
    modified = mod.copy()
    return modified


def apply_augmentation(csv_path: str, output_path: str, aug_type: str):
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    
    if "value" not in df.columns:
        raise KeyError("Input CSV must contain a 'value' column.")
    
    original_series = df["value"]
    augmented_series = None
    
    if aug_type == "convex_hull_method":
        augmented_series = convex_hull_method(original_series)
    elif aug_type == "rolling_means":
        augmented_series = rolling_means(original_series)
    elif aug_type == "peak_multiplier":
        augmented_series = peak_multiplier(original_series)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    df["augmented_value"] = augmented_series
    
    # Save the new DataFrame with the added column
    df.to_csv(output_path, index=True)
