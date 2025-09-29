import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline

def run_chronos(
    csv_path: str,
    ctx_length: int,
    pred_length: int,
    batch_size: int = 32,
    test_length: int = 24*7,
    model_size: str = "small",
    patch_size: str = "auto",
    model_name: str = "chronos",
    target_column: str= "value",
    quantile: float = 0.9,
) -> pd.DataFrame:
    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    df = df.sort_index()
    df.index = df.index.floor("h") # type: ignore

    # Load pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    # Reset index for easier assignment
    df = df.reset_index().rename(columns={"time": "time"})
    df["predicted_value"] = np.nan
    df["quantile_lower"] = np.nan
    df["quantile_upper"] = np.nan

    values = df[target_column].values.astype(np.float32)

    # Start context with last ctx_length values before test set
    start_idx = len(values) - test_length - ctx_length
    context = torch.tensor(values[start_idx:start_idx + ctx_length], dtype=torch.float32)

    preds = []
    quantile_lower_preds = []
    quantile_upper_preds = []
    lower_quantile = (1 - quantile) / 2
    upper_quantile = 1 - lower_quantile
    
    steps_done = 0
    while steps_done < test_length:
        steps = min(pred_length, test_length - steps_done)
        quantiles, _ = pipeline.predict_quantiles(
            context=context,
            prediction_length=steps,
            quantile_levels=[lower_quantile, 0.5, upper_quantile],
        )
        y_pred = quantiles[0, :, 1].numpy()  # median (0.5 quantile)
        y_lower = quantiles[0, :, 0].numpy()  # lower quantile
        y_upper = quantiles[0, :, 2].numpy()  # upper quantile
        
        preds.extend(y_pred)
        quantile_lower_preds.extend(y_lower)
        quantile_upper_preds.extend(y_upper)

        # Update context with new preds
        context = torch.cat([context, torch.tensor(y_pred, dtype=torch.float32)])
        context = context[-ctx_length:]

        steps_done += steps

    # Assign preds to the last test_length rows
    df.loc[df.index[-test_length:], "predicted_value"] = preds
    df.loc[df.index[-test_length:], "quantile_lower"] = quantile_lower_preds
    df.loc[df.index[-test_length:], "quantile_upper"] = quantile_upper_preds

    # Ensure column order
    result = df[["time", "predicted_value", "quantile_lower", "quantile_upper"]]
    return result
