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
) -> pd.DataFrame:
    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    df = df.sort_index()
    df.index = df.index.floor("h")

    # Load pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    # Reset index for easier assignment
    df = df.reset_index().rename(columns={"time": "time"})
    df["predicted_value"] = np.nan

    values = df[target_column].values.astype(np.float32)

    # Start context with last ctx_length values before test set
    start_idx = len(values) - test_length - ctx_length
    context = torch.tensor(values[start_idx:start_idx + ctx_length], dtype=torch.float32)

    preds = []
    steps_done = 0
    while steps_done < test_length:
        steps = min(pred_length, test_length - steps_done)
        quantiles, _ = pipeline.predict_quantiles(
            context=context,
            prediction_length=steps,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        y_pred = quantiles[0, :, 1].numpy()
        preds.extend(y_pred)

        # Update context with new preds
        context = torch.cat([context, torch.tensor(y_pred, dtype=torch.float32)])
        context = context[-ctx_length:]

        steps_done += steps

    # Assign preds to the last test_length rows
    df.loc[df.index[-test_length:], "predicted_value"] = preds

    # Ensure column order
    result = df[["time", "predicted_value"]]
    return result
