import pandas as pd
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
import matplotlib.pyplot as plt
from gluonts.dataset.repository import dataset_recipes

from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
# from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


def run_moirai(
    csv_path: str,
    ctx_length: int,
    pred_length: int,
    batch_size: int = 32,
    test_length: int =24*7,
    model_size: str = "small",  # kept for compatibility; hardcoded
    patch_size: str = "auto",   # used only in Moirai
    model_name: str = "moirai",
    target_column: str= "value",
) -> pd.DataFrame:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Load and preprocess CSV
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("H").ffill()
    df["item_id"] = 0

    ds = PandasDataset.from_long_dataframe(df, target=target_column, item_id="item_id")

    # Split into train and test
    TEST = test_length
    train, test_template = split(ds, offset=-TEST)
    test_data = test_template.generate_instances(
        prediction_length=pred_length,
        windows=TEST // pred_length,
        distance=pred_length
    )
    print("start")
    # Load model using fixed IDs (no dynamic model_size!)
    if model_name == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
            prediction_length=pred_length,
            context_length=ctx_length,
            patch_size=patch_size if patch_size != "auto" else 32,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif model_name == "moirai-moe":
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
            prediction_length=pred_length,
            context_length=ctx_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif model_name == "moirai2":
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                f"Salesforce/moirai-2.0-R-small"
            ),
            prediction_length=pred_length,
            context_length=ctx_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    else:
        raise ValueError("model_name must be 'moirai', 'moirai-moe' or 'moirai2'")

    # Predict
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = predictor.predict(test_data.input)

    prediction_results = []
    for forecast in forecasts:
        start_date = pd.Period(forecast.start_date).to_timestamp()
        pred_len = len(forecast.mean)
        timestamps = pd.date_range(start=start_date, periods=pred_len, freq="H")
        for timestamp, pred_value in zip(timestamps, forecast.mean):
            prediction_results.append({"time": timestamp, "predicted_value": pred_value})

    return pd.DataFrame(prediction_results)
