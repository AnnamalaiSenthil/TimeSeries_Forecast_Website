import pandas as pd
import numpy as np
import subprocess
import sys
import tempfile
import os
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import uuid

# Moirai and Chronos runners are assumed to be in the same directory
from moirai_runner import run_moirai
from chronos_runner import run_chronos

def ModelChooser(script_name):
    if script_name == "moirai-moe":
        return "moirai-moe"
    elif script_name == "chronos":
        return "chronos"
    elif script_name == "moirai2":
        return "moirai2"
    elif script_name == "moirai":
        return "moirai"
    else:
        raise ValueError(f"Unknown script: {script_name}")

def forecast_with_metrics(
    csv_path: str,
    original_csv_path: str,
    model: str = "moirai",
    target_column: str = "value",
    ctx_length: int = 168,
    pred_length: int = 24,
    test_length: int = 24*7,
    batch_size: int = 32,
    model_size: str = "small",
    patch_size: str = "auto",
    output_path: str = "predicted_data.csv",
    quantile: float = 0.99
) -> tuple[pd.DataFrame, dict]:
    model_map = {
        "moirai2": run_moirai,
        "moirai": run_moirai,
        "moirai-moe": run_moirai,
        "chronos": run_chronos,
    }
    
    if model not in model_map:
        raise ValueError(f"Unknown model: {model}")

    # Run the model on the prepared CSV with the specified target column
    df_forecast = model_map[model](
        csv_path=csv_path,
        ctx_length=ctx_length,
        pred_length=pred_length,
        target_column=target_column,
        batch_size=batch_size,
        test_length=test_length,
        model_size=model_size,
        patch_size=patch_size,
        model_name=model,
        quantile=quantile,
    )

    # Load original data for metric calculation and plotting
    original_df = pd.read_csv(original_csv_path, parse_dates=["time"])
    original_df = original_df.sort_values("time")
    
    # Merge forecast with original data for metric calculation
    merged_df = pd.merge(
        original_df.rename(columns={'time':'time'}),
        df_forecast,
        on='time',
        how='inner'
    ).dropna(subset=["value", "predicted_value"])

    # If the forecast was based on augmented data, merge that column as well
    if target_column == 'augmented_value':
        augmented_df = pd.read_csv(csv_path, parse_dates=['time'])
        merged_df = pd.merge(merged_df, augmented_df[['time', 'augmented_value']], on='time', how='left')

    actual = np.array(merged_df["value"].values, dtype=float)
    pred = np.array(merged_df["predicted_value"].values, dtype=float)

    # Calculate underpredictions outside quantile bounds if quantile columns exist
    underpredictions_outside_bounds = 0
    if 'quantile_upper' in merged_df.columns:
        quantile_upper = np.array(merged_df["quantile_upper"].values, dtype=float)
        # Count actual values greater than upper quantile bound (underpredictions outside bounds)
        underpredictions_outside_bounds = np.sum(actual > quantile_upper)

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred, squared=False)
    r2 = r2_score(actual, pred)
    smape = 100 * np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred)) / 2))
    mape = 100 * np.mean(np.abs((actual - pred) / actual))
    underpredictions = np.sum(pred < actual)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "SMAPE": smape,
        "MAPE": mape,
        "Underpredictions": underpredictions,
        "Underpredictions Outside Bounds": underpredictions_outside_bounds,
    }

    return merged_df, metrics


def plot(df_to_plot: pd.DataFrame, output_path: str, title: str = "Forecast: Predicted vs. Actual Values"):
    """
    Plots the predicted and actual values from a DataFrame, including quantile bounds.

    Args:
        df_to_plot (pd.DataFrame): The DataFrame containing 'value' and 'predicted_value' columns.
        output_path (str): The path to save the plot.
        title (str): The title of the plot.
    """
    try:
        # Check for required columns
        if 'value' not in df_to_plot.columns or 'predicted_value' not in df_to_plot.columns:
            raise KeyError("The DataFrame must contain 'value' and 'predicted_value' columns.")

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot the 'value' (normal) and 'predicted_value'
        plt.plot(df_to_plot.index, df_to_plot['value'], label='Actual Value', marker='o')
        plt.plot(df_to_plot.index, df_to_plot['predicted_value'], label='Predicted Value', marker='x')

        # Add quantile bounds if they exist
        if 'quantile_lower' in df_to_plot.columns and 'quantile_upper' in df_to_plot.columns:
            plt.fill_between(df_to_plot.index, 
                           df_to_plot['quantile_lower'], 
                           df_to_plot['quantile_upper'],
                           alpha=0.3, label='Quantile Bounds', color='gray')
            plt.plot(df_to_plot.index, df_to_plot['quantile_lower'], 
                    linestyle='--', alpha=0.7, color='gray')
            plt.plot(df_to_plot.index, df_to_plot['quantile_upper'], 
                    linestyle='--', alpha=0.7, color='gray')

            # Mark underpredictions outside quantile bounds with red vertical dashed lines
            outside_bounds_mask = df_to_plot['value'] > df_to_plot['quantile_upper']
            if outside_bounds_mask.any():
                # Get y-axis limits for full vertical lines
                y_min, y_max = plt.ylim()
                
                # Draw full vertical dashed red lines for each point outside bounds
                outside_indices = df_to_plot.index[outside_bounds_mask]
                for idx in outside_indices:
                    plt.axvline(x=idx, color='red', linestyle='--', linewidth=1, alpha=0.8)
                
                # Add a single label for the legend (only once)
                plt.axvline(x=outside_indices[0], color='red', linestyle='--', linewidth=1, 
                          alpha=0.8, label='Underpredictions Outside Bounds')

        # Add the augmented value plot if the column exists
        if 'augmented_value' in df_to_plot.columns:
            plt.plot(df_to_plot.index, df_to_plot['augmented_value'], label='Augmented Value', marker='', linestyle='--')
        
        # Add plot labels and title
        plt.title(title)
        plt.xlabel('Entry Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(output_path)
        plt.close() # Close the plot to free up memory
        print(f"Plot saved successfully as '{output_path}'")
        
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        raise
