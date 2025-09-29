import flask
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
import pandas as pd
import os
import tempfile
import uuid
import sys
import traceback
import matplotlib
matplotlib.use('Agg')

# Assuming runner.py and augmentations.py are in the same directory
from runner import forecast_with_metrics, ModelChooser, plot
from augmentations import apply_augmentation

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['STATIC_FOLDER'] = os.path.join(app.root_path, 'static')

# Define the models and augmentations to pass to the template
models = {
    "MOIRAI 2.0": "moirai",
    "CHRONOS": "chronos",
    "MOIRAI-MOE": "moirai-moe"
}
augmentations = {
    "None": "None",
    "Convex Hull Method": "convex_hull_method",
    "Rolling Means (5)": "rolling_means",
    "Peak Multiplier": "peak_multiplier"
}
metrics_explanations = {
    "MAE": "Mean Absolute Error: The average absolute difference between the predicted and actual values.",
    "RMSE": "Root Mean Squared Error: The square root of the average of the squared differences between the predicted and actual values.",
    "R2": "R-squared: The proportion of the variance in the dependent variable that is predictable from the independent variable(s).",
    "SMAPE": "Symmetric Mean Absolute Percentage Error: A percentage error based on the absolute differences between the predicted and actual values.",
    "MAPE": "Mean Absolute Percentage Error: The average of the absolute percentage errors.",
    "Underpredictions": "The number of times the predicted value is less than the actual value."
}
augmentations_explanations = {
    "convex_hull_method": "Creates a new series by iteratively finding the maximum slope between the current point and future points within a relaxation period, and then extending the current point with that slope.",
    "rolling_means": "Calculates the rolling mean of the series.",
    "peak_multiplier": "Identifies peaks in the series and multiplies them by a given factor. It then smooths the modified peaks by taking the maximum of the current point and a linearly decreasing fraction of the neighboring modified peaks."
}

# Create the static folder if it doesn't exist
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

def cleanup_old_plots():
    """Removes all .png files from the static directory."""
    static_folder = app.config['STATIC_FOLDER']
    for filename in os.listdir(static_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(static_folder, filename)
            os.remove(file_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Cleanup old plots before processing the new request
        cleanup_old_plots()

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        original_csv_path = None
        temp_file_to_clean = None
        
        try:
            # Save uploaded file to a temporary file
            original_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) # type: ignore
            file.save(original_csv_path)

            # Get user inputs from form
            script_choice = request.form['model']
            augmentation_choice = request.form['augmentation']
            ctx_length = int(request.form.get('ctx_length', 168))
            pred_length = int(request.form.get('pred_length', 24))
            batch_size = int(request.form.get('batch_size', 32))
            test_length = int(request.form.get('test_length', 24*7))
            quantile = float(request.form.get('quantile', 0.9))

            # Choose the model name
            model_name = ModelChooser(script_choice)

            results = {}

            try:
                # --- FIRST RUN: Original data forecasting ---
                # This run is always performed
                plot_filename = f"forecast_plot_original_{uuid.uuid4().hex}.png"
                plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
                
                # Pass the original CSV path as both csv_path and original_csv_path
                df_original, metrics_original = forecast_with_metrics(
                    csv_path=original_csv_path,
                    original_csv_path=original_csv_path,
                    model=model_name,
                    target_column="value",
                    ctx_length=ctx_length,
                    pred_length=pred_length,
                    batch_size=batch_size,
                    test_length=test_length,
                    quantile=quantile
                )
                
                plot(df_original, plot_path, title=f"Original Forecast for {script_choice}")
                results['original'] = {
                    'metrics': metrics_original,
                    'plot_url': url_for('static', filename=plot_filename)
                }
                
                # --- SECOND RUN: Augmented data forecasting (if chosen) ---
                if augmentation_choice != "None":
                    temp_file_to_clean = os.path.join(app.config['UPLOAD_FOLDER'], f"augmented_{uuid.uuid4().hex}.csv")
                    apply_augmentation(original_csv_path, temp_file_to_clean, augmentation_choice)
                    
                    plot_filename_aug = f"forecast_plot_augmented_{uuid.uuid4().hex}.png"
                    plot_path_aug = os.path.join(app.config['STATIC_FOLDER'], plot_filename_aug)

                    df_augmented, metrics_augmented = forecast_with_metrics(
                        csv_path=temp_file_to_clean,
                        original_csv_path=original_csv_path,
                        model=model_name,
                        target_column="augmented_value",
                        ctx_length=ctx_length,
                        pred_length=pred_length,
                        batch_size=batch_size,
                        test_length=test_length,
                        quantile=quantile
                    )

                    plot(df_augmented, plot_path_aug, title=f"Augmented Forecast for {script_choice}")
                    results['augmented'] = {
                        'metrics': metrics_augmented,
                        'plot_url': url_for('static', filename=plot_filename_aug)
                    }

            except Exception as e:
                # Catch any errors from the forecasting models
                error_message = f"An error occurred: {e}"
                flash(error_message)
                return render_template("index.html", error=error_message, models=models, augmentations=augmentations, metrics_explanations=metrics_explanations, augmentations_explanations=augmentations_explanations)
            finally:
                # Clean up temporary files
                if original_csv_path and os.path.exists(original_csv_path):
                    os.remove(original_csv_path)
                if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                    os.remove(temp_file_to_clean)

            return render_template("index.html", results=results, models=models, augmentations=augmentations, metrics_explanations=metrics_explanations, augmentations_explanations=augmentations_explanations)

        except Exception as e:
            flash(f"An unexpected error occurred: {e}")
            return render_template("index.html", error=str(e), models=models, augmentations=augmentations, metrics_explanations=metrics_explanations, augmentations_explanations=augmentations_explanations)

    return render_template("index.html", models=models, augmentations=augmentations, metrics_explanations=metrics_explanations, augmentations_explanations=augmentations_explanations)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
