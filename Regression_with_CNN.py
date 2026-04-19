# %%
import os
import json
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Create artifacts directory
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# %%
# [Your existing data loading and preprocessing code here...]

# %% [markdown]
# # Visualize training history and save plots

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True)

# Plot R2
axes[1].plot(history.history['r2_metric'], label='Train R2')
axes[1].plot(history.history['val_r2_metric'], label='Validation R2')
axes[1].set_title('Model R2 Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R2 Score')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f'{artifacts_dir}/training_history.png', dpi=300, bbox_inches='tight')
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')  # Save in root for artifact
plt.show()

# %% [markdown]
# # Visualise the predictions and residuals

# %%
# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True, alpha=0.3)
plt.savefig(f'{artifacts_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot residuals
residuals = y_test - preds

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of residuals
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residuals Distribution')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].grid(True, alpha=0.3)

# Residuals vs predictions
axes[1].scatter(preds, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predictions')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Predictions')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{artifacts_dir}/residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Save metrics to file

# %%
# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mse)

# Create metrics dictionary
metrics = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "1D CNN",
    "test_size": 0.2,
    "batch_size": batch_size,
    "epochs": epochs,
    "final_epoch": len(history.history['loss']),
    "metrics": {
        "mean_squared_error": float(mse),
        "root_mean_squared_error": float(rmse),
        "mean_absolute_error": float(mae),
        "r2_score": float(r2)
    },
    "training_history": {
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "final_train_r2": float(history.history['r2_metric'][-1]),
        "final_val_r2": float(history.history['val_r2_metric'][-1]),
        "best_val_loss": float(min(history.history['val_loss'])),
        "best_val_r2": float(max(history.history['val_r2_metric']))
    }
}

# Save metrics as JSON
with open(f'{artifacts_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save metrics as text file
with open('metrics.txt', 'w') as f:
    f.write("="*50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("="*50 + "\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write("="*50 + "\n\n")
    f.write("TRAINING HISTORY\n")
    f.write("="*50 + "\n")
    f.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"Final Train R2: {history.history['r2_metric'][-1]:.4f}\n")
    f.write(f"Final Validation R2: {history.history['val_r2_metric'][-1]:.4f}\n")
    f.write(f"Best Validation Loss: {min(history.history['val_loss']):.4f}\n")
    f.write(f"Best Validation R2: {max(history.history['val_r2_metric']):.4f}\n")
    f.write("="*50 + "\n")

# Also save metrics in the artifacts directory
with open(f'{artifacts_dir}/metrics.txt', 'w') as f:
    f.write(open('metrics.txt').read())

print("Metrics saved to metrics.txt and artifacts/metrics.json")

# %% [markdown]
# # Save model summary

# %%
# Save model architecture summary
with open('model_summary.txt', 'w') as f:
    # Redirect print output to file
    original_stdout = sys.stdout
    sys.stdout = f
    model.summary()
    sys.stdout = original_stdout

# Copy to artifacts directory
import shutil
shutil.copy('model_summary.txt', f'{artifacts_dir}/model_summary.txt')

print("Model summary saved to model_summary.txt")

# %% [markdown]
# # Save model

# %%
# Save the model in multiple formats
model.save(f'{artifacts_dir}/cnn_regression_model.h5')
model.save('cnn_regression_model.h5')  # Save in root for artifact

# Save as SavedModel format (TensorFlow standard)
model.save(f'{artifacts_dir}/saved_model', save_format='tf')

# Save model weights separately
model.save_weights(f'{artifacts_dir}/model_weights.h5')

print(f"Model saved to {artifacts_dir}/cnn_regression_model.h5")

# %% [markdown]
# # Save data information

# %%
# Save data information
data_info = {
    "train_samples": len(data),
    "test_samples": len(dtest),
    "features_count": X.shape[1],
    "categorical_variables_original": len(cat_vars) if 'cat_vars' in locals() else 0,
    "suspicious_features_dropped": len(suspiciousData) if 'suspiciousData' in locals() else 0,
    "numerical_features_after_processing": X.shape[1],
    "target_mean": float(y.mean()),
    "target_std": float(y.std()),
    "target_min": float(y.min()),
    "target_max": float(y.max())
}

with open(f'{artifacts_dir}/data_info.json', 'w') as f:
    json.dump(data_info, f, indent=4)

print("Data information saved to artifacts/data_info.json")

# %% [markdown]
# # Save submission file

# %%
# Create submission file
submission = pd.DataFrame({
    "ID": dtest["ID"],
    "y": predictions
})

# Save in multiple locations
submission.to_csv('submission_5.csv', index=False)
submission.to_csv(f'{artifacts_dir}/submission_5.csv', index=False)

print(f"\nSubmission saved to submission_5.csv and {artifacts_dir}/submission_5.csv")

# %%
# Create a summary report
summary_report = f"""
================================================================================
CNN REGRESSION MODEL - TRAINING SUMMARY
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA INFORMATION:
- Training samples: {data_info['train_samples']}
- Test samples: {data_info['test_samples']}
- Features used: {data_info['features_count']}
- Target range: [{data_info['target_min']:.4f}, {data_info['target_max']:.4f}]

MODEL PERFORMANCE:
- MSE: {mse:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
- R2 Score: {r2:.4f}

MODEL ARCHITECTURE:
- Type: 1D Convolutional Neural Network
- Conv1D layers: 3
- Dense layers: 3
- Total parameters: {model.count_params():,}

FILES GENERATED:
- model_results.png (training history plot)
- predictions_vs_actual.png
- residuals_analysis.png
- metrics.txt (performance metrics)
- metrics.json (detailed metrics in JSON format)
- model_summary.txt (model architecture)
- cnn_regression_model.h5 (saved model)
- submission_5.csv (predictions for test set)
- data_info.json (dataset information)

================================================================================
"""

with open('summary_report.txt', 'w') as f:
    f.write(summary_report)

with open(f'{artifacts_dir}/summary_report.txt', 'w') as f:
    f.write(summary_report)

print(summary_report)

# %%
# List all generated artifacts
print("\n" + "="*50)
print("GENERATED ARTIFACTS:")
print("="*50)
for file in os.listdir('.'):
    if file.endswith(('.png', '.txt', '.csv', '.h5', '.json')):
        size = os.path.getsize(file) / 1024  # Size in KB
        print(f"  - {file} ({size:.2f} KB)")

print(f"\nAll artifacts also saved in '{artifacts_dir}/' directory")
