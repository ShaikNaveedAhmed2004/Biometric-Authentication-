import joblib
import pandas as pd
import numpy as np

fp_data = joblib.load('enhanced_fingerprint_features.pkl')
features_dict = fp_data['features']
labels = list(features_dict.keys())

print(f"Total FP images loaded: {len(labels)}")
sample_id = labels[0]
sample_df = features_dict[sample_id]
print(f"Sample FP ID: {sample_id}")
print(f"Sample FP feature type: {type(sample_df)}")
print(f"Sample FP DataFrame shape: {sample_df.shape}")
print(f"Sample FP Columns: {sample_df.columns.tolist()}")

# Check Iris
ir_f = np.load('iris_features.npy')
ir_l = np.load('iris_labels.npy')
sample_iris_label = ir_l[0]
print(f"\nSample Iris Label: {sample_iris_label}")
print(f"Sample Iris Feature Shape: {ir_f[0].shape}")
