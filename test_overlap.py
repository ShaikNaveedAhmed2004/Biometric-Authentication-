import joblib
import numpy as np
import pandas as pd
import re

# 1. Load Data
try:
    fp_data = joblib.load('enhanced_fingerprint_features.pkl')
    fp_features_dict = fp_data['features']
    
    ir_f = np.load('iris_features.npy')
    ir_l = np.load('iris_labels.npy')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Extract Person IDs
# FP IDs look like 'a_100_1' or 'b_5_2'
person_fp = {}
for fid, df in fp_features_dict.items():
    # regex to extract person number, e.g., 'a_100_1' -> '100'
    match = re.search(r'_(\d+)_', fid)
    if match:
        person_str = match.group(1).zfill(3) # '100', '005'
        if person_str not in person_fp:
            person_fp[person_str] = []
        
        # Aggregate variable length triangle dataframe to fixed length using mean + std!
        mean_vec = df.mean().values
        std_vec = df.std().values
        # fillna if any std is nan
        std_vec = np.nan_to_num(std_vec)
        agg_features = np.concatenate([mean_vec, std_vec])
        person_fp[person_str].append(agg_features)

person_ir = {}
for feats, label in zip(ir_f, ir_l):
    person_str = str(label).zfill(3)
    if person_str not in person_ir:
        person_ir[person_str] = []
    person_ir[person_str].append(feats)

print(f"FP Persons: {len(person_fp)}")
print(f"Iris Persons: {len(person_ir)}")

# Overlap
common_persons = sorted(list(set(person_fp.keys()).intersection(set(person_ir.keys()))))
print(f"Common overlap: {len(common_persons)} persons")

if len(common_persons) > 0:
    sample_person = common_persons[0]
    print(f"Person {sample_person} -> FP samples: {len(person_fp[sample_person])}, Iris samples: {len(person_ir[sample_person])}")
    print(f"FP feature dim: {person_fp[sample_person][0].shape}")
    print(f"Iris feature dim: {person_ir[sample_person][0].shape}")
