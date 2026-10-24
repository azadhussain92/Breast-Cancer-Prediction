import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 500

radius_mean = np.random.normal(15, 5, n_samples)
texture_mean = np.random.normal(20, 6, n_samples)
perimeter_mean = np.random.normal(100, 25, n_samples)
area_mean = np.random.normal(700, 200, n_samples)
smoothness_mean = np.random.normal(0.1, 0.05, n_samples)

labels = np.random.choice([0, 1], size=n_samples)

data = pd.DataFrame({
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'perimeter_mean': perimeter_mean,
    'area_mean': area_mean,
    'smoothness_mean': smoothness_mean,
    'target': labels
})

data.to_csv('breast_cancer_data.csv', index=False)
print("CSV file 'breast_cancer_data.csv' has been generated.")
