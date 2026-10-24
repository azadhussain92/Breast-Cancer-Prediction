import tensorflow as tf
import numpy as np

# 1. Create clear training data
X_benign = np.random.uniform(low=[5, 10, 15, 50, 0.05], high=[12, 18, 40, 200, 0.1], size=(500, 5))
y_benign = np.zeros(500)

X_malignant = np.random.uniform(low=[18, 22, 40, 500, 0.12], high=[50, 40, 100, 1500, 0.5], size=(500, 5))
y_malignant = np.ones(500)

X_train = np.vstack([X_benign, X_malignant])
y_train = np.concatenate([y_benign, y_malignant])

# 2. Build the model
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32)

# FIX: Save exactly as .keras 
model.save("breast_cancer_model.keras")
print("Model successfully saved as breast_cancer_model.keras")