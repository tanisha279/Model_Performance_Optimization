import os

# TensorFlow Model Optimization Toolkit uses legacy Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print("Loading Iris dataset...")


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Normalize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# Create a small neural network
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])


# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel architecture:")
model.summary()


# Train
print("\nTraining original model...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    verbose=1
)


# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nOriginal Model Accuracy:", accuracy)


# Save model
model.save("models/original_model.keras")

print("Original model saved successfully!")