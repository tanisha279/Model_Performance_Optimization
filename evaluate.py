import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


# Normalize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------
# Original model
# --------------------------------

original = tf.keras.models.load_model(
    "models/original_model.keras"
)

_, original_accuracy = original.evaluate(
    X_test,
    y_test,
    verbose=0
)


# --------------------------------
# Pruned model
# --------------------------------
pruned = tf.keras.models.load_model(
    "models/pruned_model.keras"
)

# Compile the pruned model before evaluation
pruned.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

_, pruned_accuracy = pruned.evaluate(
    X_test,
    y_test,
    verbose=0
)

# --------------------------------
# Model sizes
# --------------------------------

original_size = os.path.getsize(
    "models/original_model.keras"
) / 1024

pruned_size = os.path.getsize(
    "models/pruned_model.keras"
) / 1024

quantized_size = os.path.getsize(
    "models/quantized_model.tflite"
) / 1024


# --------------------------------
# Size reduction
# --------------------------------

pruned_reduction = (
    (original_size - pruned_size)
    / original_size
) * 100

quantized_reduction = (
    (original_size - quantized_size)
    / original_size
) * 100


# --------------------------------
# Display results
# --------------------------------

print("\n========================================")
print("     MODEL PERFORMANCE COMPARISON")
print("========================================")


print("\nOriginal Model")
print("-----------------------------")
print("Accuracy :", round(original_accuracy, 4))
print("Size     :", round(original_size, 2), "KB")


print("\nPruned Model")
print("-----------------------------")
print("Accuracy :", round(pruned_accuracy, 4))
print("Size     :", round(pruned_size, 2), "KB")


print("\nQuantized Model")
print("-----------------------------")
print("Size     :", round(quantized_size, 2), "KB")


print("\n========================================")
print("       EFFICIENCY IMPROVEMENT")
print("========================================")


print(
    "\nPruned model size reduction:",
    round(pruned_reduction, 2),
    "%"
)


print(
    "Quantized model size reduction:",
    round(quantized_reduction, 2),
    "%"
)