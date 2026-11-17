import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print("Loading original model...")

model = tf.keras.models.load_model(
    "models/original_model.keras"
)

print("Original model loaded successfully!")


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Same train/test split
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


# Pruning configuration
pruning_params = {
    "pruning_schedule":
        tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=100
        )
}


print("\nApplying pruning...")

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    **pruning_params
)


# Compile
pruned_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train
print("\nTraining pruned model...")

pruned_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    verbose=1,
    callbacks=[
        tfmot.sparsity.keras.UpdatePruningStep()
    ]
)


# Evaluate
loss, accuracy = pruned_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nPruned Model Accuracy:", accuracy)


# Remove pruning wrappers
pruned_model = tfmot.sparsity.keras.strip_pruning(
    pruned_model
)


# Save
pruned_model.save(
    "models/pruned_model.keras"
)

print("Pruned model saved successfully!")