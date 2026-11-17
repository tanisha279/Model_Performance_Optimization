import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf


print("Loading original model...")

model = tf.keras.models.load_model(
    "models/original_model.keras"
)

print("Original model loaded successfully!")


print("\nApplying quantization...")


# Create TensorFlow Lite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


# Convert model
quantized_model = converter.convert()


# Save quantized model
with open(
    "models/quantized_model.tflite",
    "wb"
) as file:
    file.write(quantized_model)


size_kb = len(quantized_model) / 1024

print("Quantized model saved successfully!")

print(
    "Quantized model size:",
    round(size_kb, 2),
    "KB"
)