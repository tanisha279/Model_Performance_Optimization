import matplotlib.pyplot as plt


# ========================================
# Values automatically obtained from
# evaluate.py output
# ========================================

original_accuracy = 0.7333
pruned_accuracy = 0.7667

original_size = 26.20
pruned_size = 12.99
quantized_size = 3.01

pruned_reduction = 50.43
quantized_reduction = 88.51


# ========================================
# GRAPH 1: ACCURACY COMPARISON
# ========================================

models = ["Original", "Pruned"]
accuracies = [
    original_accuracy * 100,
    pruned_accuracy * 100
]

plt.figure(figsize=(7, 5))

bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
    dpi=300
)

plt.show()


# ========================================
# GRAPH 2: MODEL SIZE COMPARISON
# ========================================

models = [
    "Original",
    "Pruned",
    "Quantized"
]

sizes = [
    original_size,
    pruned_size,
    quantized_size
]

plt.figure(figsize=(7, 5))

bars = plt.bar(models, sizes)

plt.title("Model Size Comparison")
plt.xlabel("Model")
plt.ylabel("Model Size (KB)")

for bar, value in zip(bars, sizes):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f"{value:.2f} KB",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "model_size_comparison.png",
    dpi=300
)

plt.show()


# ========================================
# DISPLAY FINAL RESULTS
# ========================================

print("\n========================================")
print("          GRAPH GENERATION COMPLETE")
print("========================================")

print("\nAccuracy:")
print("Original :", round(original_accuracy * 100, 2), "%")
print("Pruned   :", round(pruned_accuracy * 100, 2), "%")

print("\nModel Size:")
print("Original  :", original_size, "KB")
print("Pruned    :", pruned_size, "KB")
print("Quantized :", quantized_size, "KB")

print("\nSize Reduction:")
print("Pruned    :", pruned_reduction, "%")
print("Quantized :", quantized_reduction, "%")

print("\nCreated:")
print("accuracy_comparison.png")
print("model_size_comparison.png")