import json
import matplotlib.pyplot as plt

# Load files
with open('results\\capstone_results.json') as f:
    train_res = json.load(f)

with open('results\\capstone_latency.json') as f:
    lat_res = json.load(f)

# Merge by model + resolution
merged = []

for train in train_res:
    for lat in lat_res:
        if (
            train['model_name'] == lat['model']
            and train['resolution'] == lat['resolution']
        ):
            merged.append({
                'label': f"{train['model_name']}-{train['resolution']}",
                'accuracy': train['best_val_acc'] * 100,
                'latency': lat['mean_ms']
            })

# Plot
plt.figure(figsize=(10,6))

for item in merged:
    plt.scatter(item['latency'], item['accuracy'], s=120)

    plt.annotate(
        item['label'],
        (item['latency'], item['accuracy']),
        xytext=(8,8),
        textcoords='offset points',
        fontsize=9
    )

# Real-time threshold
plt.axvline(x=50, linestyle='--', label='50 ms threshold')

# Accuracy target
plt.axhline(y=85, linestyle='--', label='85% target')

plt.xlabel('Inference Latency (ms)')
plt.ylabel('Validation Accuracy (%)')
plt.title('Accuracy vs Latency Tradeoff')

plt.grid(True)
plt.legend()

plt.savefig('results\\pareto_chart.png')
plt.show()

print("Pareto chart saved!")