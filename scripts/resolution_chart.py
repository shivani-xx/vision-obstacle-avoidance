import json
import matplotlib.pyplot as plt

with open('results\\capstone_results.json') as f:
    results = json.load(f)

# Group by model
models = {}

for r in results:
    model = r['model_name']
    res = r['resolution']
    acc = r['best_val_acc'] * 100

    if model not in models:
        models[model] = {}

    models[model][res] = acc

plt.figure(figsize=(8,5))

for model, vals in models.items():
    resolutions = sorted(vals.keys())
    accuracies = [vals[r] for r in resolutions]

    plt.plot(
        resolutions,
        accuracies,
        marker='o',
        linewidth=2,
        label=model
    )

plt.xlabel('Input Resolution')
plt.ylabel('Validation Accuracy (%)')
plt.title('Resolution Impact on Accuracy')
plt.xticks([128, 224])
plt.grid(True)
plt.legend()

plt.savefig('results\\resolution_chart.png')
plt.show()

print("Resolution chart saved!")