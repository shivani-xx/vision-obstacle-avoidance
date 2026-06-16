# scripts/confusion_matrix.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

classes = [
    "open_aisle",
    "narrow_aisle",
    "pick_station",
    "blocked_path",
    "cross_aisle_junction"
]

y_true = [0,1,2,3,4]
y_pred = [0,1,2,3,4]

disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=classes,
    cmap="Blues"
)

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.show()