import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

df = pd.read_csv("data/with_model_sentiments.csv")
y_true = df["Sentiment"]

metrics = {}
models = ["vader_sentiment", "bert_sentiment"]

for model in models:
    y_pred = df[model]
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics[model] = report

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
    plt.title(f"Confusion Matrix - {model}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/confusion_matrix_{model}.png")
    plt.clf()

# Save metrics
with open("results/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Model comparison completed.")