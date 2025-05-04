import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_charts(metrics_dict):
    os.makedirs("results/confusion_matrices", exist_ok=True)

    accuracies = {}
    precisions = {}
    recalls = {}
    f1s = {}

    for model_name, metrics in metrics_dict.items():
        accuracies[model_name] = metrics['accuracy']
        # use weighted avg for overall performance
        report = metrics['classification_report']['weighted avg']
        precisions[model_name] = report['precision']
        recalls[model_name] = report['recall']
        f1s[model_name] = report['f1-score']

        # plot and save confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrices/{model_name}_confusion_matrix.png')
        plt.close()

    def plot_metric(metric_dict, metric_name, annotate=False):
        plt.figure(figsize=(10, 5))
        bars = plt.bar(metric_dict.keys(), metric_dict.values(), color='skyblue')
        plt.ylim(0, 1)
        plt.title(f'{metric_name} Comparison')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)

        if annotate:
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'results/{metric_name.lower().replace(" ", "_")}_comparison.png')
        plt.close()

    plot_metric(accuracies, 'Accuracy', annotate=True)
    plot_metric(precisions, 'Precision', annotate=True)
    plot_metric(recalls, 'Recall', annotate=True)
    plot_metric(f1s, 'F1 Score', annotate=True)