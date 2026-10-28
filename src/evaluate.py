# ============================================================
# EVALUATION: print results in the terminal and save plots
# ============================================================
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def accuracy_per_snr(y_test, y_pred, snr_test, snrs):
    # accuracy separately for each SNR level
    result = []
    for snr in snrs:
        mask = (snr_test == snr)
        result.append(accuracy_score(y_test[mask], y_pred[mask]) * 100)
    return result


def show_results(name, y_test, y_pred, snr_test, modulations, snrs, results_dir):
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average="macro") * 100
    recall = recall_score(y_test, y_pred, average="macro") * 100
    f1 = f1_score(y_test, y_pred, average="macro") * 100
    kappa = cohen_kappa_score(y_test, y_pred)      # 0 = random guessing, 1 = perfect
    mcc = matthews_corrcoef(y_test, y_pred)        # -1 to +1, 1 = perfect
    snr_acc = accuracy_per_snr(y_test, y_pred, snr_test, snrs)
    high_snr_values = []
    for i in range(len(snrs)):
        if snrs[i] >= 0:
            high_snr_values.append(snr_acc[i])
    high_snr_acc = np.mean(high_snr_values)

    # ---------- terminal output ----------
    print("\n" + "=" * 60)
    print(name.upper(), "RESULTS")
    print("=" * 60)
    print(f"Accuracy               : {accuracy:.2f} %")
    print(f"Precision (macro)      : {precision:.2f} %")
    print(f"Recall (macro)         : {recall:.2f} %")
    print(f"F1-score (macro)       : {f1:.2f} %")
    print(f"Cohen's Kappa          : {kappa:.4f}")
    print(f"MCC                    : {mcc:.4f}")
    print(f"Accuracy at SNR >= 0 dB: {high_snr_acc:.2f} %")
    print(f"Best SNR accuracy      : {max(snr_acc):.2f} %")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=modulations, digits=4))

    print("Confusion matrix (rows = true class, columns = predicted class):")
    cm = confusion_matrix(y_test, y_pred)
    header = "        "
    for m in modulations:
        header += f"{m[:6]:>7}"
    print(header)
    for i in range(len(modulations)):
        row = f"{modulations[i][:6]:<8}"
        for value in cm[i]:
            row += f"{value:>7}"
        print(row)

    print("\nAccuracy for each SNR level:")
    for i in range(len(snrs)):
        print(f"  {snrs[i]:4d} dB : {snr_acc[i]:6.2f} %")

    # ---------- plots ----------
    file_name = name.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=modulations).plot(cmap="Blues", xticks_rotation=45, ax=ax)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{file_name}_confusion_matrix.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(snrs, snr_acc, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{name} - Accuracy vs SNR")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{file_name}_accuracy_vs_snr.png"), dpi=200)
    plt.close()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "kappa": kappa, "mcc": mcc, "high_snr": high_snr_acc, "best_snr": max(snr_acc),
            "snr_acc": snr_acc}


def compare_models(all_results, snrs, results_dir):
    # ---------- comparison table ----------
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<24}", end="")
    for name in all_results:
        print(f"{name:>16}", end="")
    print()
    rows = [("Accuracy (%)", "accuracy"), ("Precision (%)", "precision"), ("Recall (%)", "recall"),
            ("F1-score (%)", "f1"), ("Cohen's Kappa", "kappa"), ("MCC", "mcc"),
            ("Accuracy SNR >= 0 dB (%)", "high_snr"), ("Best SNR accuracy (%)", "best_snr")]
    for label, key in rows:
        print(f"{label:<24}", end="")
        for name in all_results:
            value = all_results[name][key]
            if key in ("kappa", "mcc"):
                print(f"{value:>16.4f}", end="")
            else:
                print(f"{value:>16.2f}", end="")
        print()

    best = None
    for name in all_results:
        if best is None or all_results[name]["accuracy"] > all_results[best]["accuracy"]:
            best = name
    print(f"\nBest model: {best} ({all_results[best]['accuracy']:.2f} %)")

    # ---------- comparison plot ----------
    plt.figure(figsize=(10, 5))
    for name, r in all_results.items():
        plt.plot(snrs, r["snr_acc"], marker="o", label=f"{name} ({r['accuracy']:.2f}%)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs SNR - all models")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_accuracy_vs_snr.png"), dpi=200)
    plt.close()

    names = []
    accuracies = []
    for name in all_results:
        names.append(name)
        accuracies.append(all_results[name]["accuracy"])
    plt.figure(figsize=(8, 5))
    plt.bar(names, accuracies)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha="center")
    plt.ylabel("Accuracy (%)")
    plt.title("Overall accuracy - all models")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_accuracy.png"), dpi=200)
    plt.close()

    # all main metrics side by side for every model
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    bar_width = 0.8 / len(names)
    plt.figure(figsize=(10, 5))
    for i, name in enumerate(names):
        values = []
        for key in metric_keys:
            values.append(all_results[name][key])
        positions = np.arange(len(metric_names)) + i * bar_width
        plt.bar(positions, values, width=bar_width, label=name)
    plt.xticks(np.arange(len(metric_names)) + bar_width * (len(names) - 1) / 2, metric_names)
    plt.ylabel("Score (%)")
    plt.title("Accuracy, Precision, Recall and F1-score - all models")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_metrics.png"), dpi=200)
    plt.close()
