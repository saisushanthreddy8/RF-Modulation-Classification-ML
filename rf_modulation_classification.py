# # RF Modulation Classification - RadioML 2016.10a
# **Random Forest vs SVM vs XGBoost vs KNN**
#
# Steps: load dataset → extract 176 features → 80/20 split → train and test 4 models → print metrics → compare.
#
# **How to run:** put `RML2016.10a_dict.pkl` in the `dataset/` folder (download link in the README), select a Python kernel and click **Run All**. All results (metrics, tables and plots) appear directly below the cells.


# ## 0. Dataset
# The notebook looks for `dataset/RML2016.10a_dict.pkl`. If the file is not found there, you are asked to type its path.


import os

DATASET_FILE = os.path.join("dataset", "RML2016.10a_dict.pkl")
RUN_SVM = True     # SVM takes about 1 hour to train - set to False to skip it

while not os.path.exists(DATASET_FILE):
    print("Dataset not found:", DATASET_FILE)
    DATASET_FILE = input("Enter the full path of RML2016.10a_dict.pkl: ").strip().strip('"')
print("Dataset ready:", DATASET_FILE)


# ## Imports


import time
import pickle
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ## Feature extraction
# From each signal (128 I/Q samples) we calculate 176 features.


# Each RF signal has 128 I/Q samples. From each signal we
# calculate 176 numbers (features) that describe it.

def basic_features(I, Q):
    # The original 27 features: amplitude, phase, frequency,
    # moments, I/Q statistics and spectrum shape
    z = I + 1j * Q
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    freq = np.diff(phase)

    m20 = np.mean(z ** 2)
    m40 = np.mean(z ** 4)
    c40 = m40 - 3 * (m20 ** 2)

    freqs, psd = scipy.signal.welch(amp, nperseg=64)
    centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-12))

    skewness = np.mean((amp - np.mean(amp)) ** 3) / (np.std(amp) ** 3 + 1e-12)
    kurtosis = np.mean((amp - np.mean(amp)) ** 4) / (np.std(amp) ** 4 + 1e-12)

    return [
        np.mean(amp), np.std(amp), np.var(amp), np.sqrt(np.mean(amp ** 2)), skewness, kurtosis,
        np.mean(phase), np.std(phase), np.var(phase),
        np.mean(freq), np.std(freq), np.var(freq),
        np.real(m20), np.imag(m20), np.real(m40), np.imag(m40), np.real(c40), np.imag(c40),
        np.real(np.mean(z * np.conj(z))), np.real(np.mean((z ** 2) * (np.conj(z) ** 2))),
        np.mean(I), np.std(I), np.mean(Q), np.std(Q),
        centroid, spread, np.max(psd),
    ]


def cumulant_features(z):
    # 4th-order cumulants: different modulations have different
    # textbook values (e.g. C42 = -2 for BPSK, -1 for QPSK, -0.68 for QAM16)
    x = z - np.mean(z)
    m20 = np.mean(x ** 2)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x ** 4)
    m41 = np.mean(x ** 3 * np.conj(x))
    m42 = np.mean(np.abs(x) ** 4)

    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2

    # divide by power^2 so the values do not depend on signal strength
    return [np.abs(c40) / m21 ** 2, np.abs(c41) / m21 ** 2, c42 / m21 ** 2]


def power_peak_features(z):
    # Raising the signal to a power removes PSK modulation and leaves a
    # strong spectral peak: BPSK at z^2, QPSK at z^4, 8PSK at z^8
    result = []
    for k in [1, 2, 4, 8]:
        spectrum = np.abs(np.fft.fft(z ** k)) ** 2
        result.append(np.max(spectrum) / (np.sum(spectrum) + 1e-12))
    return result


def phase_features(z):
    # Phase moments: tell how many phase levels the signal uses
    amp = np.abs(z)
    unit = z / (amp + 1e-12)          # keep only the phase
    result = []
    for k in [1, 2, 4, 8]:
        result.append(np.abs(np.mean(unit ** k)))
    return result


def amplitude_features(z):
    # Amplitude distribution: 10%, 25%, 50%, 75% and 90% points
    amp = np.abs(z)
    amp = amp / (np.mean(amp) + 1e-12)
    return list(np.quantile(amp, [0.10, 0.25, 0.50, 0.75, 0.90]))


def autocorrelation_features(z):
    # How similar the signal is to a shifted copy of itself
    result = []
    for lag in [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]:
        result.append(np.abs(np.mean(z[lag:] * np.conj(z[:-lag]))))
    # same for z^2 (useful for phase modulations)
    z2 = z ** 2
    for lag in [1, 2, 4, 8]:
        result.append(np.abs(np.mean(z2[lag:] * np.conj(z2[:-lag]))))
    return result


def snr_feature(z):
    # Rough estimate of the signal-to-noise ratio (M2M4 estimator)
    amp = np.abs(z)
    m2 = np.mean(amp ** 2)
    m4 = np.mean(amp ** 4)
    s = np.sqrt(max(2 * m2 ** 2 - m4, 0))
    noise = max(m2 - s, 1e-6)
    ratio = min(max(s / noise, 1e-3), 1e3)
    return [10 * np.log10(ratio)]


def spectrum_bins(signal, n_bins):
    # Power spectrum split into n_bins equal parts (each part = share of total power)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(signal)) ** 2)
    spectrum = spectrum / (np.sum(spectrum) + 1e-12)
    size = len(spectrum) // n_bins
    result = []
    for i in range(n_bins):
        result.append(np.sum(spectrum[i * size:(i + 1) * size]))
    return result


def spectrum_features(z):
    result = []
    result += spectrum_bins(z, 64)          # spectrum of the signal (bandwidth, shape)
    result += spectrum_bins(z ** 2, 16)     # spectrum of z^2
    result += spectrum_bins(z ** 4, 16)     # spectrum of z^4
    # envelope spectrum: how the amplitude changes (AM signals)
    amp = np.abs(z)
    envelope = np.abs(np.fft.fft(amp - np.mean(amp)))[:64] ** 2
    envelope = envelope / (np.sum(envelope) + 1e-12)
    for i in range(16):
        result.append(np.sum(envelope[i * 4:(i + 1) * 4]))
    return result


def frequency_features(z):
    # Instantaneous frequency distribution (useful for FSK / FM)
    freq = np.diff(np.unwrap(np.angle(z)))
    return list(np.quantile(freq, [0.05, 0.25, 0.50, 0.75, 0.95]))


def extract_features(signal):
    # signal has shape (2, 128): row 0 = I, row 1 = Q
    I = signal[0]
    Q = signal[1]

    # complex signal scaled to power 1 (so features do not depend on signal strength)
    z = I.astype(np.float64) + 1j * Q.astype(np.float64)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    features = []
    features += basic_features(I, Q)          # 27
    features += cumulant_features(z)          # 3
    features += power_peak_features(z)        # 4
    features += phase_features(z)             # 4
    features += amplitude_features(z)         # 5
    features += autocorrelation_features(z)   # 15
    features += snr_feature(z)                # 1
    features += spectrum_features(z)          # 112
    features += frequency_features(z)         # 5
    return np.nan_to_num(np.array(features, dtype=np.float32))


# ## Models
# Best settings found by hyperparameter tuning.


def build_random_forest():
    return RandomForestClassifier(n_estimators=200, max_depth=20,
                                  max_features="sqrt", random_state=42, n_jobs=-1)


def build_svm():
    # SVM needs all features on the same scale, so StandardScaler comes first
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=1, gamma="scale", cache_size=1000))


def build_xgboost():
    return XGBClassifier(n_estimators=364, max_depth=8, learning_rate=0.05,
                         subsample=0.8, min_child_weight=5, gamma=0.3,
                         tree_method="hist", random_state=42, n_jobs=-1)


def build_knn():
    # KNN uses distances, so all features must be on the same scale
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=301, weights="distance", n_jobs=-1))


# ## Evaluation functions
# Print metrics (accuracy, precision, recall, F1, Kappa, MCC, SNR accuracy) and draw plots.


def accuracy_per_snr(y_test, y_pred, snr_test, snrs):
    # accuracy separately for each SNR level
    result = []
    for snr in snrs:
        mask = (snr_test == snr)
        result.append(accuracy_score(y_test[mask], y_pred[mask]) * 100)
    return result


def show_results(name, y_test, y_pred, snr_test, modulations, snrs):
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

    # ---------- printed results ----------
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
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=modulations).plot(cmap="Blues", xticks_rotation=45, ax=ax)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(snrs, snr_acc, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{name} - Accuracy vs SNR")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "kappa": kappa, "mcc": mcc, "high_snr": high_snr_acc, "best_snr": max(snr_acc),
            "snr_acc": snr_acc}


def compare_models(all_results, snrs):
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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close()


# ## 1. Load dataset and extract features
# Feature extraction takes several minutes.


print("Loading dataset...")
with open(DATASET_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

modulations = sorted(set(key[0] for key in data.keys()))
snrs = sorted(set(key[1] for key in data.keys()))
print("Classes:", modulations)
print("SNR levels:", snrs[0], "to", snrs[-1], "dB")

print("Extracting features (takes several minutes)...")
X, y, snr_labels = [], [], []
for (modulation, snr), signals in data.items():
    for signal in signals:
        X.append(extract_features(signal))
        y.append(modulations.index(modulation))
        snr_labels.append(snr)
X, y, snr_labels = np.array(X), np.array(y), np.array(snr_labels)

print("Feature matrix:", X.shape)


# ## 2. Train / test split (80% / 20%)


X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, y, snr_labels, test_size=0.20, random_state=42, stratify=y)
print("Train samples:", len(X_train), "| Test samples:", len(X_test))


# ## 3. Train and test each model
# Results for each model appear below. SVM takes about 1 hour to train.


models = {
    "Random Forest": build_random_forest,
    "SVM": build_svm,
    "XGBoost": build_xgboost,
    "KNN": build_knn,
}
if not RUN_SVM:
    del models["SVM"]

all_results = {}
for name, build_model in models.items():
    print("\n" + "#" * 60)
    print("#", name)
    print("#" * 60)

    print("Training...")
    start = time.time()
    model = build_model()
    model.fit(X_train, y_train)
    print(f"Training time: {(time.time() - start) / 60:.1f} min")

    print("Testing...")
    y_pred = model.predict(X_test)
    all_results[name] = show_results(name, y_test, y_pred, snr_test, modulations, snrs)


# ## 4. Compare all models


compare_models(all_results, snrs)
