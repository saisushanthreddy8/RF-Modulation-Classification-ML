# ============================================================
# RF MODULATION CLASSIFICATION - RadioML 2016.10a
# Random Forest vs SVM vs XGBoost vs KNN
#
# Run:   python main.py           (uses saved models if they exist)
#        python main.py --train   (trains all models again)
# ============================================================
import os
import sys
import time
import pickle
import gzip
import shutil
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.features import extract_features
from src.evaluate import show_results, compare_models
from src.models import random_forest, svm, xgboost_model, knn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset", "RML2016.10a_dict.pkl")
FEATURES_FILE = os.path.join(BASE_DIR, "dataset", "features.npz")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. LOAD DATASET AND EXTRACT FEATURES
# ------------------------------------------------------------
# the dataset is stored on GitHub as 3 compressed parts (GitHub limit is 100 MB per file)
# join and unzip them into RML2016.10a_dict.pkl (first run only)
if not os.path.exists(DATASET_FILE):
    print("Rebuilding dataset from its 3 parts (first run only)...")
    gz_file = DATASET_FILE + ".gz"
    with open(gz_file, "wb") as out:
        for i in range(3):
            with open(DATASET_FILE + ".gz.part" + str(i), "rb") as f:
                shutil.copyfileobj(f, out)
    with gzip.open(gz_file, "rb") as f_in, open(DATASET_FILE, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_file)

print("Loading dataset...")
with open(DATASET_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

modulations = sorted(set(key[0] for key in data.keys()))
snrs = sorted(set(key[1] for key in data.keys()))
print("Classes:", modulations)
print("SNR levels:", snrs[0], "to", snrs[-1], "dB")

if os.path.exists(FEATURES_FILE):
    # features were already calculated in an earlier run
    print("Loading saved features...")
    saved = np.load(FEATURES_FILE)
    X, y, snr_labels = saved["X"], saved["y"], saved["snr"]
else:
    print("Extracting features (first run only, takes several minutes)...")
    X, y, snr_labels = [], [], []
    for (modulation, snr), signals in data.items():
        for signal in signals:
            X.append(extract_features(signal))
            y.append(modulations.index(modulation))
            snr_labels.append(snr)
    X, y, snr_labels = np.array(X), np.array(y), np.array(snr_labels)
    np.savez_compressed(FEATURES_FILE, X=X, y=y, snr=snr_labels)

print("Feature matrix:", X.shape)


# ------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (80% train, 20% test)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, y, snr_labels, test_size=0.20, random_state=42, stratify=y)
print("Train samples:", len(X_train), "| Test samples:", len(X_test))


# ------------------------------------------------------------
# 3. TRAIN AND TEST EACH MODEL
# ------------------------------------------------------------
models = {
    "Random Forest": random_forest,
    "SVM": svm,
    "XGBoost": xgboost_model,
    "KNN": knn,
}
retrain = "--train" in sys.argv
all_results = {}

for name, model_file in models.items():
    print("\n" + "#" * 60)
    print("#", name)
    print("#" * 60)
    model_path = os.path.join(MODELS_DIR, name.lower().replace(" ", "_") + ".joblib")

    if os.path.exists(model_path) and not retrain:
        print("Loading saved model:", model_path)
        model = joblib.load(model_path)
    else:
        print("Training... (SVM can take up to an hour)")
        start = time.time()
        model = model_file.build_model()
        model.fit(X_train, y_train)
        print(f"Training time: {(time.time() - start) / 60:.1f} min")
        joblib.dump(model, model_path, compress=3)

    print("Testing...")
    y_pred = model.predict(X_test)
    all_results[name] = show_results(name, y_test, y_pred, snr_test, modulations, snrs, RESULTS_DIR)


# ------------------------------------------------------------
# 4. COMPARE ALL MODELS
# ------------------------------------------------------------
compare_models(all_results, snrs, RESULTS_DIR)
print("\nPlots saved in:", RESULTS_DIR)
