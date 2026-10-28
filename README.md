# RF Modulation Classification (RadioML 2016.10a)

Classifies radio signals into **11 modulation types** with four classical machine-learning models: **Random Forest, SVM, XGBoost and KNN**.

Course project, 23AID205 Introduction to AI & ML, Group A-8, Amrita Vishwa Vidyapeetham, Coimbatore.

## Results (test set: 44,000 signals, all 20 SNR levels)

| Metric | Random Forest | SVM | **XGBoost** | KNN |
| --- | --- | --- | --- | --- |
| Accuracy (%) | 58.41 | 57.98 | **59.68** | 53.53 |
| Precision, macro (%) | 70.93 | 71.21 | 66.49 | 63.59 |
| Recall, macro (%) | 58.41 | 57.98 | **59.68** | 53.53 |
| F1-score, macro (%) | 60.93 | 60.34 | **61.00** | 55.11 |
| Cohen's Kappa | 0.543 | 0.538 | **0.557** | 0.489 |
| MCC | 0.564 | 0.561 | **0.567** | 0.506 |
| Accuracy at SNR ≥ 0 dB (%) | 87.55 | 87.42 | 87.33 | 83.15 |

## Dataset

RadioML 2016.10a (DeepSig): 220,000 I/Q signals of shape 2 × 128, 11 classes, SNR from −20 to +18 dB.

The file is too large for the repository (641 MB), so it is attached to this repository's **Releases** page:

1. Download `RML2016.10a_dict.pkl` from the latest release.
2. Put it in the `dataset/` folder.

Original source: [deepsig.ai/datasets](https://www.deepsig.ai/datasets) (O'Shea & West, GNU Radio Conference 2016). The dataset is licensed under CC BY-NC-SA 4.0.

## How to run

```
pip install -r requirements.txt
```

Open `rf_modulation_classification.ipynb` in VS Code or Jupyter, select the Python kernel and click **Run All**. All metrics, the classification report, confusion matrices and accuracy-vs-SNR plots appear below the cells. The same code is in `rf_modulation_classification.py`.

Feature extraction takes a few minutes; SVM training takes about one hour (set `RUN_SVM = False` in the first code cell to skip it).

## Method

1. Load the dataset and extract **176 features** per signal (statistics, 4th-order cumulants, power-of-signal peaks, phase moments, amplitude quantiles, autocorrelation, SNR estimate, spectrum bins, instantaneous-frequency quantiles).
2. Stratified 80/20 train/test split (random_state = 42).
3. Train Random Forest, SVM (RBF), XGBoost and KNN with tuned hyperparameters.
4. Evaluate with accuracy, macro precision/recall/F1, Cohen's Kappa, MCC, confusion matrix and accuracy per SNR.

## Team

S. Sujitkumar · M. Hemanthkumar · N. Sai Sushanth Reddy · H. Abinavkumaran

Instructor: Sruthi S
