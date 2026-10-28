from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model():
    # SVM needs all features on the same scale, so StandardScaler comes first
    # best settings found by hyperparameter tuning
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=1, gamma="scale", cache_size=1000))
