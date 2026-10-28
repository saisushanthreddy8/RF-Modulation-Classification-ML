from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def build_model():
    # KNN uses distances, so all features must be on the same scale (StandardScaler first)
    # best settings found by testing different k values
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=301, weights="distance", n_jobs=-1))
