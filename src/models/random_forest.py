from sklearn.ensemble import RandomForestClassifier


def build_model():
    # best settings found by hyperparameter tuning
    return RandomForestClassifier(n_estimators=200, max_depth=20,
                                  max_features="sqrt", random_state=42, n_jobs=-1)
