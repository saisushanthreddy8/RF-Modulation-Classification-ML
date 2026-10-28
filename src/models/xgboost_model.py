from xgboost import XGBClassifier


def build_model():
    # best settings found by hyperparameter tuning
    return XGBClassifier(n_estimators=364, max_depth=8, learning_rate=0.05,
                         subsample=0.8, min_child_weight=5, gamma=0.3,
                         tree_method="hist", random_state=42, n_jobs=-1)
