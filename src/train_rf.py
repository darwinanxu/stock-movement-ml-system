from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_classifier


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_prob)
    return model, metrics