"""Train a tiny classifier and persist it as the artifact we will serve.

This script runs ONCE, offline. Serving never trains — it only loads.
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42  # pin randomness so the artifact is reproducible


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"holdout accuracy: {acc:.3f}")

    # Persist the FITTED estimator — weights, tree structure, everything.
    joblib.dump(model, "model.joblib")
    print("wrote model.joblib")


if __name__ == "__main__":
    main()
