import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

def run_random_forest(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column.")

    # Feature set = everything except returns + target
    feature_cols = [c for c in df.columns if c not in ["target"]]
    X = df[feature_cols]
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    print(f"\nRandom Forest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    # Feature importance
    print("Feature Importances:")
    for name, val in sorted(
        zip(feature_cols, model.feature_importances_), 
        key=lambda x: x[1], reverse=True
    ):
        print(f"{name:<20} {val:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python rand_forest.py <path_to_features_csv>")
        sys.exit(1)

    run_random_forest(sys.argv[1])


if __name__ == "__main__":
    main()
