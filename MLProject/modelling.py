#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path):
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    return X_train, X_test, y_train, y_test

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(args):
    X_train, X_test, y_train, y_test = load_data(args.data_path)

    mlflow.set_experiment("Telco_Churn_Experiment_RF")
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(n_estimators=100, random_state=args.random_seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc}")

        # save confusion matrix artifact
        os.makedirs("artifacts", exist_ok=True)
        cm_path = os.path.join("artifacts", "confusion_matrix.png")
        plot_confusion(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # explicit model log (autolog already logs, but explicit ensures artifact)
        mlflow.sklearn.log_model(model, "model_explicit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="telco_preprocessing")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
