import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_and_prepare(csv_file):
    df = pd.read_csv(csv_file, sep=";")
    def label(q):
        if q <= 5: return "Low"
        elif q <= 7: return "Medium"
        else: return "High"
    df["quality_label"] = df["quality"].apply(label)
    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]
    return X, y

def train(wine_type, csv_file):
    print(f"\nTraining model: {wine_type.upper()} WINE")
    X, y = load_and_prepare(csv_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.2%}")
    print(classification_report(y_test, y_pred))
    joblib.dump(model,           f"model_{wine_type}.pkl")
    joblib.dump(scaler,          f"scaler_{wine_type}.pkl")
    joblib.dump(list(X.columns), f"features_{wine_type}.pkl")
    print(f"Model disimpan: model_{wine_type}.pkl")
    return acc

acc_red   = train("red",   "winequality-red.csv")
acc_white = train("white", "winequality-white.csv")
print(f"\nRed Wine Accuracy  : {acc_red:.2%}")
print(f"White Wine Accuracy: {acc_white:.2%}")
print("\nSemua model berhasil disimpan!")