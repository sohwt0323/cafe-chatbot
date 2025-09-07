#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix

DATA = "data/intents.csv"
OUT_DIR = Path("models"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if df.shape[1] == 1:  # ; separated?
        df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower()
    if "text" not in df.columns:
        for c in ["utterance","question","query","message"]:
            if c in df.columns: df = df.rename(columns={c:"text"}); break
    if "text" not in df.columns or "intent" not in df.columns:
        raise SystemExit(f"Need 'text,intent' columns, got {list(df.columns)}")
    df = df[["text","intent"]].dropna().copy()
    df["text"]   = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()
    alias = {
        "opening_hour":"opening_hours","operating_hour":"opening_hours","operating_hours":"opening_hours",
        "business_hour":"opening_hours","business_hours":"opening_hours",
        "payment":"payment_methods","payment_method":"payment_methods","pay":"payment_methods",
        "location":"location_parking","parking":"location_parking","address":"location_parking",
    }
    df["intent"] = df["intent"].str.replace(r"\s+","_",regex=True).map(lambda x: alias.get(x,x))
    return df[(df["text"]!="") & (df["intent"]!="")]

def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)),
    ])

def svm_estimator(min_count):
    base = LinearSVC(class_weight="balanced")
    if min_count < 2:
        # no calibration possible
        return base
    cv = max(2, min(5, int(min_count)))
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=cv)

def make_pipelines(min_count):
    return {
        "LR":  Pipeline([("feats", features()), ("clf", LogisticRegression(max_iter=400, class_weight="balanced", C=3.0))]),
        "NB":  Pipeline([("feats", features()), ("clf", ComplementNB())]),
        "SVM": Pipeline([("feats", features()), ("clf", svm_estimator(min_count))]),
    }

def main():
    df = load_data(DATA)
    counts = df["intent"].value_counts()
    min_count = counts.min()
    split = dict(test_size=0.2, random_state=42)
    if min_count >= 2: split["stratify"] = df["intent"]
    else: print("⚠️ Some intents <2 samples; no stratify.")

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], **split)
    models = make_pipelines(min_count)

    rows = []
    for name, pipe in models.items():
        print(f"\n=== Training {name} ===")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print(classification_report(y_test, y_pred))
        print("Macro F1:", round(macro_f1, 4))
        # save model too (optional)
        out = OUT_DIR / f"intent_model_{name.lower()}.joblib"
        try:
            import joblib; joblib.dump(pipe, out)
            print("Saved:", out)
        except Exception as e:
            print("Save failed:", e)
        # keep predictions
        rows.append(pd.DataFrame({"algo":name, "text":X_test.values, "true":y_test.values, "pred":y_pred}))

        # optional: confusion matrix
        labels = sorted(df["intent"].unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
        cm_df.to_csv(f"models/confusion_{name.lower()}.csv", encoding="utf-8", index=True)

    comp = pd.concat(rows, ignore_index=True)
    comp_pivot = comp.pivot_table(index=["text","true"], columns="algo", values="pred", aggfunc="first").reset_index()
    comp_pivot.to_csv("models/predictions_comparison.csv", index=False, encoding="utf-8")
    print("\n📄 Wrote models/predictions_comparison.csv (side-by-side predictions).")

if __name__ == "__main__":
    main()
