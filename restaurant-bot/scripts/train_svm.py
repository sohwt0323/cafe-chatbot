#!/usr/bin/env python3
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

DATA = "data/intents.csv"
OUT  = "models/intent_model_svm.joblib"

def load_intents(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower()

    if "text" not in df.columns:
        for cand in ["utterance", "question", "query", "message"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "text"})
                break
    if "text" not in df.columns or "intent" not in df.columns:
        raise SystemExit(f"CSV must have headers 'text,intent'. Found: {list(df.columns)}")

    df = df[["text", "intent"]].dropna().copy()
    df["text"]   = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()

    alias_map = {
        "opening_hour":"opening_hours","opening_hours":"opening_hours",
        "operating_hour":"opening_hours","operating_hours":"opening_hours",
        "business_hour":"opening_hours","business_hours":"opening_hours",
        "payment":"payment_methods","payment_method":"payment_methods","payment_methods":"payment_methods","pay":"payment_methods",
        "location":"location_parking","parking":"location_parking","address":"location_parking",
    }
    df["intent"] = df["intent"].str.replace(r"\s+","_",regex=True).map(lambda x: alias_map.get(x,x))
    df = df[(df["text"]!="") & (df["intent"]!="")]
    return df

df = load_intents(DATA)
counts = df["intent"].value_counts()
min_count = int(counts.min())
print(f"Class counts (min={min_count}):\n{counts}\n")

split_kwargs = dict(test_size=0.2, random_state=42)
if min_count >= 2:
    split_kwargs["stratify"] = df["intent"]
else:
    print("⚠️ Some intents have <2 samples; training without stratify.")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], **split_kwargs)

# ---- Choose SVM estimator depending on data size ----
base = LinearSVC(class_weight="balanced")

if min_count >= 2:
    # use the largest CV we safely can (at least 2, at most 5)
    cv = max(2, min(5, min_count))
    print(f"Using CalibratedClassifierCV with cv={cv}")
    try:
        svm_est = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)  # new sklearn
    except TypeError:
        svm_est = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=cv)  # old sklearn
else:
    print("⚠️ At least one class has only 1 sample; using plain LinearSVC (no calibration).")
    svm_est = base  # server can handle no predict_proba

pipe = Pipeline([
    ("feats", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)),
    ])),
    ("clf", svm_est),
])

pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
joblib.dump(pipe, OUT)
print("✅ Saved SVM model to", OUT)
