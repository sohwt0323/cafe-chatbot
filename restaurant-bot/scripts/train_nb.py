#!/usr/bin/env python3
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report

DATA = "data/intents.csv"
OUT  = "models/intent_model_nb.joblib"   # NB model file

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
        "payment":"payment_methods","payment_method":"payment_methods",
        "payment_methods":"payment_methods","pay":"payment_methods",
        "location":"location_parking","parking":"location_parking","address":"location_parking",
    }
    df["intent"] = df["intent"].str.replace(r"\s+","_",regex=True).map(lambda x: alias_map.get(x,x))
    df = df[(df["text"]!="") & (df["intent"]!="")]
    return df

df = load_intents(DATA)
counts = df["intent"].value_counts()
split_kwargs = dict(test_size=0.2, random_state=42)
if counts.min() >= 2:
    split_kwargs["stratify"] = df["intent"]
else:
    print("⚠️ Some intents have <2 samples; training without stratify.")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], **split_kwargs)

pipe = Pipeline([
    ("feats", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)),
    ])),
    ("clf", ComplementNB())   # NB supports predict_proba
])

pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
joblib.dump(pipe, OUT)
print("✅ Saved NB model to", OUT)
