import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA = "data/intents.csv"
MODEL = "models/intent_model.joblib"

def load_intents(path):
    # 1) try comma
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    # if only one column, try semicolon
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # accept common header variants
    if "text" not in df.columns:
        for cand in ["utterance", "question", "query", "message"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "text"})
                break

    if "text" not in df.columns or "intent" not in df.columns:
        raise SystemExit(f"CSV must have headers 'text,intent'. Found: {list(df.columns)}")

    # keep only required cols, drop empties
    df = df[["text", "intent"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()

    # ---- alias normalization (critical!) ----
    alias_map = {
        # opening hours
        "opening_hour": "opening_hours",
        "opening_hours": "opening_hours",
        "operating_hour": "opening_hours",
        "operating_hours": "opening_hours",
        "business_hour": "opening_hours",
        "business_hours": "opening_hours",
        # payment
        "payment": "payment_methods",
        "payment_method": "payment_methods",
        "payment_methods": "payment_methods",
        "pay": "payment_methods",
        # location / parking
        "location": "location_parking",
        "parking": "location_parking",
        "address": "location_parking",
    }
    # unify spaces to underscores, then map
    df["intent"] = df["intent"].str.replace(r"\s+", "_", regex=True).map(lambda x: alias_map.get(x, x))

    # remove empties
    df = df[(df["text"] != "") & (df["intent"] != "")]
    return df

df = load_intents(DATA)

# sanity: each class >= 2? (for stratify)
counts = df["intent"].value_counts()
can_stratify = (counts.min() >= 2)

split_kwargs = dict(test_size=0.2, random_state=42)
if can_stratify:
    split_kwargs["stratify"] = df["intent"]
else:
    print("⚠️ Some intents have <2 samples; training without stratify. Add more examples.")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], **split_kwargs)

# stronger features (word + char)
clf = Pipeline([
    ("feats", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)),
    ])),
    ("logreg", LogisticRegression(max_iter=400, class_weight="balanced", C=3.0))
])

clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
joblib.dump(clf, MODEL)
print("✅ Saved model to", MODEL)
