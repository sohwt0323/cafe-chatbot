import joblib, numpy as np

MODEL = "models/intent_model.joblib"
clf = joblib.load(MODEL)

def predict_intent(text, threshold=0.65):
    probs = clf.predict_proba([text])[0]
    classes = clf.classes_
    idx = int(np.argmax(probs))
    intent, conf = classes[idx], float(probs[idx])
    if conf < threshold:
        return "fallback", conf
    return intent, conf

while True:
    q = input("You: ").strip()
    if not q: break
    intent, conf = predict_intent(q)
    print(f"→ intent={intent} conf={conf:.2f}")
