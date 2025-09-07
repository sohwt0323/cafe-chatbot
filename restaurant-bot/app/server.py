from flask import Flask, request, jsonify
import joblib, numpy as np
from collections import defaultdict
import re, os

# tolerant imports
try:
    from app.brain import handle
    from app.utils import any_catalog_name_in
except ImportError:
    from brain import handle
    from utils import any_catalog_name_in

# ---------------- Model loading & helpers ----------------
MODEL_DIR = "models"
MODEL_FILES = {
    "lr":  os.path.join(MODEL_DIR, "intent_model_lr.joblib"),
    "nb":  os.path.join(MODEL_DIR, "intent_model_nb.joblib"),
    "svm": os.path.join(MODEL_DIR, "intent_model_svm.joblib"),
}
LEGACY_DEFAULT = os.path.join(MODEL_DIR, "intent_model.joblib")  # old logistic model name

def _safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

MODELS = {k: m for k, p in MODEL_FILES.items() if (m := _safe_load(p))}
# If you only trained the legacy file, treat it as LR
if "lr" not in MODELS:
    legacy = _safe_load(LEGACY_DEFAULT)
    if legacy is not None:
        MODELS["lr"] = legacy

if not MODELS:
    raise RuntimeError(
        "No intent models found. Train and save to models/intent_model_{lr,nb,svm}.joblib "
        "or models/intent_model.joblib (legacy LR)."
    )

DEFAULT_ALGO = "lr" if "lr" in MODELS else next(iter(MODELS.keys()))

def predict_with_model(clf, text):
    """Return (classes, probs, idx_of_max) for either proba or decision_function models."""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([text])[0]
        classes = clf.classes_
    else:
        # e.g., LinearSVC without proba
        scores = clf.decision_function([text])[0]
        classes = clf.classes_
        scores = np.atleast_1d(scores)
        exps = np.exp(scores - np.max(scores))
        probs = exps / exps.sum()
    idx = int(np.argmax(probs))
    return classes, probs, idx

# ---------------- App state & regexes ----------------
THRESH = 0.50
user_state = defaultdict(dict)

# whole-word greeting (no "hi" inside "chicken")
GREET_RE = re.compile(r"\b(hi|hello|hey)\b", re.IGNORECASE)

# price phrases (with typo tolerance for "mush")
PRICE_PATTERNS = [
    re.compile(r"\bprice\b"), re.compile(r"\bprice\s+of\b"),
    re.compile(r"\bcost\b"),  re.compile(r"\bcost\s+of\b"),
    re.compile(r"\bhow\s+m[uo]?sh\b"), re.compile(r"\bhow\s+m[uo]?sh\s+(is|are)\b"),
    re.compile(r"\bhow\s+much\b"),     re.compile(r"\bhow\s+much\s+(is|are)\b"),
]

# chef recommendation / special / signature (with typo tolerance)
CHEF_PATTERNS = [
    re.compile(r"chef'?s?\s*(recom+\w*d|recommend(ed|ation)?|choice|pick|special|signature)"),
    re.compile(r"(recom+\w*d|recommend(ed|ation)?)\s+by\s+chef"),
    re.compile(r"\bchef\s+recommend"),
]

def client_id(req):
    return (req.remote_addr or "local")

app = Flask(__name__, static_folder="static")

# ---------------- Routes ----------------
@app.get("/")
def serve_chat():
    return app.send_static_file("chat.html")

@app.post("/set_algo")
def set_algo():
    data = request.get_json(force=True) or {}
    algo = (data.get("algo") or "").lower()
    if algo not in MODELS:
        return jsonify({"ok": False, "error": f"Model '{algo}' not available"}), 400
    user_state[client_id(request)]["algo"] = algo
    return jsonify({"ok": True, "algo": algo})

@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    requested_algo = (data.get("algo") or "").lower()

    if not text:
        return jsonify({"intent": "fallback", "confidence": 0, "reply": "Say something 🙂"}), 200

    t = text.lower()
    cid = client_id(request)
    state = user_state[cid]

    # choose model (payload > saved per-user > default)
    algo = requested_algo or state.get("algo") or DEFAULT_ALGO
    clf = MODELS.get(algo, MODELS[DEFAULT_ALGO])

    # ---------- reservation flow ----------
    if state.get("reservation_stage") == "awaiting_people":
        m = re.search(r"(\d+)", t)
        ppl = m.group(1) if m else text
        state["people"] = ppl
        state["reservation_stage"] = "awaiting_time"
        return jsonify({"intent": "make_reservation", "confidence": 1.0,
                        "reply": f"Got it, {ppl} people. What time would you like?",
                        "algo": algo}), 200

    if state.get("reservation_stage") == "awaiting_time":
        state["time"] = text
        ppl = state.get("people", "some")
        time_str = state.get("time", text)
        state.clear()
        return jsonify({"intent": "make_reservation", "confidence": 1.0,
                        "reply": f"✅ Reservation confirmed for {ppl} people at {time_str}. Thank you!",
                        "algo": algo}), 200

    # ---------- one-turn memory ----------
    if state.get("expecting") == "price_item":
        intent, conf = "price_query", 1.0
        state["expecting"] = None
    else:
        # ---------- rules (order matters) ----------
        menu_words      = ["menu", "drinks", "drink", "milkshake", "milk shake", "shake", "dessert", "coffee", "tea"]
        book_words      = ["book", "booking", "reserve", "reservation"]
        hours_words     = ["opening hour", "opening hours", "operating hour", "operating hours",
                           "business hour", "business hours", "what time do you open",
                           "what time open", "what time close", "closing time", "open time"]
        payment_words   = ["payment", "pay", "payment method", "payment methods", "pay method",
                           "cash", "card", "visa", "master", "mastercard", "credit card", "debit card",
                           "grabpay", "tng", "touch n go", "e-wallet", "ewallet"]
        location_words  = ["where are you", "location", "address", "parking", "car park"]
        recommend_words = ["suggest", "recommend", "recommendation", "pick one", "choose one",
                           "how about", "any good", "what's good"]
        best_words      = ["best seller", "bestseller", "best sellers", "most popular",
                           "top pick", "top picks", "top seller", "signature", "popular"]

        if any(w in t for w in hours_words):
            intent, conf = "opening_hours", 1.0
        elif any(p.search(t) for p in PRICE_PATTERNS):
            intent, conf = "price_query", 1.0
        elif any(p.search(t) for p in CHEF_PATTERNS):
            intent, conf = "dish_query", 1.0       # chef picks handled in brain.py
        elif any(w in t for w in best_words):
            intent, conf = "dish_query", 1.0
        elif any(w in t for w in recommend_words):
            intent, conf = "dish_query", 1.0
        elif any(w in t for w in payment_words):
            intent, conf = "payment_methods", 1.0
        elif any(w in t for w in location_words):
            intent, conf = "location_parking", 1.0
        elif any_catalog_name_in(t):
            intent, conf = "dish_query", 1.0
        elif any(w in t for w in menu_words):
            intent, conf = "menu_items", 1.0
        elif any(w in t for w in book_words):
            intent, conf = "make_reservation", 1.0
        elif GREET_RE.search(t):
            intent, conf = "greet", 1.0
        else:
            classes, probs, idx = predict_with_model(clf, text)
            intent, conf = classes[idx], float(probs[idx])

    if conf < THRESH:
        intent = "fallback"

    reply = handle(intent, text)

    if intent == "price_query" and "Which dish price" in reply:
        state["expecting"] = "price_item"
    if intent == "make_reservation" and "how many" in reply.lower():
        state["reservation_stage"] = "awaiting_people"

    return jsonify({"intent": intent, "confidence": conf, "reply": reply, "algo": algo}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055)
