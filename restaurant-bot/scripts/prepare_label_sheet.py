import pandas as pd, re

RAW = "data/conversationo.csv"
OUT = "data/intents_to_label.csv"

df = pd.read_csv(RAW, engine="python", on_bad_lines="skip")
df = df.drop_duplicates(subset=["Question"]).dropna(subset=["Question"])
df["text"] = df["Question"].str.strip()

# keyword rules to suggest intents
rules = {
    "opening_hours": r"open|hours|time.*open|close",
    "menu_items": r"\bmenu\b|what.*(serve|available)|recommend|popular",
    "dish_query": r"(chicken|prawn|beef|rice|soup|dessert|cake|coffee|tea)",
    "price_query": r"how much|price|cost",
    "dietary_options": r"vegetarian|vegan|halal|spicy|gluten|allerg",
    "delivery_info": r"deliver|delivery|order online|home delivery",
    "takeaway_info": r"take ?away|pickup|pick up|self[- ]?collect",
    "make_reservation": r"book|reservation|reserve|table|seat",
    "modify_reservation": r"change.*(booking|reservation)|reschedule",
    "cancel_reservation": r"cancel.*(booking|reservation)",
    "location_parking": r"address|where.*located|location|parking|landmark",
    "payment_methods": r"pay|payment|cash|card|visa|master|grabpay|tng",
    "greet": r"\b(hi|hello|hey|good (morning|afternoon|evening))\b",
    "goodbye": r"\b(bye|goodbye|see you|thanks|thank you)\b",
}

def guess(t):
    t = str(t).lower()
    for intent, pat in rules.items():
        if re.search(pat, t):
            return intent
    return ""

df["suggested_intent"] = df["text"].apply(guess)
df[["text","suggested_intent"]].to_csv(OUT, index=False)
print(f"✅ Saved {len(df)} rows to {OUT}. Now open it and correct/fill intents, then save as data/intents.csv")
