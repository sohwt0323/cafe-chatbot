# scripts/check_intent_counts.py
import pandas as pd
df = pd.read_csv("data/intents.csv")
counts = df["intent"].value_counts().sort_values()
print(counts)
print("\n⚠️ Intents with < 2 samples:")
print(counts[counts < 2])
