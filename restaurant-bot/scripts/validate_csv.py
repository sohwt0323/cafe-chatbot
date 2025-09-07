import sys, csv

PATH = "data/intents.csv"
bad = []
with open(PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, start=1):
        if i == 1:
            # header must be exactly ['text','intent']
            if row != ["text","intent"]:
                print(f"⚠️  Header should be: text,intent  (found: {row})")
            continue
        if len(row) != 2:
            bad.append((i, row))

if bad:
    print("❌ Problem rows (not exactly 2 columns):")
    for line, row in bad:
        print(f"  line {line}: {row}")
    print("\nHint: put double quotes around any text that contains commas.")
else:
    print("✅ intents.csv looks structurally OK (2 columns per row).")
