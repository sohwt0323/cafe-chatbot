import csv, io, pathlib

src = pathlib.Path("data/intents.csv")
bak = src.with_suffix(".csv.bak")

text = src.read_text(encoding="utf-8")
bak.write_text(text, encoding="utf-8")

out_lines = []
for i, line in enumerate(text.splitlines(), start=1):
    if i == 1:
        out_lines.append("text,intent")
        continue
    # Skip empty lines
    if not line.strip():
        continue
    # If already valid with 1 comma, keep
    if line.count(",") == 1:
        out_lines.append(line)
        continue
    # If more than 1 comma, try to wrap the first field in quotes
    # Find last comma as the separator between text and intent
    last = line.rfind(",")
    if last == -1:
        # no comma at all -> invalid, skip or warn
        print(f"⚠️  line {i} has no comma, skipping:", line)
        continue
    text_field = line[:last].strip()
    intent_field = line[last+1:].strip()
    # Ensure intent has no comma
    if "," in intent_field:
        print(f"⚠️  line {i} intent contains comma. Please fix manually:", line)
    # Quote the text field if not already quoted
    if not (text_field.startswith('"') and text_field.endswith('"')):
        text_field = '"' + text_field.replace('"','""') + '"'
    out_lines.append(f"{text_field},{intent_field}")

src.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print(f"✅ Wrote fixed file. Backup at {bak}")

