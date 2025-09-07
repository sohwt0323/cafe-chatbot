import json, re
from pathlib import Path

# Optional fuzzy matcher
try:
    from rapidfuzz import process, fuzz
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False
    from difflib import get_close_matches

MENU = json.loads(Path("data/menu.json").read_text(encoding="utf-8"))

# ---- load extra catalog rows from CSVs (name,price,tags) ----
def load_extra_items():
    out = []
    for fname in ["data/food_cleaned.csv", "data/food_full.csv", "data/food_from_conversation.csv", "data/food.csv", "data/Item_to_id.csv"]:
        p = Path(fname)
        if not p.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(p)
            cols = {c.lower().strip(): c for c in df.columns}
            name_col = cols.get("name") or cols.get("item") or cols.get("dish")
            if not name_col:
                continue
            price_col = cols.get("price")
            tags_col  = cols.get("tags")
            for _, r in df.iterrows():
                name = str(r[name_col]).strip()
                if not name or name.lower() == "nan":
                    continue
                item = {"name": name}
                if price_col and price_col in df.columns and str(r[price_col]) != "nan":
                    item["price"] = r[price_col]
                if tags_col and tags_col in df.columns and str(r[tags_col]) != "nan":
                    tags = re.split(r"[;,]\s*", str(r[tags_col]))
                    item["tags"] = [t.strip() for t in tags if t.strip()]
                out.append(item)
        except Exception:
            pass
    return out

EXTRA = load_extra_items()

# ---- build unified catalog ----
CATALOG = []
seen = set()
for src in (MENU + EXTRA):
    key = src.get("name","").strip().lower()
    if not key or key in seen:
        continue
    seen.add(key)
    # ensure tags list
    if isinstance(src.get("tags"), str):
        src["tags"] = [t.strip() for t in re.split(r"[;,]\s*", src["tags"]) if t.strip()]
    CATALOG.append(src)

# ---- synonyms / aliases (add your own here) ----
ALIASES = {
    "tomyam": "Tom Yum Soup",
    "tom yam": "Tom Yum Soup",
    "tom yum": "Tom Yum Soup",
    "simple coffee": "Simple Coffee",
    "americano coffee": "Americano",
    "latte": "Latte Macchiato",   # pick your preferred latte name
}

CATEGORY_SYNONYMS = {
    "milkshake": ["milkshake", "shake", "milk shake"],
    "drink": ["drink", "drinks", "beverage", "coffee", "tea", "juice"],
    "dessert": ["dessert", "sweet", "sweets"],
    "pastry": ["pastry", "puff", "roll", "pastries"],
}

def normalize_query(q: str) -> str:
    t = q.lower().strip()
    # expand category synonyms into query to increase hits
    for cat, syns in CATEGORY_SYNONYMS.items():
        if any(s in t for s in syns):
            t += " " + " ".join(syns)
    # map aliases to canonical names
    for k, v in ALIASES.items():
        if k in t:
            t = t.replace(k, v.lower())
    return t

def _format_name_for_fuzzy(item):
    # prefer canonical name for fuzzy lists
    return item["name"]

def match_dishes(text: str, limit: int = 6):
    """Return best-matching items by name or tag (handles typos/synonyms)."""
    q = normalize_query(text)
    hits = []

    # 1) exact tag hits
    q_tokens = set(re.findall(r"\w+", q))
    for item in CATALOG:
        tags = [t.lower() for t in item.get("tags", [])]
        if tags and (set(tags) & q_tokens):
            hits.append((100.0, item))

    # 2) alias direct hits
    for k, v in ALIASES.items():
        if k in q:
            for it in CATALOG:
                if it["name"].lower() == v.lower():
                    hits.append((99.0, it))

    # 3) fuzzy name hits
    names = [_format_name_for_fuzzy(it) for it in CATALOG]
    if HAVE_FUZZ and names:
        # Try two scorers for robustness on short names
        # WRatio then token_set_ratio
        for name, score, idx in process.extract(q, names, scorer=fuzz.WRatio, limit=limit*3):
            if score >= 60:
                hits.append((float(score), CATALOG[idx]))
        for name, score, idx in process.extract(q, names, scorer=fuzz.token_set_ratio, limit=limit*3):
            if score >= 60:
                hits.append((float(score), CATALOG[idx]))
    else:
        from difflib import get_close_matches
        for name in get_close_matches(q, names, n=limit*3, cutoff=0.55):
            it = next((x for x in CATALOG if x["name"] == name), None)
            if it:
                hits.append((80.0, it))

    # dedupe & sort
    out, seen_names = [], set()
    for score, it in sorted(hits, key=lambda x: -x[0]):
        key = it["name"].lower()
        if key in seen_names:
            continue
        out.append(it)
        seen_names.add(key)
        if len(out) >= limit:
            break
    return out

def list_by_tag(tag: str, limit: int = 12):
    """Return items that have a tag (e.g., 'milkshake', 'dessert', 'coffee')."""
    t = tag.lower()
    res = [it for it in CATALOG if t in [x.lower() for x in it.get("tags", [])] or t in it["name"].lower()]
    # if tag matches a category synonym, include its synonyms too
    for cat, syns in CATEGORY_SYNONYMS.items():
        if t == cat or t in syns:
            for syn in syns:
                res += [it for it in CATALOG if syn in [x.lower() for x in it.get("tags", [])] or syn in it["name"].lower()]
    # unique by name
    uniq, seen = [], set()
    for it in res:
        k = it["name"].lower()
        if k in seen:
            continue
        uniq.append(it)
        seen.add(k)
        if len(uniq) >= limit:
            break
    return uniq

def any_catalog_name_in(text: str) -> bool:
    t = text.lower()
    return any(it["name"].lower() in t for it in CATALOG)
