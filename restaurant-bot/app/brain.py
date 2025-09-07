# app/brain.py
import json, re, csv
from pathlib import Path

# tolerant import for utils (works whether running as a package or a script)
try:
    from app.utils import match_dishes, list_by_tag
except ImportError:
    from utils import match_dishes, list_by_tag  # fallback

# ---- Hours config (edit here, one place) ----
OPEN_DAYS  = "daily"
OPEN_TIME  = "10am"
CLOSE_TIME = "10pm"
LAST_ORDER = "30 minutes before closing"

# ---- Curated chef picks (EDIT to your real dishes) ----
CHEF_OVERRIDES = [
    "Honey Butter Fried Chicken",
    "Spicy Sambal Prawns",
    "Ayam Masak Merah",
]

# -------------------------------------------------------
# Load menu
# -------------------------------------------------------
MENU_PATHS = [
    Path("data/menu.json"),
    Path(__file__).resolve().parent / "data" / "menu.json",
    Path(__file__).resolve().parent.parent / "data" / "menu.json",
]
MENU = None
for p in MENU_PATHS:
    if p.exists():
        MENU = json.loads(p.read_text(encoding="utf-8"))
        break
if MENU is None:
    MENU = []  # fail-safe

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _format_price(price):
    try:
        p = float(str(price).strip().replace("RM", ""))
        s = f"{p:.2f}"
        return s.rstrip("0").rstrip(".")
    except Exception:
        return str(price).strip()

def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[-_/]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()

def _popular_first(items):
    return sorted(items, key=lambda x: x.get("popularity", 0), reverse=True)

def _has_price(it):
    if not isinstance(it, dict):
        return False
    price = it.get("price", None)
    if price is None:
        return False
    s = str(price).strip().lower()
    return s not in ("", "nan", "none")

# -------------------------------------------------------
# Make a tolerant price index from menu + CSVs
# -------------------------------------------------------
PRICE_IDX = {}

def _ingest_price_pair(name, price):
    if not name:
        return
    m = re.search(r"(\d+(?:\.\d+)?)", str(price))
    if not m:
        return
    PRICE_IDX[_norm_name(name)] = float(m.group(1))

def _load_price_index():
    # 1) prices already present in menu.json
    for it in MENU:
        if _has_price(it):
            _ingest_price_pair(it.get("name", ""), it.get("price"))

    # 2) tolerant CSV readers (food.csv, Item_to_id.csv, items.csv)
    candidates = [
        Path("data/food.csv"),
        Path(__file__).resolve().parent / "data" / "food.csv",
        Path(__file__).resolve().parent.parent / "data" / "food.csv",
        Path("data/Item_to_id.csv"),
        Path(__file__).resolve().parent / "data" / "Item_to_id.csv",
        Path(__file__).resolve().parent.parent / "data" / "Item_to_id.csv",
        Path("data/items.csv"),
        Path(__file__).resolve().parent / "data" / "items.csv",
        Path(__file__).resolve().parent.parent / "data" / "items.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows = []

        # Try using csv.Sniffer to detect the delimiter; if it fails, fallback to regex split.
        try:
            dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;|\t")
            reader = csv.reader(text.splitlines(), dialect)
            rows = list(reader)
        except Exception:
            rows = [re.split(r"[;,|\t]", line) for line in text.splitlines()]

        for row in rows:
            cells = [c.strip() for c in row if c and c.strip()]
            if not cells:
                continue
            name, price = None, None
            for c in cells:
                if name is None and re.search(r"[A-Za-z]", c) and "price" not in c.lower():
                    name = c
                if price is None and re.search(r"\d", c):
                    m = re.search(r"(\d+(?:\.\d+)?)", c)
                    if m:
                        price = m.group(1)
            if name and price is not None:
                _ingest_price_pair(name, price)

_load_price_index()

def _format_with_idx(name_or_item):
    """Format 'Name (RMxx)' using menu price OR PRICE_IDX if needed."""
    if isinstance(name_or_item, dict):
        it = name_or_item
        name = (it.get("name") or "").strip() or "Item"
        if _has_price(it):
            return f"{name} (RM{_format_price(it['price'])})"
        p = PRICE_IDX.get(_norm_name(name))
        return f"{name} (RM{_format_price(p)})" if p is not None else name
    # plain string
    nm = str(name_or_item)
    p = PRICE_IDX.get(_norm_name(nm))
    return f"{nm} (RM{_format_price(p)})" if p is not None else nm

def _find_by_names(names):
    lookup = { _norm_name(it.get("name","")): it for it in MENU }
    hits = []
    for nm in names:
        it = lookup.get(_norm_name(nm))
        if it:
            hits.append(it)
    return hits

def _find_price_by_name_like(name):
    """Return an item dict (possibly synthesized) that has a price for 'name'."""
    qn = _norm_name(name)
    if not qn:
        return None
    # exact/substring in menu with price
    for it in MENU:
        nm = _norm_name(it.get("name",""))
        if (nm == qn or qn in nm or nm in qn) and _has_price(it):
            return it
    # from price index (CSV) — synthesize an item
    p = PRICE_IDX.get(qn)
    if p is not None:
        return {"name": name, "price": p}
    # try any key in index that contains query (or vice versa)
    for key, val in PRICE_IDX.items():
        if qn in key or key in qn:
            return {"name": name, "price": val}
    return None

# -------------------------------------------------------
# Main handler
# -------------------------------------------------------
def handle(intent, text):
    """Return a bot reply string for the given intent and user text."""
    t = _normalize(text)

    # ---------- info intents ----------
    if intent == "opening_hours":
        if re.search(r"\b(close|closing|closing\s+time|closing\s+hour|close\s*time|last\s*order)\b", t):
            return f"We close at {CLOSE_TIME}. Last order is {LAST_ORDER}."
        return f"We’re {OPEN_DAYS} from {OPEN_TIME} to {CLOSE_TIME}."

    if intent == "location_parking":
        return "We’re at PV128, Setapak (above Togather Cafe). Free parking after 6pm."

    if intent == "payment_methods":
        return "We accept cash, Visa/Master, GrabPay, and TNG eWallet."

    if intent == "delivery_info":
        return "Yes, we deliver within 5km. Order via our website or WhatsApp."

    if intent == "takeaway_info":
        return "Yes, takeaway is available."

    # ---------- menu / categories ----------
    if intent == "menu_items":
        if any(w in t for w in ["milkshake", "milk shake", "shake"]):
            ms = list_by_tag("milkshake")
            if ms:
                return "Drinks (milkshakes): " + ", ".join(_format_with_idx(x) for x in ms)
        if any(w in t for w in ["drink", "drinks", "coffee", "tea", "beverage"]):
            dr = list_by_tag("drink")
            if dr:
                return "Drinks: " + ", ".join(_format_with_idx(x) for x in dr[:12])
        if any(w in t for w in ["dessert", "desserts", "sweet"]):
            ds = list_by_tag("dessert")
            if ds:
                return "Desserts: " + ", ".join(_format_with_idx(x) for x in ds[:12])
        popular = _popular_first(MENU)[:5] if MENU else []
        if popular:
            return "Popular now: " + ", ".join(_format_with_idx(m) for m in popular)
        return "Our menu is being updated—try asking for drinks, milkshakes or desserts."

    # ---------- dietary / preferences ----------
    if intent == "dietary_options":
        if "vegan" in t:
            vs = list_by_tag("vegan")
            return "Vegan dishes: " + ", ".join(_format_with_idx(m) for m in vs) if vs else "We can make some dishes vegan on request."
        if "vegetarian" in t:
            vg = list_by_tag("vegetarian")
            return "Vegetarian options: " + ", ".join(_format_with_idx(m) for m in vg) if vg else "Yes, we have vegetarian options."
        if "spicy" in t:
            sp = list_by_tag("spicy")
            return "Spicy picks: " + ", ".join(_format_with_idx(m) for m in sp) if sp else "We can make dishes spicier on request."
        if "sweet" in t or "dessert" in t:
            ds = list_by_tag("dessert")
            return "Desserts / sweet picks: " + ", ".join(_format_with_idx(m) for m in ds) if ds else "We’ve got some sweet treats too!"
        return "We have vegan, vegetarian, spicy, dessert and drinks—any preference?"

    # ---------- dish lookup / recommendation / best-sellers / chef ----------
    if intent == "dish_query":
        want_one = any(k in t for k in ["suggest", "recommend", "recommendation", "pick one", "choose one", "how about"])
        ask_best = any(k in t for k in [
            "best seller", "bestseller", "best sellers", "best selling",
            "most popular", "top pick", "top picks", "top seller",
            "signature", "popular"
        ])
        ask_chef = ("chef" in t) and any(k in t for k in ["recom", "special", "signature", "choice", "pick"])

        if ask_chef:
            picks = _find_by_names(CHEF_OVERRIDES)
            if not picks:
                raw = []
                for tag in ["chef", "signature", "recommended", "special", "best"]:
                    raw += list_by_tag(tag)
                seen, uniq = set(), []
                for it in raw:
                    n = _norm_name(it.get("name",""))
                    if not n or n in seen:
                        continue
                    seen.add(n); uniq.append(it)
                picks = _popular_first(uniq) if uniq else _popular_first(MENU)
            picks = picks[:3] if picks else []
            if picks:
                return "Chef’s recommendations today: " + ", ".join(_format_with_idx(x) for x in picks)
            return "Chef is crafting something special today!"

        if ask_best:
            if any(k in t for k in ["milkshake", "milk shake", "shake"]):
                ms = _popular_first(list_by_tag("milkshake"))[:3]
                if ms: return "Best-selling milkshakes: " + ", ".join(_format_with_idx(x) for x in ms)
            if any(k in t for k in ["dessert", "desserts", "sweet"]):
                ds = _popular_first(list_by_tag("dessert"))[:3]
                if ds: return "Best-selling desserts: " + ", ".join(_format_with_idx(x) for x in ds)
            if any(k in t for k in ["drink", "drinks", "coffee", "tea", "beverage"]):
                dr = _popular_first(list_by_tag("drink"))[:3]
                if dr: return "Best-selling drinks: " + ", ".join(_format_with_idx(x) for x in dr)
            overall = _popular_first(MENU)[:3]
            return "Our best sellers: " + ", ".join(_format_with_idx(x) for x in overall) if overall else "Our best sellers change daily!"

        hits = match_dishes(t)
        if hits:
            if want_one:
                best = hits[0]
                return f"I'd suggest: {_format_with_idx(best)}. Want another recommendation?"
            if any(k in t for k in ["how about", "this one", "that one"]) or len(hits) == 1:
                best = hits[0]
                return f"{_format_with_idx(best)} — great choice!"
            return "You might like: " + ", ".join(_format_with_idx(h) for h in hits[:6])

        if any(k in t for k in ["milkshake", "milk shake", "shake"]):
            ms = list_by_tag("milkshake")
            if ms:
                choice = _popular_first(ms)[0]
                return f"Try this milkshake: {_format_with_idx(choice)}."
        if any(k in t for k in ["dessert", "desserts", "sweet"]):
            ds = list_by_tag("dessert")
            if ds:
                choice = _popular_first(ds)[0]
                return f"My pick: {_format_with_idx(choice)}."
        return "Which dish are you looking for?"

    # ---------- price ----------
    if intent == "price_query":
        q = t
        q = re.sub(r"\b(how\s+m[uo]?sh|how\s+much)\s+(is|are)\b", " ", q)
        q = re.sub(r"\b(how\s+m[uo]?sh|how\s+much)\b", " ", q)
        q = re.sub(r"\b(price|cost)\s+(of|for)\b", " ", q)
        q = re.sub(r"\b(price|cost)\b", " ", q)
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()

        # category prices
        if any(k in q for k in ["milkshake", "shake"]):
            ms = list_by_tag("milkshake")
            if ms:
                return "Prices (milkshakes): " + "; ".join(_format_with_idx(x) for x in ms[:8])
        if any(k in q for k in ["dessert", "desserts", "sweet"]):
            ds = list_by_tag("dessert")
            if ds:
                return "Prices (desserts): " + "; ".join(_format_with_idx(x) for x in ds[:8])

        # specific dish
        hits = match_dishes(q or t)
        if hits:
            item = hits[0]
            if _has_price(item):
                return "Price: " + _format_with_idx(item)
            alt = _find_price_by_name_like(item.get("name", "")) or _find_price_by_name_like(q)
            if alt and ( _has_price(alt) or (isinstance(alt, dict) and alt.get("price") is not None) ):
                return "Price: " + _format_with_idx(alt)
            alts = [h for h in hits[1:6] if _has_price(h) or PRICE_IDX.get(_norm_name(h.get("name","")))]
            if alts:
                return f"Price for {item.get('name','this item')} isn’t listed. Similar items: " + "; ".join(_format_with_idx(a) for a in alts)
            return f"Sorry, I couldn’t find a price for {item.get('name','that item')}."

        # last-chance: try direct lookup by the raw text
        alt = _find_price_by_name_like(q)
        if alt:
            return "Price: " + _format_with_idx(alt)

        return "Which dish price are you asking for?"

    # ---------- reservations ----------
    if intent == "make_reservation":
        return "Sure, how many people?"
    if intent == "modify_reservation":
        return "Okay, what’s the new time or party size?"
    if intent == "cancel_reservation":
        return "Please provide your booking name or phone number to cancel."

    # ---------- greet / goodbye ----------
    if intent == "greet":
        return "Hello! How can I help you today?"
    if intent == "goodbye":
        return "Thanks for visiting—see you soon!"

    # ---------- fallback ----------
    return "Sorry, I didn’t catch that. Do you want to see our menu or make a booking?"
