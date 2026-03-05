"""
Fetch product entries from Baserow (Products database, productsDetails table).
Run from local machine; uses REST API with token auth.
"""

import json
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv
from langdetect import DetectorFactory, detect_langs

# Reproducible language detection
DetectorFactory.seed = 0

# Minimum confidence for English; reject if Dutch is top (en/nl are often confused)
ENGLISH_MIN_CONFIDENCE = 0.70

# Dutch words/phrases: if present, do not treat as English (langdetect often mislabels short Dutch as en)
DUTCH_INDICATORS = frozenset([
    "ontwerp", "editie", "exclusieve", "functioneel", "iconisch", "multifunctioneel",
    "perfectie", "tot in de", "en meer", "voor uw", "in uw", "met een", "van de",
    "een moderne", "de perfecte", "uw ruimte", "onze collectie",
    # From 200-row run: short Dutch headings that were misclassified as English
    "zitcomfort", "actief zitcomfort", "omhullend",
    "subliem", "optimaal", "ultiem", "wooncomfort", "pluche comfort",
    "functie", "isolatie", "comfort & functie", "comfort & isolatie",
    "transformeer", " uw ", "uw huis", "uw ruimte",
    "stof", " rhythm stof", "shiitake stof", "luxe rhythm",
    "baenks", "transformeer uw",
    # From english_entries_first_200.json review: additional Dutch headings/terms
    "luxe wol comfort", "actieve ontspanning", "hoogwaardige afwerking",
    "strak & compact", "innovatief", "sensationeel", "duale lichtoplossing",
    "productiviteit", "efficiëntie", "warme led gloed", "flexibele installatie",
    "flexibel comfort", "adaptief comfort", "complete ontspanningsset",
    "boeiend contrast", "actief comfort", "exclusief massief hout",
    "perfecte jori match", "geproportioneerd", "perfecte aanpassing",
    "robuust massief hout", "comfortabele zitting", "authentiek wolvilt",
    "premium wol comfort", "uniek rastereffect", "visionair ribbeldesign",
    "comfortabele omhelzing", "functionaliteit", "perfecte afwerking",
    "artifort kwaliteit", "ontdek de", "salontafel", "vloerlamp", "plafondlamp",
    "draai fauteuil", "bijzettafel", "hanglamp", "modulaire hanglamp",
    "multifunctionele", "antraciet", "pastelgroen", "mat zwart",
    "geborsteld messing", "gepolijst chroom", "modulaire sofasysteem",
])

# Columns to fetch and the only columns checked for English (exact names as in Baserow)
PRODUCT_COLUMNS = [
    "Accordion_Product_Description",
    "Accordion_Product_Configuration",
    "Accordion_Product_Maintenance",
    "Accordion_Product_Sustainability",
    "Lifestyle_Image_Tagline",
    "ModelScrollSection_Heading_1",
    "ModelScrollSection_Paragraph_1",
    "ModelScrollSection_Heading_2",
    "ModelScrollSection_Paragraph_2",
    "ModelScrollSection_Heading_3",
    "ModelScrollSection_Paragraph_3",
    "ModelScrollSection_Heading_4",
    "ModelScrollSection_Paragraph_4",
]

PAGE_SIZE = 200  # Baserow max per page
MAX_VALUE_DISPLAY = 80  # truncate long values when printing
ENGLISH_ENTRIES_SAVE_LIMIT = None  # save first N rows with English to file (None = save all found)
ENGLISH_ENTRIES_OUTPUT_FILE = "english_entries_all.json"  # filename when saving limited entries
FETCH_LIMIT = None  # limit rows fetched from API; set to None to fetch all


def _str_val(v: Any) -> str:
    """String for display; truncate long values."""
    if v is None:
        return "(empty)"
    s = str(v).strip()
    if not s:
        return "(empty)"
    if len(s) > MAX_VALUE_DISPLAY:
        return s[:MAX_VALUE_DISPLAY] + "..."
    return s


def print_row_values(row: dict[str, Any], columns: list[str], indent: str = "      ") -> None:
    """Print one row's column values (truncated) to stdout."""
    for col in columns:
        val = row.get(col)
        print(f"{indent}{col}: {_str_val(val)}", flush=True)


def _has_dutch_indicators(text: str) -> bool:
    """True if text contains common Dutch words/phrases (case-insensitive)."""
    lower = text.lower()
    return any(ind in lower for ind in DUTCH_INDICATORS)


def is_english(text: str) -> bool:
    """
    Return True only if text is confidently English. Uses detect_langs and a Dutch
    keyword check so short Dutch phrases (e.g. "Iconisch Mid-Century Ontwerp") are
    not misclassified as English.
    """
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if _has_dutch_indicators(s):
        return False
    try:
        langs = detect_langs(s)
        if not langs:
            return False
        top = langs[0]
        if top.lang == "nl":
            return False
        if top.lang == "en" and top.prob >= ENGLISH_MIN_CONFIDENCE:
            return True
        return False
    except Exception:
        return False


def get_config() -> dict[str, str]:
    """Load Baserow URL, token, and table ID from environment."""
    load_dotenv()
    url = (os.getenv("BASEROW_URL") or "").rstrip("/")
    token = os.getenv("BASEROW_TOKEN") or ""
    table_id = os.getenv("TABLE_ID") or ""
    if not url or not token:
        print("Missing BASEROW_URL or BASEROW_TOKEN. Copy .env.example to .env and set them.", file=sys.stderr)
        sys.exit(1)
    if not table_id:
        table_id = resolve_table_id(url, token)
        if not table_id:
            print("TABLE_ID not set and could not resolve from database. Set TABLE_ID in .env.", file=sys.stderr)
            sys.exit(1)
    return {"base_url": url, "token": token, "table_id": table_id}


def resolve_table_id(base_url: str, token: str) -> str:
    """If DATABASE_ID is set, list tables and return table ID for 'productsDetails'."""
    database_id = os.getenv("DATABASE_ID") or ""
    if not database_id:
        return ""
    resp = requests.get(
        f"{base_url}/api/database/tables/database/{database_id}/",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if not resp.ok:
        return ""
    data = resp.json()
    for t in data:
        if (t.get("name") or "").strip().lower() == "productsdetails":
            return str(t.get("id", ""))
    return ""


def fetch_all_rows(
    base_url: str,
    token: str,
    table_id: str,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Paginate through table rows and return all rows (with user field names). If max_rows is set, stop after that many."""
    cols = columns or PRODUCT_COLUMNS
    all_rows: list[dict[str, Any]] = []
    page = 1
    size = PAGE_SIZE if max_rows is None else min(PAGE_SIZE, max_rows)
    while True:
        resp = requests.get(
            f"{base_url}/api/database/rows/table/{table_id}/",
            params={
                "user_field_names": "true",
                "page": page,
                "size": size,
            },
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Baserow API error {resp.status_code}: {resp.text}")
        data = resp.json()
        results = data.get("results") or []
        all_rows.extend(results)
        if max_rows is not None and len(all_rows) >= max_rows:
            all_rows = all_rows[:max_rows]
            break
        if not results or len(results) < size:
            break
        page += 1
        if max_rows is not None:
            size = min(PAGE_SIZE, max_rows - len(all_rows))
            if size <= 0:
                break
    return all_rows


def pick_columns(row: dict[str, Any], columns: list[str], keep_id: bool = False) -> dict[str, Any]:
    """Return only the requested columns; missing keys become None. Optionally keep row id."""
    out = {c: row.get(c) for c in columns}
    if keep_id and "id" in row:
        out["id"] = row["id"]
    return out


def fetch_products(columns: list[str] | None = None, max_rows: int | None = FETCH_LIMIT) -> list[dict[str, Any]]:
    """
    Fetch all entries from productsDetails and return only the requested columns.

    Uses BASEROW_URL, BASEROW_TOKEN, and TABLE_ID from environment (.env).
    Optionally set DATABASE_ID to resolve TABLE_ID from table name "productsDetails".
    If max_rows is set (e.g. 20 for test), only that many rows are fetched.

    Returns:
        List of dicts, each with keys from `columns` (default: PRODUCT_COLUMNS).
    """
    cols = columns or PRODUCT_COLUMNS
    cfg = get_config()
    rows = fetch_all_rows(cfg["base_url"], cfg["token"], cfg["table_id"], cols, max_rows=max_rows)
    return [pick_columns(r, cols, keep_id=True) for r in rows]


def find_and_print_english_entries(products: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    """
    For each row, detect which column values are in English. Print only Row ID and
    English columns. Saves the first ENGLISH_ENTRIES_SAVE_LIMIT entries to a JSON file.
    Checks every non-empty value so no English entry is skipped.
    """
    cols = columns or PRODUCT_COLUMNS
    saved: list[dict[str, Any]] = []
    for row in products:
        row_id = row.get("id", "(no id)")
        english_cols: list[tuple[str, Any]] = []
        for col in cols:
            val = row.get(col)
            if val is None:
                continue
            s = str(val).strip()
            if not s:
                continue
            if is_english(s):
                english_cols.append((col, val))
        if english_cols:
            print(f"\nRow ID {row_id} — English column(s):", flush=True)
            for col_name, val in english_cols:
                print(f"  {col_name}: {_str_val(val)}", flush=True)
            limit = ENGLISH_ENTRIES_SAVE_LIMIT
            if limit is None or len(saved) < limit:
                saved.append({
                    "row_id": row_id,
                    "english_columns": {col_name: val for col_name, val in english_cols},
                })
            if limit is not None and len(saved) >= limit:
                break
    if saved:
        with open(ENGLISH_ENTRIES_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved {len(saved)} English entries to {ENGLISH_ENTRIES_OUTPUT_FILE}", flush=True)
    else:
        print("\nNo English entries found. No file saved.", flush=True)


def main() -> None:
    load_dotenv()
    products = fetch_products()
    find_and_print_english_entries(products, PRODUCT_COLUMNS)


if __name__ == "__main__":
    main()
