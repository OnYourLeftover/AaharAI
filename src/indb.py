"""INDB loading, cleaning and table building. Reused by the notebooks and later by the app."""
import re
import pandas as pd

from .config import INDB_DIR, DATA_DIR
from . import vocab

# Approximate unit -> grams (ml treated as g). Units not listed (sprig, nos, pinch...) give NaN grams.
UNIT_G = {"g": 1, "ml": 1, "tsp": 5, "tbsp": 15, "C": 240}

# Ordered refinement rules on the ORIGINAL ingredient text: (pattern, canonical, exclude_pattern).
# Later rules override earlier ones. Patterns are matched against the lower-cased, stripped original name.
# Edit rules HERE; edit categories in vocab.py.
RULES = [
    # chillies are handled separately (they depend on the INDB name too), see add_canonical
    # bread
    (r"crumb",                                        "breadcrumbs",       None),
    (r"brown bread",                                  "brown bread",       None),
    (r"white bread|bread slice|fresh bread",          "white bread",       r"crumb"),
    # cheese
    (r"cheddar",                                      "cheddar cheese",    None),
    (r"cheese",                                       "processed cheese",  r"cheddar|paneer|cottage"),
    (r"cottage cheese|paneer",                        "paneer",            None),
    # chicken
    (r"chicken",                                      "chicken",           r"bones and|trimming|stock"),
    (r"^(chicken )?bones and|trimming",               "chicken bones",     None),     # not "boneless" / "with bones"
    # coconut
    (r"des?s?iccated|dry coconut|coconut powder",     "dried coconut",     None),
    (r"fresh coconut",                                "fresh coconut",     None),
    # cream
    (r"whipp",                                        "whipping cream",    None),
    (r"cream",                                        "fresh cream",       r"whipp|ice cream|tartar|sour|cheese|coconut|creamed|milk"),
    (r"cream of tartar",                              "cream of tartar",   None),
    (r"sour cream",                                   "sour cream",        None),
    # eggs
    (r"egg yolk",                                     "egg yolk",          None),
    (r"egg white",                                    "egg white",         None),
    (r"^eggs?$",                                      "whole egg",         None),
    # flours
    (r"cornflour|corn flour|corn starch",             "cornflour",         None),
    (r"besan|gram flour|chickpea flour",              "gram flour",        None),
    (r"rice flour",                                   "rice flour",        None),
    (r"whole wheat flour|(?<!maize )\batta\b",        "whole wheat flour", None),
    (r"refined (wheat )?flour|maida|all purpose flour", "refined flour",   None),
    # ginger
    (r"ginger powder|dry ginger|saunth|ground ginger|ginger dry powder", "ground ginger", None),
    (r"ginger\s*(&|and)?\s*garlic paste",             "ginger garlic paste", None),
    (r"fresh ginger|ginger paste|grated ginger|shredded ginger", "fresh ginger", r"garlic"),
    # lemon
    (r"lemon juice|lime juice",                       "lemon juice",       None),
    (r"lemon rind",                                   "lemon rind",        None),
    (r"^lemon$|lemon slice",                          "whole lemon",       None),
    # milk
    (r"condensed milk",                               "condensed milk",    None),
    (r"buttermilk",                                   "buttermilk",        None),
    (r"^milk$|whole milk|cow's milk|hot milk|full cream milk|full fat milk", "whole milk", None),
    # wheat products
    (r"vermicelli",                                   "vermicelli",        None),
    (r"semolina|suji|\brava\b",                       "semolina",          r"rice rava"),
    # stock is a liquid base, not the meat/vegetable it is named after (runs last so it wins)
    (r"\bstock\b",                                    "stock",             r"cube"),
    # generic fat must not force butter onto a recipe
    (r"^(fat|oil or butter|oil/butter|oil/ghee)$",    "oil",               None),
]
_COMPILED = [(re.compile(p), c, re.compile(x) if x else None) for p, c, x in RULES]


def load_raw():
    rec = pd.read_excel(INDB_DIR / "recipes.xlsx")
    nut = pd.read_excel(INDB_DIR / "INDB.xlsx", sheet_name="Nutrient Data")
    srv = pd.read_excel(INDB_DIR / "recipes_servingsize.xlsx")
    return rec, nut, srv


def _clean(name) -> str:
    # Remove bracketed Latin names FIRST (they can contain commas), then keep the text before the first comma.
    return re.sub(r"\s*\(.*?\)", "", str(name)).split(",")[0].strip().lower()


def add_canonical(rec: pd.DataFrame) -> pd.DataFrame:
    rec = rec.copy()
    rec["canonical"] = rec.food_name.apply(_clean).replace(vocab.RENAME)
    org = rec.ingredient_name_org.astype(str).str.strip().str.lower()

    # chillies: green vs dry red depends on the original text
    chilli = rec.canonical == "chillies"
    green = org.str.contains("green", na=False)
    rec.loc[chilli & green, "canonical"] = "green chilli"
    rec.loc[chilli & ~green, "canonical"] = "dry red chilli"

    for pat, canon, excl in _COMPILED:
        m = org.map(lambda s: pat.search(s) is not None)
        if excl is not None:
            m &= ~org.map(lambda s: excl.search(s) is not None)
        rec.loc[m, "canonical"] = canon

    rec["cat"] = rec.canonical.map(vocab.category)
    rec["grams"] = pd.to_numeric(rec.amount, errors="coerce") * rec.unit.map(UNIT_G)
    return rec


def nutrition_table(nut: pd.DataFrame, srv: pd.DataFrame) -> pd.DataFrame:
    cols = {"food_code": "recipe_code", "food_name": "recipe_name", "servings_unit": "serving_unit",
            "unit_serving_energy_kcal": "kcal", "unit_serving_protein_g": "protein_g",
            "unit_serving_carb_g": "carb_g", "unit_serving_fat_g": "fat_g", "energy_kcal": "kcal_100g"}
    t = nut[list(cols)].rename(columns=cols)
    s = srv[["recipe_code", "no_of_servings", "size_of_servings", "remarks_1"]].rename(
        columns={"remarks_1": "serving_note"})
    t = t.merge(s, on="recipe_code", how="left")
    t["serving_guessed"] = t.serving_note.astype(str).str.contains("guessed", case=False)
    return t


def build_tables(min_cov: float = 0.8, save: bool = True):
    """Returns (rec, kept, nutrition); writes recipes_clean.csv, recipes_kept.csv, nutrition.csv to DATA_DIR."""
    rec, nut, srv = load_raw()
    rec = add_canonical(rec)
    core = rec[rec.cat.isin(["detectable", "manual_add", "other"])]
    per = core.groupby("recipe_code").agg(
        n_core=("cat", "size"),
        n_vocab=("cat", lambda s: s.isin(["detectable", "manual_add"]).sum()))
    per["vocab_cov"] = per.n_vocab / per.n_core
    has_serving = set(srv.dropna(subset=["no_of_servings", "size_of_servings"]).recipe_code)
    kept = per[(per.vocab_cov >= min_cov) & per.index.isin(has_serving)].reset_index()
    nutrition = nutrition_table(nut, srv)
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rec.to_csv(DATA_DIR / "recipes_clean.csv", index=False)
        kept.to_csv(DATA_DIR / "recipes_kept.csv", index=False)
        nutrition.to_csv(DATA_DIR / "nutrition.csv", index=False)
    return rec, kept, nutrition