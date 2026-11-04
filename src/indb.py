"""INDB loading, cleaning and table building. Reused by the notebooks and later by the app."""
import re
import pandas as pd

from .config import INDB_DIR, DATA_DIR
from . import vocab

# Approximate unit -> grams (ml treated as g). Good enough for weighting ingredients.
UNIT_G = {"g": 1, "ml": 1, "tsp": 5, "tbsp": 15, "C": 240}
GENERIC_FAT = {"fat", "oil or butter", "oil/butter", "oil/ghee"}


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
    chilli = rec.canonical == "chillies"
    green = org.str.contains("green", na=False)
    rec.loc[chilli & green, "canonical"] = "green chilli"
    rec.loc[chilli & ~green, "canonical"] = "dry red chilli"
    rec.loc[org.isin(GENERIC_FAT), "canonical"] = "oil"       # generic "Fat" must not force butter onto a recipe
    # Only splits that matter for matching: a detected 'milk' / 'bread' / 'ginger' must not satisfy these.
    rec.loc[org.str.contains("buttermilk", na=False), "canonical"] = "buttermilk"
    rec.loc[org.str.contains("condensed milk", na=False), "canonical"] = "condensed milk"
    rec.loc[org.str.contains(r"bread ?crumb", na=False), "canonical"] = "breadcrumbs"
    rec.loc[org.str.contains(r"ginger powder|dry ginger|ground ginger|saunth|sonth", na=False), "canonical"] = "ground ginger"
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
    # INDB.do: serving = recipe total / no_of_servings; unit_serving = serving / size_of_servings.
    # So `kcal` is per ONE serving unit (1 idli, 1 bowl...) and a typical serving is size_of_servings units.
    t["kcal_serving"] = t.kcal * t.size_of_servings
    # INDB sums raw ingredient amounts (all the frying oil, all the syrup), so fried and syrupy dishes come out
    # inflated (e.g. 4,700 kcal for a bowl of korma). Flag them, show a warning, don't silently trust them.
    t["nutrition_flag"] = ((t.kcal_100g > 600) | (t.kcal > 800)).map({True: "check", False: ""})
    return t


def build_tables(min_cov: float = 0.8, save: bool = True):
    """Returns (rec, kept, nutrition) and writes recipes_clean.csv, recipes_kept.csv, nutrition.csv to DATA_DIR."""
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
        rec.to_csv(DATA_DIR / "recipes_clean.csv", index=False)
        kept.to_csv(DATA_DIR / "recipes_kept.csv", index=False)
        nutrition.to_csv(DATA_DIR / "nutrition.csv", index=False)
    return rec, kept, nutrition
