import re
from numpy import generic
import pandas as pd
from .config import INDB_DIR, DATA_DIR
from . import vocab

UNIT_G = {"g": 1, "ml": 1, "tsp": 5, "tbsp": 15, "C": 240}   # approximate; ml treated as g

def load_raw():
    rec = pd.read_excel(INDB_DIR / "recipes.xlsx")
    nut = pd.read_excel(INDB_DIR / "INDB.xlsx", sheet_name="Nutrient Data")
    srv = pd.read_excel(INDB_DIR / "recipes_servingsize.xlsx")
    return rec, nut, srv

def _clean(name):
    return re.sub(r"\s*\(.*?\)", "", str(name).split(",")[0]).strip().lower()

def add_canonical(rec):
    rec = rec.copy()
    rec["canonical"] = rec.food_name.apply(_clean).replace(vocab.RENAME)
    chilli = rec.canonical == "chillies"
    green = rec.ingredient_name_org.str.contains("green", case=False, na=False)
    rec.loc[chilli & green, "canonical"] = "green chilli"
    rec.loc[chilli & ~green, "canonical"] = "dry red chilli"

    generic = rec.ingredient_name_org.astype(str).str.strip().str.lower().isin(
        ["fat", "oil or butter", "oil/butter", "oil/ghee"]
    )
    rec.loc[generic, "canonical"] = "oil"
    rec["cat"] = rec.canonical.map(vocab.category)
    rec["grams"] = pd.to_numeric(rec.amount, errors="coerce") * rec.unit.map(UNIT_G)
    return rec

def nutrition_table(nut, srv):
    cols = {"food_code": "recipe_code", "food_name": "recipe_name", "servings_unit": "serving_unit",
            "unit_serving_energy_kcal": "kcal", "unit_serving_protein_g": "protein_g",
            "unit_serving_carb_g": "carb_g", "unit_serving_fat_g": "fat_g", "energy_kcal": "kcal_100g"}
    t = nut[list(cols)].rename(columns=cols)
    return t.merge(srv[["recipe_code", "no_of_servings", "size_of_servings"]], on="recipe_code")

def build_tables(min_cov=0.8, save=True):
    rec, nut, srv = load_raw()
    rec = add_canonical(rec)
    core = rec[rec.cat.isin(["detectable", "manual_add", "other"])]
    per = core.groupby("recipe_code").agg(
        n_core=("cat", "size"),
        n_vocab=("cat", lambda s: s.isin(["detectable", "manual_add"]).sum()))
    per["vocab_cov"] = per.n_vocab / per.n_core
    ok = set(srv.dropna(subset=["no_of_servings", "size_of_servings"]).recipe_code)
    kept = per[(per.vocab_cov >= min_cov) & per.index.isin(ok)].reset_index()
    nutrition = nutrition_table(nut, srv)
    if save:
        rec.to_csv(DATA_DIR / "recipes_clean.csv", index=False)
        kept.to_csv(DATA_DIR / "recipes_kept.csv", index=False)
        nutrition.to_csv(DATA_DIR / "nutrition.csv", index=False)
    return rec, kept, nutrition