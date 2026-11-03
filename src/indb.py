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

    # Initial standardisation using the INDB food names
    rec["canonical"] = rec.food_name.apply(_clean).replace(vocab.RENAME)

    # Preserve meaningful ingredient distinctions using original names
    org = rec.ingredient_name_org.astype(str).str.strip().str.lower()

    # Chillies
    chilli = rec.canonical == "chillies"
    green = org.str.contains("green", na=False)
    rec.loc[chilli & green, "canonical"] = "green chilli"
    rec.loc[chilli & ~green, "canonical"] = "dry red chilli"

    # Bread
    rec.loc[org.str.contains(r"crumb|breadcrumb", na=False), "canonical"] = "breadcrumbs"
    rec.loc[org.str.contains(r"brown bread", na=False), "canonical"] = "brown bread"
    rec.loc[org.str.contains(r"white bread|bread slice|fresh bread", na=False), "canonical"] = "white bread"

    # Cheese
    rec.loc[org.str.contains("cheddar", na=False), "canonical"] = "cheddar cheese"
    rec.loc[org.str.contains("paneer", na=False), "canonical"] = "paneer"
    rec.loc[org.str.contains("cheese", na=False) & ~org.str.contains("cheddar|paneer", na=False),
            "canonical"] = "processed cheese"

    # Chicken
    rec.loc[org.str.contains(r"bone|trimming", na=False), "canonical"] = "chicken bones"
    rec.loc[org.str.contains("chicken", na=False) & ~org.str.contains(r"bone|trimming", na=False),
            "canonical"] = "chicken"

    # Coconut
    rec.loc[org.str.contains(r"des?iccated|dry coconut|coconut powder", na=False),
            "canonical"] = "dried coconut"
    rec.loc[org.str.contains(r"fresh coconut", na=False), "canonical"] = "fresh coconut"

    # Cream
    rec.loc[org.str.contains(r"whipp|whipped", na=False), "canonical"] = "whipping cream"
    rec.loc[org.str.contains("cream", na=False) & ~org.str.contains(r"whipp|whipped", na=False),
            "canonical"] = "fresh cream"

    # Eggs
    rec.loc[org.str.contains("egg yolk", na=False), "canonical"] = "egg yolk"
    rec.loc[org.str.contains("egg white", na=False), "canonical"] = "egg white"
    rec.loc[org.str.fullmatch(r"eggs?", na=False), "canonical"] = "whole egg"

    # Flour
    rec.loc[org.str.contains(r"cornflour|corn flour|corn starch", na=False),
            "canonical"] = "cornflour"
    rec.loc[org.str.contains(r"besan|bengal gram flour|gram flour|chickpea flour", na=False),
            "canonical"] = "gram flour"
    rec.loc[org.str.contains("rice flour", na=False), "canonical"] = "rice flour"

    # Ginger
    rec.loc[org.str.contains(r"ginger powder|dry ginger|saunth|ground ginger", na=False),
            "canonical"] = "ground ginger"
    rec.loc[org.str.contains(r"ginger garlic paste", na=False), "canonical"] = "ginger garlic paste"
    rec.loc[org.str.contains(r"fresh ginger|ginger paste|grated ginger|shredded ginger", na=False),
            "canonical"] = "fresh ginger"

    # Lemon
    rec.loc[org.str.contains(r"lemon juice|lime juice", na=False), "canonical"] = "lemon juice"
    rec.loc[org.str.contains(r"lemon rind", na=False), "canonical"] = "lemon rind"
    rec.loc[org.str.contains(r"^lemon$|lemon slice", na=False), "canonical"] = "whole lemon"

    # Milk
    rec.loc[org.str.contains("condensed milk", na=False), "canonical"] = "condensed milk"
    rec.loc[org.str.contains("buttermilk", na=False), "canonical"] = "buttermilk"
    rec.loc[org.str.contains(r"^milk$|whole milk|cow's milk|hot milk|full cream milk", na=False),
            "canonical"] = "whole milk"

    # Wheat
    rec.loc[org.str.contains(r"vermicelli", na=False), "canonical"] = "vermicelli"
    rec.loc[org.str.contains(r"semolina|suji|rava", na=False), "canonical"] = "semolina"

    # Wheat flour
    rec.loc[org.str.contains(r"whole wheat flour|atta", na=False), "canonical"] = "whole wheat flour"
    rec.loc[org.str.contains(r"refined flour|refined wheat flour|maida|all purpose flour", na=False),
            "canonical"] = "refined flour"

    # Butter/oil ambiguity: do not silently classify as butter
    rec.loc[org.isin(["fat", "oil or butter", "oil/butter", "oil/ghee"]), "canonical"] = "oil"

    # Reassign categories after all canonical corrections
    rec["cat"] = rec.canonical.map(vocab.category)

    # Convert ingredient quantities to grams using the existing unit mapping
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