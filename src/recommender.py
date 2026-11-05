"""Recipe recommender: rank INDB recipes by how well the user's ingredients cover what a recipe needs.

Scoring modes (all explainable, no training):
  count   : |have & required| / |required|                      (coverage by ingredient count)
  weight  : grams of required ingredients you have / total grams (coverage by amount)
  jaccard : |have & required| / |have | required|                (also rewards using what you have)
  smooth  : |have & required| / (|required| + alpha)             (coverage shrunk so tiny recipes cannot win)
Ranking: highest score, then fewest missing ingredients, then fewest required ingredients.
Default is "jaccard": in the hide test (see hide_test) coverage-first ranking collapses once the pantry holds
unrelated items, because tiny recipes become "fully covered". Count and weight stay as ablation baselines.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from . import vocab

DEFAULT_GRAMS = 5.0          # amount used when INDB gives a unit we cannot convert (sprig, nos, ...)


@dataclass
class Hit:
    recipe_code: str
    name: str
    score: float
    matched: list = field(default_factory=list)
    missing: list = field(default_factory=list)


class Recommender:
    def __init__(self, rec: pd.DataFrame, kept: pd.DataFrame, nutrition: pd.DataFrame):
        keep_codes = set(kept.recipe_code)
        rows = rec[rec.recipe_code.isin(keep_codes)]
        self.names = dict(zip(nutrition.recipe_code, nutrition.recipe_name))
        self.nutrition = nutrition.set_index("recipe_code")
        self.core, self.staples = {}, {}
        for code, g in rows.groupby("recipe_code"):
            core = g[g.cat.isin(["detectable", "manual_add", "other"])]
            self.core[code] = core.groupby("canonical").grams.sum(min_count=1).fillna(DEFAULT_GRAMS).to_dict()
            st = g[(g.cat == "staple") & (g.canonical != "water")]
            self.staples[code] = set(st.canonical)
        self.nonveg = {c: any(vocab.is_nonveg(i) for i in ing) for c, ing in self.core.items()}
        self.alpha = 3.0
        self.freq = Counter(i for ing in self.core.values() for i in ing)   # how common each ingredient is

    def recommend(self, have, k=5, mode="jaccard", staples_available=True, veg_only=False,
                  max_kcal=None, min_match=2):
        have = set(have)
        out = []
        for code, ing in self.core.items():
            if veg_only and self.nonveg[code]:
                continue
            req = dict(ing)
            if not staples_available:                       # user says the pantry is bare
                req.update({s: DEFAULT_GRAMS for s in self.staples[code]})
            matched = [i for i in req if i in have]
            if len(matched) < min_match:
                continue
            missing = [i for i in req if i not in have]
            if mode == "count":
                score = len(matched) / len(req)
            elif mode == "weight":
                score = sum(req[i] for i in matched) / sum(req.values())
            elif mode == "jaccard":
                score = len(matched) / len(have | set(req))
            elif mode == "smooth":
                score = len(matched) / (len(req) + self.alpha)
            else:
                raise ValueError(mode)
            if max_kcal is not None:
                kcal = self.nutrition.kcal.get(code)
                if pd.notna(kcal) and kcal > max_kcal:
                    continue
            out.append((-score, len(missing), len(req), code, matched, missing))
        out.sort(key=lambda t: t[:3])
        return [Hit(c, self.names.get(c, c), -s, m, x) for s, _, _, c, m, x in out[:k]]


def hide_test(rec: Recommender, hide_fracs=(0.2, 0.3, 0.5), modes=("count", "weight", "jaccard"),
              min_core=5, seed=42, ks=(1, 5, 10), n_extra=0):
    """Hide a fraction of each recipe's ingredients, treat the rest as the pantry, and check whether the
    source recipe comes back in the top k. Reproducible (fixed seed); only recipes with >= min_core ingredients.
    n_extra adds that many unrelated 'distractor' ingredients to the pantry (sampled by how common they are),
    because a real fridge holds items the target recipe does not use."""
    rng = random.Random(seed)
    pool, weights = zip(*rec.freq.items())
    codes = sorted(c for c, ing in rec.core.items() if len(ing) >= min_core)
    rows = []
    for h in hide_fracs:
        cases = []
        for c in codes:
            ings = sorted(rec.core[c])
            rng.shuffle(ings)
            pantry = ings[: max(2, round(len(ings) * (1 - h)))]
            extra = set()
            while len(extra) < n_extra:
                x = rng.choices(pool, weights)[0]
                if x not in rec.core[c]:
                    extra.add(x)
            cases.append((c, pantry + sorted(extra)))
        for mode in modes:
            hits = {k: 0 for k in ks}
            for code, pantry in cases:
                ranked = [x.recipe_code for x in rec.recommend(pantry, k=max(ks), mode=mode)]
                for k in ks:
                    hits[k] += code in ranked[:k]
            rows.append({"hidden": h, "extra": n_extra, "mode": mode, "n": len(cases), **{f"hit@{k}": hits[k] / len(cases) for k in ks}})
    return pd.DataFrame(rows)
