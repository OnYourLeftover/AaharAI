"""Pure-Python evaluation helpers for the OWLv2 ingredient test (no torch needed, so they are unit-testable).

raw   : {photo: {label: best_score}}   detections at a low threshold, after NMS
truth : {photo: {labels}}              what you wrote in labels.csv
vocab : set of labels the detector is asked about (vocab.DETECT)

Ground-truth items outside `vocab` can never be detected, so they are reported separately
(out-of-vocabulary misses) instead of being blamed on the model.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

ALIASES = {"brinjal": "brinjal - all varieties", "eggplant": "brinjal - all varieties", "aubergine": "brinjal - all varieties",
           "spring onion": "spring onions", "mushroom": "button mushroom", "mushrooms": "button mushroom",
           "chilli": "green chilli", "green chillies": "green chilli", "coriander": "coriander leaves",
           "cilantro": "coriander leaves", "beans": "french beans", "green beans": "french beans",
           "tomatoes": "tomato", "potatoes": "potato", "onions": "onion", "carrots": "carrot", "eggs": "egg",
           "celery": "celery stalk", "mint": "mint leaves", "curd": "yogurt", "curry leaf": "curry leaves",
           "bell pepper": "capsicum", "green onion": "spring onions", "green onions": "spring onions",
           "scallion": "spring onions", "scallions": "spring onions", "lemons": "lemon", "bananas": "banana", "oranges": "orange"}


def clean_label(s) -> str:
    s = " ".join(str(s).lower().split())
    return ALIASES.get(s, s)


def load_truth(df: pd.DataFrame) -> dict:
    truth = {}
    for r in df.itertuples():
        items = {clean_label(x) for x in str(r.ingredients).split(";") if x.strip() and x.strip().lower() != "nan"}
        if items:
            truth[r.image] = items
    return truth


def select(scores: dict, thr: float, rel: float = 0.0) -> set:
    """Labels whose score is >= thr and >= rel * (best score in that photo)."""
    if not scores:
        return set()
    top = max(scores.values())
    return {k for k, v in scores.items() if v >= thr and v >= rel * top}


def evaluate(raw: dict, truth: dict, vocab: set, thr: float, rel: float = 0.0) -> dict:
    tp = fp = fn_in = fn_oov = 0
    for img, t in truth.items():
        if img not in raw:
            continue
        pred = select(raw[img], thr, rel)
        miss = t - pred
        tp += len(pred & t); fp += len(pred - t)
        fn_oov += len(miss - vocab); fn_in += len(miss & vocab)
    P = tp / max(tp + fp, 1)
    R_in = tp / max(tp + fn_in, 1)                     # recall over ingredients the detector is asked about
    R_all = tp / max(tp + fn_in + fn_oov, 1)           # recall over everything you labelled
    f = lambda p, r: 2 * p * r / max(p + r, 1e-9)
    return {"thr": thr, "rel": rel, "precision": P, "recall_in_vocab": R_in, "recall_all": R_all,
            "f1_in_vocab": f(P, R_in), "f1_all": f(P, R_all), "tp": tp, "fp": fp, "fn_in_vocab": fn_in, "fn_oov": fn_oov}


def sweep(raw, truth, vocab, thrs=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6), rels=(0.0, 0.4, 0.6)) -> pd.DataFrame:
    rows = [evaluate(raw, truth, vocab, t, r) for t in thrs for r in rels]
    return pd.DataFrame(rows).sort_values("f1_in_vocab", ascending=False).reset_index(drop=True)


def error_report(raw, truth, vocab, thr, rel=0.0):
    """Per-photo table plus the most invented / most missed labels at one setting."""
    rows, invented, missed_in, missed_oov = [], Counter(), Counter(), Counter()
    for img, t in truth.items():
        if img not in raw:
            continue
        pred = select(raw[img], thr, rel)
        fp, miss = pred - t, t - pred
        invented.update(fp); missed_in.update(miss & vocab); missed_oov.update(miss - vocab)
        rows.append({"image": img, "truth": ";".join(sorted(t)), "predicted": ";".join(sorted(pred)),
                     "false_pos": ";".join(sorted(fp)), "missed_in_vocab": ";".join(sorted(miss & vocab)),
                     "missed_not_in_vocab": ";".join(sorted(miss - vocab))})
    return pd.DataFrame(rows), invented, missed_in, missed_oov


def two_tier(raw: dict, truth: dict, vocab: set, hi: float = 0.4, lo: float = 0.25) -> dict:
    """UI design check: detections >= hi are pre-selected ('confident'); those in [lo, hi) are listed as unselected 'maybe'.
    candidate_recall = share of ingredients (in vocabulary) that appear anywhere in either list.
    extra_maybe_per_photo = how many extra unselected suggestions the user has to look at."""
    tp = fp = total_in = cand_hit = maybe = n = 0
    for img, t in truth.items():
        if img not in raw:
            continue
        conf, cand = select(raw[img], hi), select(raw[img], lo)
        t_in = t & vocab
        tp += len(conf & t); fp += len(conf - t)
        total_in += len(t_in); cand_hit += len(cand & t_in)
        maybe += len(cand - conf); n += 1
    return {"hi": hi, "lo": lo, "confident_precision": tp / max(tp + fp, 1),
            "confident_recall": tp / max(total_in, 1), "candidate_recall": cand_hit / max(total_in, 1),
            "extra_maybe_per_photo": maybe / max(n, 1)}


def missed_scores(raw: dict, truth: dict, vocab: set, thr: float) -> pd.DataFrame:
    """For every in-vocabulary ingredient that was missed at `thr`, the score the detector actually gave it (0 = not seen)."""
    rows = []
    for img, t in truth.items():
        if img not in raw:
            continue
        for lab in (t - select(raw[img], thr)) & vocab:
            rows.append({"image": img, "label": lab, "score": raw[img].get(lab, 0.0)})
    return pd.DataFrame(rows, columns=["image", "label", "score"])


def confusions(rep: pd.DataFrame) -> Counter:
    """(false positive, missed) pairs that occur together in the same photo: a rough picture of what is confused with what."""
    c = Counter()
    for r in rep.itertuples():
        fps = [x for x in str(r.false_pos).split(";") if x]
        ms = [x for x in str(r.missed_in_vocab).split(";") if x]
        for f in fps:
            for m in ms:
                c[(f, m)] += 1
    return c
