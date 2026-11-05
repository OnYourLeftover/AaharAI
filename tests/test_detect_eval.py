import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import detect_eval as de

VOCAB = {"tomato", "onion", "capsicum"}
RAW = {"a.jpg": {"tomato": 0.6, "onion": 0.25}, "b.jpg": {"capsicum": 0.5, "tomato": 0.2}}

def test_case_and_alias_normalisation():
    df = pd.DataFrame({"image": ["a.jpg"], "ingredients": ["Tomatoes; Onion;; radish"]})
    assert de.load_truth(df) == {"a.jpg": {"tomato", "onion", "radish"}}

def test_oov_misses_are_not_blamed_on_the_model():
    truth = {"a.jpg": {"tomato", "radish"}, "b.jpg": {"capsicum"}}
    r = de.evaluate(RAW, truth, VOCAB, thr=0.3)
    assert (r["tp"], r["fp"], r["fn_in_vocab"], r["fn_oov"]) == (2, 0, 0, 1)
    assert r["recall_in_vocab"] == 1.0 and abs(r["recall_all"] - 2 / 3) < 1e-9

def test_threshold_and_relative_threshold():
    assert de.select({"x": 0.5, "y": 0.3}, 0.25) == {"x", "y"}
    assert de.select({"x": 0.5, "y": 0.3}, 0.25, rel=0.7) == {"x"}
    assert de.select({}, 0.1) == set()

def test_sweep_and_report_run():
    truth = {"a.jpg": {"tomato"}, "b.jpg": {"capsicum", "onion"}}
    sw = de.sweep(RAW, truth, VOCAB)
    assert sw.f1_in_vocab.iloc[0] >= sw.f1_in_vocab.iloc[-1]
    rep, inv, miss_in, miss_oov = de.error_report(RAW, truth, VOCAB, 0.3)
    assert len(rep) == 2 and miss_in["onion"] == 1 and not inv

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
