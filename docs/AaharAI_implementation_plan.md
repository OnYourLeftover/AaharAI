# AaharAI

**Vision-Based Indian Recipe Recommendation and Nutrition Estimation System: Implementation Plan (Final)**

---

## 1. Overview

**AaharAI** ("Aahar" is Hindi for food) is a single system with two vision entry points feeding one shared food database:

- **Mode A: Dish recognition.** The user uploads a photo of a prepared dish. The system predicts the Indian dish and returns its reference nutrition per serving.
- **Mode B: Ingredient recognition.** The user uploads a photo of available ingredients (for example a fridge). The system detects the ingredients, lets the user edit the list, recommends Indian recipes and shows nutrition per serving for each.

Both modes use the same canonical ingredient vocabulary and the same nutrition database, so the project is one coherent system, not separate models side by side.

**Goals**

1. Recognise Indian dishes from photos (EfficientNet-B0 on Khana).
2. Detect ingredients from photos with an open-vocabulary detector (OWLv2).
3. Recommend recipes from available ingredients with an explainable ranking algorithm.
4. Report nutrition per serving from a verified database, clearly labelled as an estimate.
5. Evaluate each design decision, not only the final accuracy.

---

## 2. System Architecture

```
                      User uploads image
                              |
          +-------------------+-------------------+
          |                                       |
   MODE A: Dish photo                    MODE B: Ingredient photo
          |                                       |
   EfficientNet-B0 classifier             OWLv2 text-conditioned detector
   (trained on Khana)                     (60-80 ingredient vocabulary)
          |                                       |
   Predicted dish + confidence            Boxes -> NMS -> canonical ingredients
          |                                       |
   Verified Khana -> INDB mapping         EDITABLE ingredient list (user fixes errors)
          |                                       |
          |                              Recipe recommender (count vs weight coverage)
          |                                       |
          +-------------------+-------------------+
                              |
                  INDB shared food database
     Recipes | Ingredients | Quantities | Nutrition | Serving sizes
                              |
                      Results interface
     Dish / recipes | Calories | Protein | Carbs | Fat | confidence
```

---

## 3. Technology Stack (frozen)

| Module | Choice |
|---|---|
| Language | Python |
| Deep learning | PyTorch, `timm`, Hugging Face Transformers |
| Dish classifier | EfficientNet-B0 (pretrained, fine-tuned) |
| Dish dataset | Khana |
| Ingredient detection | OWLv2 (`google/owlv2-base-patch16-ensemble`) |
| Ingredient vocabulary | 60-80 canonical ingredients derived from INDB |
| Recipe + nutrition database | Indian Nutrient Databank (INDB) |
| Recommendation | Custom coverage-based ranking (count and weight variants) |
| Fuzzy matching | RapidFuzz (with manual verification) |
| Explainability | Grad-CAM (optional) |
| Interface | Gradio |
| Training | Google Colab / Kaggle |
| Demo hosting | Local / Colab (guaranteed), Hugging Face ZeroGPU (optional) |
| Version control | Git + GitHub |

**Not used in version 1:** YOLO, SAM-based portion estimation, additional architectures, LLM recipe generation, React frontend, a second recipe dataset.

---

## 4. Data Sources

### 4.1 Khana (dish images)
- About 131K images across 80 Indian dish classes, 500x500 pixels.
- Paper: arXiv 2509.06006. Dataset page: `khana.omkar.xyz`.
- Check the licence and download size on the dataset page before downloading. It is intended for research and educational use, and the image copyrights are not owned by the dataset creators.
- Use a stratified subset (about 300 images per class, roughly 24k images) for fast training.

### 4.2 INDB: Indian Nutrient Databank (recipes and nutrition)
- 1,014 recipes and 1,095 raw food items.
- Recipes include ingredient amounts, total recipe weight, number of servings and serving sizes.
- Public repository: `github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB`.
- Recipe nutrients are computed by summing ingredient values, and most ingredients have no cooking-retention adjustment. Values are therefore **raw-ingredient estimates**, not measured cooked values.
- Serving size is missing for the "pickles and preserves" and "weaning foods" categories. Exclude those recipes from anything that needs per-serving nutrition.
- Quantities are not all in grams (they include ml, teaspoons, tablespoons, sprigs and items). Convert to grams with ingredient-specific conversions before any weight-based calculation.
- **Licence caution:** no clear data redistribution licence was found. Keep the data local/private. Do not commit it to a public repository or package it into a public Space until permission is confirmed with the authors.

---

## 5. Shared Data Layer

### 5.1 Ingredient vocabulary (`ingredients.csv`)

| Column | Purpose |
|---|---|
| `ingredient_id` | Stable identifier used by every module |
| `canonical_name` | e.g. `tomato` |
| `aliases` | e.g. `tomatoes, tamatar` |
| `owl_prompt` | Text query for OWLv2, e.g. `a photo of a tomato` |
| `indb_food_code` | Link to INDB raw food item |
| `category` | `detectable`, `staple`, or `ignore` |

**How to build it**
1. Extract all ingredients from the INDB recipe file.
2. Normalise names and aliases.
3. Mark pantry staples (salt, oil, turmeric powder, chilli powder, cumin seeds, and similar).
4. Count the frequency of the remaining ingredients.
5. Keep the top 60-80 that are both frequent **and** visually detectable. Frequency alone is not enough.
6. Keep only recipes whose non-staple ingredients are mostly covered by the vocabulary.

Staples are configurable in the UI. A user who has run out of oil should be able to turn that assumption off.

### 5.2 Dish-to-nutrition mapping (`mapping.csv`)

| khana_label | indb_recipe_id | status | notes |
|---|---|---|---|
| Rajma | (id) | verified | |
| Unmatched dish | | none | no nutrition shown |

- Use RapidFuzz to propose the top-3 candidates per Khana label.
- **Verify every match by hand.** A similar name does not prove it is the same preparation.
- Never invent nutrition values for unmatched dishes. Show "nutrition unavailable".

```python
from rapidfuzz import process, fuzz
cand = process.extract(khana_label, indb_names, scorer=fuzz.token_set_ratio, limit=3)
```

### 5.3 Recipe table (derived from INDB)
`recipe_id, name, ingredients{ingredient_id: grams}, servings, serving_size_g, kcal, protein, carbs, fat (per serving)`

Vegetarian/non-vegetarian is **derived** from the ingredient list (meat, fish, egg keywords), because INDB does not provide a diet field. INDB also has no cooking-time field, so do not claim time-based ranking.

---

## 6. Module Specifications

### 6.1 Mode A: Dish classifier
- **Model:** EfficientNet-B0, ImageNet-pretrained, final layer replaced for the number of mapped classes.
- **Split:** stratified 70 / 15 / 15 (train / validation / test). Remove duplicate images first so nothing leaks across splits.
- **Training:** AdamW (lr about 3e-4), label smoothing 0.1, 10-15 epochs, early stopping on validation loss. Augmentation (RandomResizedCrop, flip, light ColorJitter) on the training set only.
- **Output:** top-5 predictions with confidences, then nutrition from `mapping.csv` (or "unavailable").
- **Optional:** Grad-CAM heatmap, only after the classifier works.

```python
import timm
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
```

### 6.2 Mode B: Ingredient detector
Pipeline: image -> OWLv2 -> boxes and scores -> top label per box -> confidence threshold -> class-agnostic NMS -> canonical ingredient set -> **editable list**.

```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms

proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

inputs = proc(text=[[f"a photo of {x}" for x in VOCAB]], images=img, return_tensors="pt")
outputs = model(**inputs)

# OWLv2 pads images to a square, so use (max(h, w), max(h, w)) as target_sizes.
# results = proc.post_process_object_detection(outputs, threshold=T, target_sizes=...)
# keep = nms(boxes, scores, iou_threshold=IOU)   # tune IOU on labelled photos
# ingredients = {VOCAB[l] for l in labels[keep]}
```

Notes
- Zero-shot does not mean reliable. Expect trouble with chopped herbs, items in transparent containers, occlusion and clutter.
- Tune the **score threshold** and the **NMS IoU** on labelled photos. Lowering the threshold increases false positives. A high IoU keeps genuinely different overlapping ingredients, and a low IoU may delete them.
- The editable list and a manual "add ingredient" box are mandatory features.

### 6.3 Recipe recommender
Candidate recipes come from the filtered INDB table. Staples and water are excluded from scoring (staples only while the staples toggle is on).

**Count coverage**

```
C_count = |Have ∩ Required| / |Required|
```

**Weight coverage** (ablation)

```python
def wcov(have, recipe):   # recipe.core = {ingredient_id: grams}, staples and water removed
    tot = sum(recipe.core.values())
    return sum(g for i, g in recipe.core.items() if i in have) / tot
```

**Ranking:** highest coverage, then fewest missing ingredients, then fewest non-staple ingredients.

```python
ranked = sorted(cands, key=lambda r: (-r.coverage, len(r.missing), len(r.core)))[:5]
```

**Filters:** vegetarian / non-vegetarian, calorie cap, staples toggle.
**Explanation shown to the user:** which ingredients they have and which are missing for each recipe.

Weight-based scoring is not assumed to be better. A missing key ingredient such as paneer can make a recipe impossible even when most of the weight is available. Compare both methods and report the result.

### 6.4 Nutrition
- **Mode A:** retrieve the verified INDB record for the predicted dish. Per-serving values are scaled by a user-selected portion (S / M / L = 0.75 / 1 / 1.5).
- **Mode B:** retrieve per-serving nutrition for each recommended INDB recipe.
- Every value is labelled **approximate**. Missing records show "nutrition unavailable".
- This is a database lookup, **not** a trained nutrition prediction model. Do not present it as one.

### 6.5 Interface (Gradio)
- App title: "AaharAI", with one tab per mode.
- Tab 1: Dish scan. Output: top-5 dishes with confidence, nutrition card.
- Tab 2: Ingredient scan. Output: editable ingredient chips, staples toggle, filters, ranked recipes with missing-ingredient lists and nutrition.
- Build a stub version of both tabs on Day 2 and plug each real module in as soon as it works.

---

## 7. 12-Day Schedule

| Day | Work | Deliverable |
|---|---|---|
| 1 | Data audit and feasibility gates (section 8): inspect Khana and INDB, build vocabulary, build and verify mapping, test OWLv2 on 10-15 real photos | Go / no-go decision, clean data foundation |
| 2 | Gradio skeleton with two tabs, shared schema, stub predictors | App runs end to end with placeholder outputs |
| 3-5 | Dish classifier: subset sampling, dedup, splits, training, early stopping, confusion matrix, top-5; connect to verified nutrition | Working Mode A |
| 6-7 | Ingredient detection: OWLv2 integration, threshold and NMS tuning, canonical mapping, editable chips | Working ingredient list from photos |
| 8-9 | Recipe recommender (count and weight), missing-ingredient explanations, diet and calorie filters, nutrition per serving | Ranked recipes with nutrition |
| 10 | Full integration, confidence and "approximate" labels, failure-case handling (no detection, no nutrition match) | Complete working demo |
| 11 | Evaluation (section 9), tables and graphs | Results for the report |
| 12 | Demo rehearsal, documentation, architecture diagram, viva preparation | Final deliverable |

**Rule:** do not start training until the Day 1 gates pass.

---

## 8. Day 1 Go / No-Go Gates

| Gate | Pass condition (project targets, adjust if needed) |
|---|---|
| Data joins | Vocabulary, INDB recipes and mapping CSV load into one schema |
| Mapping coverage | At least about 60 of the 80 Khana classes have a **manually verified** INDB match. If fewer, train on the matched classes only |
| Detector feasibility | OWLv2 finds most common vegetables in 10-15 real kitchen photos. If not, tighten the vocabulary and retune before building further |
| Class counts | Every Khana class has enough images for a 70/15/15 split |
| INDB units | Unit-conversion data is available for converting quantities to grams |

---

## 9. Evaluation Plan

| Module | Method | Metrics |
|---|---|---|
| Dish classifier | Held-out test split **plus** about 30 independent phone photos (different lighting and devices) | Top-1, top-5, macro-F1, confusion matrix, most confused pairs |
| Ingredient detection | 30-50 manually labelled images (simple, cluttered, occluded). Split into a dev set (tune threshold and IoU) and a test set (report) | Image-level multilabel precision, recall, F1 |
| Recommender (synthetic) | Hide 20 / 30 / 50% of **non-staple** ingredients of a recipe, rank, check if the source recipe is in the top 5. Fixed random seed. Use only recipes with enough non-staple ingredients | Hit@5, count vs weight |
| Recommender (manual) | 10-15 hand-made pantry scenarios, inspect whether suggestions are sensible | Qualitative table |
| Nutrition | Lookup checks | Lookup coverage, serving-scaling correctness, missing-record handling |

```python
import random
random.seed(42)

def make_case(recipe, hide=0.3):
    ings = list(recipe.core_ings)
    random.shuffle(ings)
    keep = ings[:max(2, int(len(ings) * (1 - hide)))]
    return keep, recipe.id        # hit@5 = recipe.id in top-5 for `keep`
```

Result table to fill in:

| Hidden ingredients | Count-based Hit@5 | Weight-based Hit@5 |
|---|---|---|
| 20% | | |
| 30% | | |
| 50% | | |

Hit@5 measures recovery of the source recipe, not whether every recommendation is suitable. Do not report an MAE of nutrition against INDB itself, since that would only measure a database lookup.

---

## 10. Deployment and Demo

- **Guaranteed demo:** run Gradio locally or in Colab (`share=True` gives a temporary public link).
- **Optional public demo:** a Hugging Face ZeroGPU Space. Gradio and Docker Spaces now require a paid plan, except that free personal accounts (verified email, account older than 30 days) can host up to 2 ZeroGPU Spaces. Visitors have a small daily GPU quota (about 5 minutes on a free account). GPU functions need the `@spaces.GPU` decorator, and models should be loaded once, not per request. Test with a separate account before the evaluation.
- **Licence precaution:** do not upload INDB data into a public Space unless permission is confirmed. A private Space or local demo avoids the issue.

---

## 11. Risks and Limitations

| Risk | Mitigation |
|---|---|
| OWLv2 misses or confuses ingredients | Editable list, manual add, tuned threshold and NMS, visually detectable vocabulary |
| Dish and nutrition mismatch | Manual mapping verification, "unavailable" instead of guessing |
| Nutrition is approximate | INDB values are raw-ingredient sums, portion is user-selected, always labelled as an estimate |
| Domain shift (internet photos vs real plates) | Independent phone-photo test set |
| Unit inconsistency in INDB | Convert to grams with ingredient-specific factors before weight scoring |
| INDB licence unclear | Keep data local or private, seek permission before sharing |
| Scope creep | Follow the priority list below |

---

## 12. Scope Priorities

**Must implement:** dish classification, ingredient detection, editable ingredient list, recipe recommendation, nutrition per serving, two-mode interface, proper evaluation.

**If time permits:** Grad-CAM, MobileNetV3 comparison, TF-IDF baseline, confidence-based alternative predictions, ZeroGPU public demo.

**Skip in version 1:** training a detector, SAM portion estimation, large architecture comparisons, LLM recipe generation, neural recommenders, React frontend.

---

## 13. Viva Preparation

1. **Why OWLv2 instead of CLIP or YOLO?** CLIP gives image-level similarity and cannot localise items in a cluttered image. A standard pretrained YOLO does not know these ingredient classes. OWLv2 takes text queries, so it detects without training.
2. **Why a closed vocabulary of 60-80 ingredients?** One shared vocabulary keeps the detector, recipes, recommender and UI consistent, and it limits detection to ingredients that can actually be seen.
3. **Why coverage-based recommendation?** It is explainable and easy to evaluate, and it does not need an LLM or training data. Count and weight variants are compared experimentally.
4. **Why is the nutrition approximate?** Values come from raw-ingredient sums in INDB without full cooking-loss adjustment, the portion is user-selected, and a photo cannot give true weight.
5. **Why Hit@5 and not precision@5?** The synthetic test checks recovery of the source recipe, not suitability of every recommendation.
6. **Why not report nutrition MAE?** Nutrition is a database lookup, not a prediction model.

---

## 14. Suggested Repository Structure

```
AaharAI/
  README.md
  SPEC.md
  requirements.txt
  docs/
    AaharAI_implementation_plan.md
  data/                      # local only, not committed (INDB licence)
    indb/
    khana_subset/
    ingredients.csv
    mapping.csv
  src/
    schema.py
    classifier.py
    ingredients.py           # OWLv2 + NMS + canonical mapping
    recommender.py
    nutrition.py
    app.py                   # Gradio
  notebooks/
    01_data_audit.ipynb
    02_train_classifier.ipynb
    03_evaluation.ipynb
  tests/
  results/
```

---

## 15. References

- Khana: Prabhu, O. (2025). *Khana: A Comprehensive Indian Cuisine Dataset.* arXiv:2509.06006.
- INDB: Indian Nutrient Databank. Repository: `github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB`. Paper: *Current Developments in Nutrition*, 2024 (8(7):103790).
- OWLv2: Hugging Face model `google/owlv2-base-patch16-ensemble`.
- EfficientNet: Tan and Le (2019), implemented through `timm`.

---

## 16. Decisions log: what changed after the Day 1 data audit

These supersede earlier text in this document where they differ.

### 16.1 INDB data layer
- Final tables (built by `src/indb.py`): `recipes_clean.csv`, `recipes_kept.csv` (931 recipes with at least 80% of their non-staple ingredients in the vocabulary and serving info), `nutrition.csv`. 526 of the kept recipes have at least 5 core ingredients and are used for the hide-ingredients test.
- Ingredient categories live in `src/vocab.py` (`detectable`, `manual_add`, `staple`, `ignore`). Real ingredients such as gelatine, syrup and cola are `manual_add`, not `ignore`, so coverage is not inflated. Rule-based renames are limited to splits that matter for matching (buttermilk, condensed milk, breadcrumbs, ground ginger), so the detector label `egg`, `milk`, `lemon` etc. still matches recipe ingredients.
- Source-data issues documented (not silently fixed): food codes with several names (E017, E516, H018, K505, L002), one recipe-name mismatch (OSR038 Greenpea/Sunflower cookies), 82 recipes without serving info, 203 recipes with a guessed serving unit.

### 16.2 Nutrition caveats
- `kcal` in `nutrition.csv` is per ONE serving unit (1 idli, 1 bowl, 1 plate). INDB defines serving = recipe total / number of servings, and unit serving = serving / size of serving. `kcal_serving` = `kcal` x size of serving.
- INDB sums raw ingredient amounts, so fried and syrupy dishes are inflated (for example 4,550 kcal for a plate of onion pakoda). Recipes with more than 600 kcal per 100 g or more than 800 kcal per unit get `nutrition_flag = "check"` (145 of 1,014 recipes). The app shows a warning on these values.

### 16.3 Khana-to-INDB mapping (`mapping.csv`)
- Statuses: `match` (same dish), `variant` (same dish, different recipe details), `proxy` (closest dish in the same family, shown as approximate), `none` (no nutrition shown).
- Result: 48 of 80 classes have nutrition (match 36, variant 4, proxy 8 in the first full pass), covering about 72.6% of all Khana images. The earlier target of 60 classes was not reachable because INDB is a home-cooking recipe collection, while Khana includes street food, sweets and non-Indian dishes.
- The dish classifier is trained on all 80 classes. Nutrition lookup is a separate step and says "nutrition unavailable" for `none`.
- Khana training: stratified 70/15/15 split, exact duplicates removed, training split capped at 1,000 images per class (validation and test untouched).

### 16.4 Recommender scoring (replaces the coverage-first ranking in section 6.3)
The hide-ingredients test (fixed seed 42, 503 recipes, Hit@5) showed that coverage-first ranking is fragile. With a realistic pantry that also holds unrelated items, tiny recipes become "fully covered" and win. Jaccard similarity, |have ∩ required| / |have ∪ required|, is far more robust, and weight-based coverage is the weakest. The default is therefore `jaccard`; `count` and `weight` remain as ablation baselines.

| Unrelated items in pantry | Hidden | Count | Weight | Jaccard |
|---|---|---|---|---|
| 0 | 20% | 0.72 | 0.49 | 0.99 |
| 0 | 30% | 0.61 | 0.39 | 0.97 |
| 0 | 50% | 0.44 | 0.35 | 0.73 |
| 5 | 20% | 0.33 | 0.11 | 0.87 |
| 5 | 30% | 0.21 | 0.11 | 0.81 |
| 5 | 50% | 0.02 | 0.05 | 0.30 |
| 10 | 20% | 0.05 | 0.01 | 0.52 |
| 10 | 30% | 0.03 | 0.02 | 0.37 |
| 10 | 50% | 0.00 | 0.00 | 0.07 |

Reproduce with `notebooks/04_recommender_eval.ipynb`. Limitation: Hit@5 measures recovery of the source recipe, not whether every suggestion is sensible, so the 10-15 manual pantry scenarios in the same notebook are reported alongside it.

### 16.5 OWLv2 ingredient detection: first dev results (15 photos of my own kitchen)
- Setting: OWLv2 `google/owlv2-base-patch16-ensemble`, prompt "a photo of {ingredient}", class-agnostic NMS (IoU 0.5), 41 detector prompts.
- Micro-averaged over the 15 photos, threshold 0.3: precision 0.47, recall 0.69, F1 0.56 (18 true positives, 20 false positives, 8 misses). At 0.2: precision 0.39, recall 0.69. Lowering the threshold to 0.1 only added one true positive, so recall is capped by something other than the threshold.
- Reading the errors: 5 of the 8 misses were ingredients outside the detector vocabulary (radish in 4 photos, corn in 1), not model failures, and one "miss" was a capitalisation bug in the evaluation code (`Capsicum` vs `capsicum`). Recall over ingredients the detector is actually asked about was therefore about 0.86-0.9. Precision is the weak part: leafy or small items are over-detected (garlic, carrot, spring onions, lettuce, ginger, cabbage, orange, parsley).
- Changes made: `radish` and `corn` moved into the detector vocabulary (43 prompts); labels are compared case-insensitively with aliases; the evaluation separates in-vocabulary and out-of-vocabulary misses; threshold, relative-threshold and prompt-template choices are tuned on the dev photos and then confirmed once on a separate held-out set of new photos.
- Product decision: the detector only *suggests*. Detections at or above the chosen confident threshold are pre-selected in the editable ingredient list; weaker candidates are listed unselected, and the user can always add or remove items manually.

### 16.6 OWLv2 results after tuning, and the held-out test
- **Dev set (15 photos, used for tuning).** Best setting: threshold 0.4, plain prompt "a photo of {ingredient}". Precision 0.79, recall over in-vocabulary ingredients 0.85, F1 0.81 (22 true positives, 6 false positives, 4 misses). Prompt templates made little difference (plain F1 0.81, "a photo of a fresh {}" 0.81, bare label 0.74), so the plain prompt is kept.
- **Held-out test (19 new photos, run once at threshold 0.4, plain prompt).** Precision 0.72, recall over in-vocabulary ingredients 0.64, recall over everything labelled 0.55, F1 0.68 (in-vocabulary) / 0.62 (overall). 68 true positives, 26 false positives, 56 misses (39 in vocabulary, 17 not in vocabulary). These are the numbers to report. The drop from 0.81 to 0.68 is expected: the setting was tuned on 15 photos and the test photos are more varied and more cluttered.
- **Error analysis.**
  - Not in the detector vocabulary (17 misses): broccoli 10, beetroot 2, avocado 2, green onion 1 (an alias of spring onions), kale 1, dill 1.
  - In-vocabulary misses (39): capsicum 8, cucumber 4, celery 3, corn 3, lettuce 3, spinach 3, parsley 2, coriander leaves 2. Leafy greens are the weakest group.
  - False positives (26): bottle gourd 6, lettuce 3, orange 2, potato 2, cabbage 2, apple 2.
- **Changes made after looking at the test errors:** broccoli and beetroot moved into the detector vocabulary (45 prompts) and "green onion" aliased to spring onions. Because these changes were chosen from the test photos, those 19 photos are now development data. Any further claim about the improved system needs a fresh set of photos.
- **Interface decision (two tiers).** Detections at or above the confident threshold are pre-selected; weaker candidates are listed as unselected "maybe" items, so the user can add them with one click. Notebook 02 measures how much recall this recovers (`candidate_recall`) and how many extra suggestions it costs (`extra_maybe_per_photo`).
- **Conclusion.** Zero-shot OWLv2 is usable as a suggestion layer for common vegetables, not as an automatic ingredient reader. The editable list and manual add are part of the design for that reason.
