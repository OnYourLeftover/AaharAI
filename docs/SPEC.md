# AaharAI - Project Specification

## Objective
Vision-based Indian recipe recommendation and nutrition estimation system.

## Mode A: Dish recognition
Input: photo of an Indian dish.
Model: EfficientNet-B0 fine-tuned on Khana (80 classes).
Output: predicted dish (top-5 with confidence) and reference nutrition from INDB via `mapping.csv`
(status match / variant / proxy / none; "nutrition unavailable" when none).

## Mode B: Ingredient recognition
Input: photo of ingredients (fridge, counter, bowl).
Model: OWLv2, zero-shot (no training), prompts built from `vocab.DETECT`; class-agnostic NMS.
Output: editable ingredient list, then ranked INDB recipes (coverage first, fewest missing next)
with nutrition per serving.

## Interface and backend (version 1)
Python + Gradio. No FastAPI, SQLAlchemy or SQLite in version 1: the data layer is a handful of CSV tables
built from INDB by `src/indb.py` (recipes_clean.csv, recipes_kept.csv, nutrition.csv) plus `mapping.csv`.
See docs/AaharAI_implementation_plan.md.

## Data
- INDB (Indian Nutrient Databank): recipes, ingredient amounts, nutrition. Local/private only, licence unclear.
- Khana: dish images, non-commercial research use. Never committed.
- Nutrition caveat: INDB per-unit values come from raw ingredient sums, so fried and syrupy dishes are inflated.
  `nutrition_flag == "check"` marks them; 203 recipes also have a guessed serving unit.

## Development principles
- Build, test and integrate one module at a time.
- Logic lives in `src/`; notebooks only run it (they do not get saved back to GitHub).
- Ingredient categories are edited in `src/vocab.py` only.
