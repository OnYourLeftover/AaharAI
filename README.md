# AaharAI

Vision-based Indian recipe recommendation and nutrition estimation system.

- **Mode A (dish photo):** EfficientNet-B0 trained on Khana -> verified INDB nutrition per serving.
- **Mode B (ingredient photo):** OWLv2 zero-shot detection -> editable ingredient list -> ranked INDB recipes with nutrition.

Full plan: `docs/AaharAI_implementation_plan.md`

## Where things live
| Place | What |
|---|---|
| This repo (GitHub) | code in `src/`, runner notebooks in `notebooks/`, docs |
| Google Drive `MyDrive/AaharAI/data/` | INDB files, cleaned CSVs, `mapping.csv`, `khana_splits.csv`, model weights |
| Google Drive `MyDrive/AaharAI/photos/` | your own test photos for the OWLv2 check |
| Colab | runs the notebooks (temporary disk; wiped on reset) |

## Rules
1. Change code in VS Code, then `git push`. Colab notebooks only *run* it (`git pull` happens in their first cell).
2. Never commit data (`data/` is git-ignored). INDB has no clear redistribution licence; Khana is non-commercial research use.
3. Do not save Colab notebooks back to GitHub: their outputs contain INDB rows.
4. Edit ingredient categories in `src/vocab.py`, never in a notebook.
