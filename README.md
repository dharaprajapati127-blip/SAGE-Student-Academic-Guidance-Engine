# 🎓 SAGE — Student Academic Guidance Engine

A faculty-assisted **early-warning system** that predicts which university students are
likely to fall into academic trouble **in the coming weeks** — early enough for a human to
intervene. SAGE is a *decision-support* tool: the AI flags and explains, faculty decide.

> **What makes this more than a classroom demo:** SAGE predicts the **future**, not the
> present. The model is trained to answer *"will this student drop below passing in the
> next 4 weeks?"* using only behavior observable today. That forward-looking framing is
> what turns a grade lookup into a genuine early-warning system — and it's what forces the
> model to learn from **trends** (declining slope, rising stress, volatility) rather than
> just reading off a current grade.

---

## 🎯 What it does

- Predicts each student's **probability of near-future academic trouble** (next 4 weeks)
- Bands students into **Low / Medium / High** risk for faculty triage
- **Explains every prediction** with SHAP, so faculty see *why* a student was flagged
- Surfaces a **priority alert queue** with suggested (non-prescriptive) actions
- Tracks **week-by-week risk trajectories** so early slides are visible before they're failures

---

## 🧠 The key design decisions (and why they matter)

| Decision | Why |
|---|---|
| **Forward-looking label** — predict trouble in weeks *t+1…t+4*, not at week *t* | Avoids *label leakage*; makes it a real early-warning task |
| **Group-aware split** (`GroupShuffleSplit` by student) | No student appears in both train and test → honest evaluation |
| **Probability metrics** (ROC-AUC, PR-AUC, Brier) vs. a base-rate baseline | Accuracy is misleading on imbalanced data (~21% positive) |
| **SHAP explanations** | Faculty can sanity-check the reason; supports human-in-the-loop ethics |
| **Per-subgroup robustness check** | Confirms the model doesn't silently fail any behavioral group (e.g. burnout) |

### A note on what was fixed from the baseline
The first version defined risk as `rolling_score_avg < 70` and then *fed that same average
to the model as a feature* — so it scored 99.8% by being handed the answer key (textbook
**label leakage**). It also used a random train/test split, which leaks adjacent weeks of
the same student across the split. Both are corrected here; the resulting ~0.96 ROC-AUC is
real predictive performance, not circular.

---

## 📊 Model performance (held-out students)

| Metric | Score | Notes |
|---|---|---|
| ROC-AUC | ~0.96 | ranking quality |
| PR-AUC | ~0.85 | vs. base-rate baseline of ~0.21 |
| Recall (at-risk) | ~0.88 | we'd rather over-flag than miss a struggling student |
| Brier | ~0.07 | well-calibrated probabilities |

Per-archetype AUC stays in the 0.95–0.99 range, so the model performs consistently across
*stable, declining, disengaging,* and *burnout* students.

*(Exact numbers regenerate when you run the notebooks; see `models/metrics.json`.)*

---

## 🗂️ Project structure

```
sage/
├── notebooks/
│   ├── 01_data_generator.ipynb            # simulate 200 students × 16 weeks, 4 archetypes
│   ├── 02_feature_engineering.ipynb       # trend features + forward-looking label  ← key fix
│   ├── 03_model_training.ipynb            # group-aware split + honest metrics
│   ├── 04_explainability_and_fairness.ipynb # SHAP + per-subgroup robustness
│   └── 05_real_world_validation.ipynb     # validates the approach on 4,424 REAL students (UCI)
├── src/                                   # same logic as importable .py modules
│   ├── simulate.py
│   ├── features.py
│   └── train.py
├── dashboard/
│   └── app.py                             # Streamlit faculty dashboard
├── data/                                  # generated CSVs
└── models/                                # trained model + metrics.json
```

## 🚀 Getting started

```bash
pip install -r requirements.txt

# Option A — run the notebooks 01 → 04 in order
# Option B — run the pipeline as scripts:
python src/simulate.py
python src/features.py
python src/train.py

# Launch the faculty dashboard
streamlit run dashboard/app.py
```

## 🛠️ Tech stack
Python · pandas · NumPy · scikit-learn · SHAP · Plotly · Streamlit · Jupyter

## 🌍 Real-world validation (notebook 05)

The synthetic data (notebooks 01–04) builds and tests the *methodology* in a controlled
setting. Notebook 05 then validates the core idea on **real data**: the UCI *Predict
Students' Dropout and Academic Success* dataset — 4,424 actual students, CC BY 4.0.

To keep the same *early-warning* spirit, we predict dropout using **only first-semester
data** and deliberately exclude second-semester features (which would leak the answer). The
notebook even demonstrates the leakage by training a second model *with* those features and
showing the score jump — proof we understand *why* we restricted to early signals.

> Get the data on your machine (it isn't bundled): download `data.csv` from the
> [UCI page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
> and save it as `data/real_students_uci.csv`, or use `pip install ucimlrepo` (instructions
> inside notebook 05).
>
> **Citation:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
> *Predict Students' Dropout and Academic Success* [Dataset]. UCI ML Repository.
> https://doi.org/10.24432/C5MC89

## ⚠️ Responsible-use note
SAGE uses **synthetic** data and is a learning/portfolio project. Risk scores are
probabilistic signals to prompt a human check-in — never automated decisions about real
students. Any real deployment would require consent, real fairness auditing across
protected groups, and faculty/counselor oversight.

## 👤 Author
**Dhara** — B.Tech Student | Aspiring Data Scientist
