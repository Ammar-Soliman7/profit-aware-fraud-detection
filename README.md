# Profit-Aware Fraud Detection

A machine-learning system that decides what to do with each card transaction —
**accept, send to manual review, or reject** — while optimising for **money saved**
rather than raw accuracy. A hybrid classifier scores each transaction, and a
reinforcement-learning agent turns those scores into decisions, taking into account
the transaction amount and a limited human-review budget.

> **Graduation project.** Trained and evaluated on the IEEE-CIS Fraud Detection dataset (Kaggle).

---

## The idea

Most fraud models are trained to maximise accuracy or AUC. But in the real world, not every
mistake costs the same: letting a big fraudulent purchase through is far more expensive than
letting a small one through, and wrongly blocking a good customer has its own cost. This
project optimises the decision policy directly for **financial outcome** — a metric called
**Profit Gain** — instead of treating every error as equal.

The system has three parts working together:

1. **A hybrid classifier** that produces a risk score for each transaction.
2. **A reinforcement-learning (PPO) agent** that learns *where* to set the accept / review /
   reject thresholds — and adjusts them based on the transaction amount.
3. **A capacity layer** that keeps the number of "send to human review" decisions within a
   realistic daily budget.

Everything is wrapped in an interactive **Streamlit dashboard** for exploring the results.

## Results (deployed run)

Evaluated on a held-out test set of **88,581 transactions**:

| Metric | Value |
|---|---|
| Profit Gain | **0.88** (captures ~88% of the profit an ideal/oracle policy would) |
| ROC-AUC (test) | 0.913 |
| Recall (fraud caught) | 0.88 |
| Precision | 0.70 |
| False-positive rate | 1.3% |
| Review rate | 27% of transactions sent to human review |

The economic model uses: a 2% profit rate on legitimate transactions, a 5x cost multiplier for
wrongly rejecting a good customer, a 3x multiplier for missed fraud, and a fixed cost per manual
review. These are the parameters the Profit Gain metric is computed against.

## How it works (the pipeline)

**1. Classifier — a hybrid score.**
A denoising **autoencoder** produces an anomaly score (how "unusual" a transaction looks), which
is blended with a **CatBoost** gradient-boosting probability. The blend is ~95% CatBoost
(alpha = 0.95); the autoencoder is a minor supporting signal. The score is left **uncalibrated**
on purpose — the RL agent relies on the raw spread of the CatBoost score, and calibration was
tested and removed because it hurt generalisation from validation to test.

**2. Decision policy — a PPO reinforcement-learning agent.**
Instead of a single fixed threshold, a **PPO** agent learns two thresholds — a lower one
(above which a transaction gets reviewed) and an upper one (above which it's rejected) — and
lets them **depend on the transaction amount**. Bigger amounts get more cautious treatment. The
agent is trained on a batched stream of transactions with the reward being (negative) financial
cost, so it learns the policy that makes the most money, not the one that's most "accurate."

**3. Capacity layer.**
Human review isn't free or unlimited. The system enforces a **binding review budget** (e.g. a
set number of reviewers x reviews per day). Under that budget it re-tunes a static base policy
and adds a budget-pressure triage step, so it never recommends more reviews than the team can
actually handle. The project also tests whether this adaptive, budget-aware policy genuinely
beats a well-tuned fixed policy (it does, within the profitable range).

The notebook also includes the supporting work you'd expect from a thesis: grid-search baselines,
amount-bucketed comparisons, a leak/negative-control test, drift analysis, ablations, and a
held-out test evaluation.

## The dashboard

A Streamlit app visualises the deployed policy:

- Per-transaction decisions (accept / review / reject) and risk scores
- Cost breakdown — where the money goes (missed fraud, false rejections, review cost, overload)
- Net financial outcome vs. a "do nothing" baseline and an ideal oracle
- A reviewer-capacity slider that re-tunes the policy live (the trained RL policy stays frozen;
  only the budget-dependent parts re-derive)

The dashboard and the notebook are connected by a small **four-file contract** — the notebook
exports four files, and the dashboard reads them. This keeps the dashboard's numbers guaranteed
to match the thesis.

## What's in this repository

- **fraud_rl_pipeline.ipynb** — the full research pipeline (classifier, RL policy, capacity
  layer, baselines, and analysis).
- **fraud_dashboard_project** — the Streamlit dashboard, its shared library (fraudcore), and the
  export/regenerate scripts.

The trained model files (the CatBoost model, autoencoder, and RL checkpoints) are not stored in
this repository because of their size. They are produced by running the notebook.

## How to run

**Prerequisites:** Python 3.9+.

**1. Get the dataset.** Download the IEEE-CIS Fraud Detection dataset from Kaggle
(https://www.kaggle.com/c/ieee-fraud-detection) and place the CSV files where the notebook
expects them (see the path constants at the top of the notebook).

**2. Run the pipeline.** Open fraud_rl_pipeline.ipynb and run it top to bottom. This trains the
classifier, the PPO policy, and the capacity layer, and writes the four files the dashboard needs.

**3. Launch the dashboard:**

```
cd fraud_dashboard_project
pip install -r requirements.txt
streamlit run dashboard.py
```

If you already have the exported snapshot and just want to rebuild the dashboard's files without
retraining:

```
python regenerate_dashboard.py --reviewers 8 --reviews-per-day 100
```

## Tech stack

Python, CatBoost, TensorFlow/Keras (autoencoder), PyTorch (PPO), NumPy, pandas, Streamlit, Plotly.

## Dataset

IEEE-CIS Fraud Detection (Kaggle). The raw data is not included here for size and licensing
reasons — download it from Kaggle directly.

---

*Built as a graduation project. The full methodology, baselines, and analysis live in the notebook.*
