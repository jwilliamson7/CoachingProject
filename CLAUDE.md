# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science research project analyzing NFL coaching tenure and performance. It scrapes NFL coaching and team performance data from pro-football-reference.com. There have been three modeling directions over the project's life:

1. **Firing-aware survival analysis (CURRENT direction)** — time-to-FIRING with a Cox-primary, competing-risks pipeline (involuntary non-retention = event; voluntary exits = competing event; coaches active at the data boundary = censored). Predicts dismissal risk at the time of hire from PRE-HIRE covariates. Target conference: JQAS. The `scripts/survival_*.py` modules and `scripts/data/engineer_career_features.py` drive this.
2. **Coaching tenure** (ordinal classification: 1-2 / 3-4 / 5+ years) — LEGACY. Earlier strong results (QWK ~0.744) were LEAKAGE-INFLATED (the target sat in the SVD imputation matrix run before the train/test split); the leakage-free truth is weak (QWK ~0.29). Those numbers have been removed from this file.
3. **Coach WAR** (regression: average Wins Above Replacement per season) — LEGACY.

The data-creation pipeline (scraping → `create_data.py`) is shared across all three. The sections below retain the legacy tenure/WAR details for reference; the data-pipeline and feature-architecture sections have been corrected to the current single-file build.

## Project Structure

```
CoachingProject/
├── archive/
│   └── notebooks/           # Archived Jupyter notebooks (v2-v5)
├── data/
│   ├── models/              # Trained model files (.pkl)
│   ├── master_data.csv      # THE single canonical modeling dataset: 371 hire stints,
│   │                        #   94 modeled feature cols, un-imputed (1970+, hygiene +
│   │                        #   corrected-tenure + engineered features). Built in one
│   │                        #   pass by create_data.py. (No more _extended / svd_imputed
│   │                        #   files — imputation is now fit per train split.)
│   ├── coach_war_trajectories_with_team.csv  # WAR data by coach/team/year (legacy)
│   └── war_prediction_data.csv  # Merged WAR prediction dataset (legacy)
├── latex/
│   └── figures/             # Paper figures ONLY (7 files used in LaTeX)
├── figures/                 # All exploratory/supplementary figures
│   ├── tenure/              # Tenure model figures (SHAP, feature importance)
│   ├── war/                 # WAR model figures
│   └── backgrounds/         # Coach background analysis figures
├── analysis/                # Generated analysis data (CSVs + text reports)
├── model/                   # Model code package
│   ├── __init__.py
│   ├── ordinal_classifier.py    # Frank-Hall ordinal classification
│   ├── coach_tenure_model.py    # Main model wrapper
│   ├── war_regressor.py         # WAR regression model
│   ├── cross_validation.py      # Coach-level CV utilities
│   ├── evaluation.py            # Ordinal and regression metrics
│   └── config.py                # Hyperparameters and settings
├── scripts/
│   ├── data/                # Data processing scripts
│   │   ├── coach_scraping.py
│   │   ├── create_data.py            # SINGLE dataset builder -> data/master_data.csv
│   │   ├── engineer_career_features.py  # feature/tenure/hygiene LIBRARY + build_modeling_dataset()
│   │   ├── create_war_data.py        # Merge WAR with features (legacy)
│   │   ├── team_data_scraping.py
│   │   ├── transform_team_data.py
│   │   ├── matrix_factorization_imputation.py  # SVDImputer class (fit per split); main() is legacy
│   │   └── detailed_data_comparison.py  # legacy comparison (obsolete inputs)
│   ├── survival_*.py            # Firing-survival pipeline (CURRENT): survival_definitive,
│   │                            #   survival_methods, survival_analysis, survival_null_baseline, ...
│   ├── build_event_labels.py / merge_event_labels.py  # firing-vs-voluntary event labels
│   ├── generate_figures.py      # Paper + exploratory tenure figures (legacy)
│   ├── generate_war_figures.py  # WAR model figures
│   ├── shap_analysis.py        # SHAP exploratory analysis
│   ├── shap_analysis_by_background.py  # SHAP by coach background
│   ├── analyze_war_subgroups.py # WAR subgroup analysis
│   ├── coach_background_from_history.py  # Coach background classification
│   ├── train.py             # Tenure model training
│   ├── train_war.py         # WAR model training
│   ├── evaluate.py          # Model evaluation
│   └── predict.py           # Predictions for new coaches
├── Coaches/                 # Raw coach data (scraped)
├── Teams/                   # Raw team data (scraped)
├── League Data/             # League statistics by year
├── data_constants.py        # Shared constants
├── requirements.txt         # Python dependencies
└── README.md
```

## Development Workflow

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Collection and Processing
```bash
# Scrape coach data (be mindful of rate limiting)
python scripts/data/coach_scraping.py

# Scrape team data
python scripts/data/team_data_scraping.py

# Transform team data into league-wide statistics
python scripts/data/transform_team_data.py

# Build THE modeling dataset (data/master_data.csv) in one pass. create_data.py
# constructs the all-era base hiring instances in memory, then calls
# engineer_career_features.build_modeling_dataset() to filter to the modern era
# (1970+), apply population hygiene (drop interim caretakers, fold interim->permanent
# promotions into the season-opening hire), correct tenure labels, and append the
# engineered career-path/rank/org/roster/team-quality features. ONE builder, ONE file.
python scripts/data/create_data.py
```

There is no separate pre-imputation step anymore. The deprecated
`svd_imputed_master_data.csv` leaked the imputation across the train/test split (and
the target sat in the SVD matrix). The corrected pipeline imputes per split: the
`SVDImputer` is fit on each split's TRAINING partition only (feature columns only),
inside `model/pipeline.leakage_free_split`.

### Tenure Model Training and Evaluation
```bash
# Train ordinal model (recommended)
python scripts/train.py

# Train multiclass model
python scripts/train.py --multiclass

# Train and compare both models
python scripts/train.py --compare

# Train with hyperparameter tuning
python scripts/train.py --tune --n-iter 500

# Evaluate trained models
python scripts/evaluate.py --compare

# Make predictions for recent hires
python scripts/predict.py
```

### WAR Model Training
```bash
# Create WAR prediction dataset (merges WAR trajectories with features)
python scripts/data/create_war_data.py

# Train WAR regression model (uses optimized params by default)
python scripts/train_war.py

# Train with hyperparameter tuning
python scripts/train_war.py --tune --n-iter 1000
```

## Model Package (`model/`)

### Ordinal Classification (Frank-Hall Method)
The project uses ordinal classification for tenure prediction since classes have natural ordering (0 < 1 < 2). The Frank-Hall method trains K-1 binary classifiers:
- **Classifier 1**: P(Y > 0) - distinguishes class 0 from classes 1+2
- **Classifier 2**: P(Y > 1) - distinguishes classes 0+1 from class 2

Class probabilities derived as:
- P(Y = 0) = 1 - P(Y > 0)
- P(Y = 1) = P(Y > 0) - P(Y > 1)
- P(Y = 2) = P(Y > 1)

### Key Classes
- `OrdinalClassifier`: Frank-Hall implementation with sklearn-compatible API
- `CoachTenureModel`: Main wrapper supporting both ordinal and multiclass modes
- `WARRegressor`: XGBoost regression model for predicting coach WAR
- `CoachLevelStratifiedKFold`: Cross-validation preventing coach data leakage

### Evaluation Metrics
- **MAE**: Mean Absolute Error (ordinal distance)
- **QWK**: Quadratic Weighted Kappa (penalizes distant errors)
- **Adjacent Accuracy**: Predictions within 1 class of true label
- **AUROC**: Area Under ROC Curve (macro-averaged)
- **Per-class F1**: Precision/recall balance per tenure class

### Model Performance (legacy)
The legacy tenure-classification and WAR-regression performance numbers were produced
under the imputation leak and have been removed (not reproducible or citable). The
project's live evaluation is the firing-survival pipeline; `CoachTenureModel` and
`WARRegressor` remain in the `model/` package for the legacy lineages.

## Data Features and Architecture

`data/master_data.csv` has **94 modeled feature columns** per hire (`iloc[:, 2:-2]`):
the 51 base features built by `create_data.py` plus 43 engineered features appended by
`engineer_career_features.build_modeling_dataset()`. (The old all-era 150-feature,
four-parallel-block layout is retired.)

### Base features from create_data.py (51)
- **Core coach (8):** age at hire; number of previous HC stints; years of experience by
  level (College vs NFL) × role (Position, Coordinator, Head Coach). Coordinator years are
  counted from history regardless of PFR ranks coverage (so special-teams coordinators
  count); each season is counted once (PFR sub-role duplicate rows are deduped).
- **Pooled unit performance (33, the `__unit` block):** the 33 base team statistics
  (points, yards, turnovers, efficiency, drive stats) pooled into ONE season-weighted,
  orientation-corrected block where positive always means "good unit". An OC season
  contributes team offense (+), a DC season opponent-allowed (−, low allowed = good), an
  HC season BOTH sides. Replaces the legacy four parallel `__oc/__dc/__hc/__opp__hc`
  blocks. Each season's team file is located year-aware (relocation-correct).
- **Hiring-team context (10):** the hiring franchise's previous 2 years (win %, points,
  yards, turnovers, playoff appearances), as-of the hire year.

### Engineered features from engineer_career_features.py (43)
- Tier 1 career-path (`cf_*`): relocation-aware employer counts, NFL share, ages, gap
  before hire, internal-hire flag, years at hiring org, most-recent role/side, structural
  OC/DC-experience indicators.
- Tier 2 rank recency/trajectory (`rf_*`): side-appropriate unit percentile + team-success
  percentile + trajectory from prior ranked seasons.
- Tier 5 org instability, Tier 6 inherited roster/talent (`hire_*`), and team-quality SRS
  block (`tq_*`) — all measured strictly before the hire year (no leakage).

### Target Variables
- `Avg 2Y Win Pct` (column −2) and `Coach Tenure Class` (column −1): 0 (≤2 yrs), 1 (3-4),
  2 (5+), −1 (recent active / insufficient data). Tenure is recomputed relocation- and
  partial-season-aware via `reconstruct_tenure`. For the survival pipeline, duration +
  firing-aware event/competing-risk labels come from `reconstruct_tenure` +
  `analysis/event_labels_final.csv` (see survival_methods).

## Important Implementation Details

### Coach-Level Cross-Validation
To prevent data leakage, train/test splits ensure no coach appears in both sets. Coaches with multiple hiring instances (e.g., Bill Belichick 1991, 2000) are kept together.

### Team Abbreviation Mapping
Franchise identity is resolved YEAR-AWARE via `data_constants.standardize_team_abbreviation(abbr, year)`,
which returns the canonical PFR franchise key and correctly disambiguates abbreviations that
meant two franchises across eras (BAL → Colts `clt` ≤1983 / Ravens `rav` after; HOU → Oilers
`oti` ≤1996 / Texans `htx` after; STL → Cardinals `crd` ≤1987 / Rams `ram` after). This is the
canonical resolver used by both the stint detector and the per-season performance file locator.
The flat `TEAM_FRANCHISE_MAPPINGS` dict is only a fallback set of legacy file spellings — on its
own it is NOT year-aware (using it alone caused the Cardinals→Rams / Patriots→Washington
misattribution bug that was fixed by making the locator try the standardized key first).

### Career Parsing Logic
- Generates a separate data point for each head-coaching hire; tracks career progression
  and identifies HC transitions (continuity is judged year-aware via the canonical
  franchise key + the head-coaching employer name, so relocations/renames and mid-season
  exit fragments don't create spurious stints).
- Population hygiene (Coach_WAR Primary_Coach resolution, in `create_data._classify_hire`):
  mid-season interim caretakers who never open a season as primary are NOT emitted;
  interim→permanent promotions are anchored at the season-opening year with the interim
  partial HC season FOLDED into prior experience (performance + HC-year count + age/context
  as-of the anchored year). `Sean Payton 2013` is a documented `EXCLUDED_HIRING_INSTANCES`
  override (2012 suspension creates a ranks gap that otherwise looks like a fresh hire).
- Handles current coaches (hired 2022+ / still active) as tenure class -1 (insufficient
  data); these are right-censored in the survival pipeline.

### Missing Data Handling
- `data/master_data.csv` is stored UN-imputed. SVD matrix-factorization imputation
  (`SVDImputer`) is fit per split on the TRAINING partition's feature columns only, then
  applied to both partitions (see `model/pipeline.leakage_free_split`). The target never
  enters the imputation matrix.
- Historical statistics (pre-~1990s) naturally have NaN for advanced drive/efficiency
  metrics; structural-missingness indicators (`cf_ever_nfl_oc/dc`) let the model condition
  on whether a unit block is structurally absent vs missing-at-random.

## Configuration

Key configuration in `model/config.py`:
- `XGBOOST_PARAM_DISTRIBUTIONS`: Hyperparameter search space
- `OPTIMIZED_XGBOOST_PARAMS`: Best parameters for tenure classification
- `OPTIMIZED_WAR_PARAMS`: Best parameters for WAR regression
- `MODEL_PATHS`: File paths for data and models
- `ORDINAL_CONFIG`: Tenure class definitions
- `WAR_CONFIG`: WAR model feature configuration

## Research Output
- LaTeX files for academic paper formatting (IEEE conference style) in `LaTeX/`
- Trained models saved to `data/models/`
- Archived notebooks in `archive/notebooks/`
