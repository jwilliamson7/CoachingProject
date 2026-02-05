# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science research project analyzing NFL coaching tenure and performance. The project was developed for a Computer Science course (CSCE421) and includes research paper materials for sports analytics conferences (SSAC/JQAS). The project scrapes NFL coaching and team performance data from pro-football-reference.com to build predictive models for:
1. **Coaching tenure** (ordinal classification: 1-2 years, 3-4 years, 5+ years)
2. **Coach WAR** (regression: average Wins Above Replacement per season)

## Project Structure

```
CoachingProject/
├── archive/
│   └── notebooks/           # Archived Jupyter notebooks (v2-v5)
├── data/
│   ├── models/              # Trained model files (.pkl)
│   ├── master_data.csv      # Raw feature data
│   ├── svd_imputed_master_data.csv  # Imputed data for training
│   ├── coach_war_trajectories_with_team.csv  # WAR data by coach/team/year
│   └── war_prediction_data.csv  # Merged WAR prediction dataset
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
│   │   ├── create_data.py
│   │   ├── create_war_data.py   # Merge WAR with features
│   │   ├── team_data_scraping.py
│   │   ├── transform_team_data.py
│   │   ├── matrix_factorization_imputation.py
│   │   └── detailed_data_comparison.py
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

# Create feature dataset
python scripts/data/create_data.py

# Impute missing values with SVD
python scripts/data/matrix_factorization_imputation.py
```

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

### Model Performance (Ordinal vs Multiclass)
| Metric | Ordinal | Multiclass | Winner |
|--------|---------|------------|--------|
| MAE | 0.339 | 0.354 | Ordinal |
| QWK | 0.731 | 0.712 | Ordinal |
| Adjacent Acc | 98.4% | 97.6% | Ordinal |
| AUROC | 0.836 | 0.843 | Multiclass |
| Class 1 F1 | 0.466 | 0.451 | Ordinal (+3.3%) |

### WAR Regression Model
Predicts average WAR (Wins Above Replacement) per season during a coach's tenure. Uses 140 features (excludes hiring team factors) since WAR measures coaching ability independent of team context. Model is trained excluding recent hires (tenure class -1) to allow unbiased predictions.

| Metric | Value |
|--------|-------|
| R² | 0.276 |
| Pearson Correlation | 0.525 |
| MAE | 0.051 (5.1 pct points) |
| RMSE | 0.063 |
| vs Baseline | +27.8% |

WAR is expressed in percentage terms (proportion of wins above replacement).

## Data Features and Architecture

The project uses `scripts/data/create_data.py` to engineer 150 features per coaching hire:

### Core Coach Features (8 features)
- Age at time of hire
- Number of previous head coaching stints
- Years of experience by level (College vs NFL) and role (Position coach, Coordinator, Head Coach)

### Team Performance Features (128 features)
- 32 offensive statistics for each role (OC, DC, HC, Opposition when HC)
- Includes points, yards, turnovers, efficiency metrics, drive statistics
- All normalized relative to league averages for that year

### Hiring Team Context (10 features)
- Previous 2 years' performance of hiring team (win %, points, yards, turnovers, playoff appearances)

### Target Variables
- 2-year average winning percentage
- Coaching tenure classification: 0 (≤2 years), 1 (3-4 years), 2 (5+ years)

## Important Implementation Details

### Coach-Level Cross-Validation
To prevent data leakage, train/test splits ensure no coach appears in both sets. Coaches with multiple hiring instances (e.g., Bill Belichick 1991, 2000) are kept together.

### Team Abbreviation Mapping
Comprehensive dictionary in `data_constants.py` handles franchise relocations and name changes:
- Raiders: oak/rai/lvr
- Rams: stl/ram/lar
- Colts: bal/clt/ind

### Career Parsing Logic
- Excludes interim positions and non-coaching roles
- Tracks career progression and identifies head coaching transitions
- Handles current coaches hired after 2022 differently (insufficient tenure data)
- Generates separate data point for each head coaching hire

### Missing Data Handling
- SVD-based matrix factorization imputation for missing statistics
- Historical statistics (pre-1990s) naturally have NaN for advanced metrics
- Imputed data saved to `data/svd_imputed_master_data.csv`

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
