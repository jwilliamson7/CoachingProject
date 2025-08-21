# NFL Coaching Analytics Platform

A comprehensive data science platform for analyzing NFL coaching performance, tenure patterns, and team dynamics. This project provides predictive modeling capabilities for coaching success and organizational decision-making in professional football.

## Overview

This platform combines extensive historical NFL data (1920-2025) with advanced feature engineering to create predictive models for coaching tenure and team performance. By analyzing coaching careers, team statistics, and organizational contexts, the system provides insights into the factors that drive coaching success in the NFL.

## Key Features

### Data Collection & Processing
- **Comprehensive Data Scraping**: Automated collection of coaching histories, team performance metrics, and league statistics from pro-football-reference.com
- **Historical Coverage**: Complete NFL data spanning from 1920 to 2025, covering over 100 years of professional football
- **Robust Data Pipeline**: Handles franchise relocations, team name changes, and historical inconsistencies

### Advanced Feature Engineering
- **154 Features per Coaching Instance**: Detailed feature set including coaching experience, team performance metrics, and organizational context
- **Multi-dimensional Analysis**: Tracks coaching progression across college and professional levels, multiple roles (Position Coach, Coordinator, Head Coach)
- **Normalized Performance Metrics**: All team statistics normalized relative to league averages for temporal consistency

### Predictive Modeling
- **Coaching Tenure Classification**: Predicts coaching longevity (≤2 years, 3-4 years, 5+ years)
- **Performance Forecasting**: Models 2-year winning percentage projections for new hires
- **Context-Aware Predictions**: Incorporates hiring team's recent performance and organizational factors

## Data Architecture

### Core Feature Categories

**Coaching Experience (8 features)**
- Age at hire
- Previous head coaching experience
- Years by coaching level and role

**Team Performance (128 features)**
- Offensive and defensive statistics by coaching role
- Points, yards, turnovers, efficiency metrics
- Drive statistics and situational performance

**Organizational Context (10 features)**
- Hiring team's recent performance trends
- Playoff history and competitive positioning

**Target Variables**
- 2-year average winning percentage
- Coaching tenure classification

### Data Quality & Validation
- Comprehensive team abbreviation mapping for franchise changes
- Career progression tracking with role transition detection
- Missing data handling with graceful error reporting
- Feature validation ensuring 154 dimensions per instance

## Getting Started

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy scipy sympy
pip install beautifulsoup4 requests
pip install scikit-learn xgboost
pip install matplotlib seaborn
pip install pyarrow  # For Feather file support
```

### Data Collection
```bash
# Collect coaching data (rate-limited for API compliance)
python coach_scraping.py

# Gather team performance data
python team_data_scraping.py

# Transform and normalize team data
python transform_team_data.py

# Generate feature-engineered dataset
python create_data.py

# Optional: Run data validation
python detailed_data_comparison.py
```

### Output Files
- `master_data.csv`: Complete feature-engineered dataset ready for machine learning
- `Coaches/`: Individual coach data files in CSV and Feather formats
- `League Data/`: Yearly league statistics (raw and normalized)
- `Teams/`: Team-specific performance data
- `XGBC_best_params_*.pkl`: Optimized model parameters for different configurations

## Applications

- **Front Office Analytics**: Support hiring decisions with data-driven coaching assessments
- **Performance Prediction**: Forecast coaching success probability for new hires
- **Organizational Planning**: Understand factors contributing to coaching longevity
- **Academic Research**: Sports analytics research in coaching effectiveness and team dynamics

## Technical Stack

- **Data Processing**: Python (pandas, numpy, scipy, sympy)
- **Web Scraping**: BeautifulSoup4, requests with rate limiting
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn
- **Storage**: CSV and Feather formats for performance optimization
- **Research Output**: LaTeX for academic publication formatting
- **Development**: Jupyter notebooks for iterative model development

## Project Structure

```
CoachingProject/
├── coach_scraping.py           # Coach data collection
├── team_data_scraping.py       # Team performance scraping
├── create_data.py              # Feature engineering pipeline
├── transform_team_data.py      # Data transformation utilities
├── matrix_factorization_imputation.py  # Missing data handling
├── data_constants.py           # Configuration and constants
├── detailed_data_comparison.py # Data validation tools
├── Coaches/                    # Individual coach data
├── League Data/                # Historical league statistics
├── Teams/                      # Team-specific data
├── Williamson-Jon-*.ipynb      # Model development notebooks
└── master_data.csv             # Final feature-engineered dataset
```

## Model Development

The project includes multiple Jupyter notebooks documenting the iterative model development process:

- **CoachTenureModels (v2-v6)**: Progressive improvements in coaching tenure classification
- **CoachWinPctModels (v1)**: Winning percentage prediction models

Each notebook version represents refinements in feature selection, hyperparameter tuning, and model evaluation strategies.

## Data Quality Notes

- **Historical Coverage**: Statistics availability varies by era (basic stats from 1920, advanced metrics from 1990s+)
- **Missing Data**: Handled appropriately with matrix factorization for imputation where suitable
- **Franchise Changes**: Comprehensive mapping handles team relocations and name changes
- **Validation**: Extensive data quality checks ensure 154 features per coaching instance

## Research Impact

This platform supports academic research in sports analytics and has been developed for presentation at sports analytics conferences (SSAC), contributing to the growing field of data-driven decision making in professional sports management. The project originated as coursework for CSCE421 and has evolved into a comprehensive research platform for NFL coaching analytics.
