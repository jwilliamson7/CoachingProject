# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science research project analyzing NFL coaching tenure and performance. The project was developed for a Computer Science course (CSCE421) and includes research paper materials for sports analytics conferences (SSAC). The project scrapes NFL coaching and team performance data from pro-football-reference.com to build predictive models for coaching success and tenure.

## Key Components

### Data Collection Scripts
- `coach_scraping.py`: Scrapes individual coach data from pro-football-reference.com, extracting coaching results, rankings, and history
- `team_data_scraping.py`: Scrapes team performance data by year, including playoff information and team statistics  
- `create_data2.py`: Main feature engineering script that combines coach and team data (this is the current/active version)

### Data Structure
- `Coaches/`: Individual coach directories containing CSV and Feather files with coaching history, results, and rankings
- `League Data/`: Year-by-year league statistics (1920-2024) with team and opponent data, both raw and normalized
- `Teams/`: Team-specific data storage with team records and playoff data
- `master_data*.csv`: Consolidated datasets combining all features for model training (master_data7.csv is latest)

## Development Workflow

### Running Data Collection and Processing
```bash
# Scrape coach data (be mindful of rate limiting)
python coach_scraping.py

# Scrape team data  
python team_data_scraping.py

# Transform and combine data (current version)
python create_data2.py
```

### Analysis Environment
- Primary dependencies: pandas, numpy, sympy, scipy for data processing
- Data stored in both CSV and Feather formats for performance
- No package.json or requirements.txt - dependencies managed manually

### Research Output
- LaTeX files for academic paper formatting (IEEE conference style)
- Model parameter files (XGBC_best_params variants)
- Generated visualizations and statistical analysis

## Data Features and Architecture

The project uses `create_data2.py` to engineer 154 features per coaching hire including:

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

### Team Abbreviation Mapping
- Comprehensive dictionary mapping handles franchise relocations and name changes
- Examples: Raiders (oak/rai/lvr), Rams (stl/ram/lar), Colts (bal/clt/ind)

### Career Parsing Logic
- Excludes interim positions and non-coaching roles
- Tracks career progression and identifies head coaching transitions
- Handles current coaches hired after 2022 differently (insufficient tenure data)
- Generates separate data point for each head coaching hire

### Data Quality Controls
- Validates 154 features per instance
- Handles missing team data gracefully with error reporting
- Uses both raw and normalized league statistics

## Current Analysis Focus
- Model evaluation focuses on coaching tenure prediction and winning percentage forecasting
- Data spans from 1920 to 2025, covering the complete modern NFL era
- Recent hires (2025) included as new hire predictions without tenure classification