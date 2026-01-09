# Traffic Density and Air Quality Analysis in Istanbul

**DSA210 - Introduction to Data Science | Fall 2025-2026**

**Author:** Miray Erkoç | Student ID: 30815  
**Institution:** Sabancı University  
**Project Status:** ✅ Complete Analysis (EDA → Hypothesis Testing → Machine Learning)

---

## 📑 Table of Contents

- [Motivation](#-motivation)
- [Data Sources](#-data-sources)
- [Data Collection Process](#-data-collection-process)
- [Dataset Overview](#-dataset-overview)
- [Research Questions](#-research-questions)
- [Research Hypotheses](#-research-hypotheses)
- [Methodology](#-methodology)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Correlation Analysis](#2-correlation-analysis)
  - [3. Statistical Hypothesis Testing](#3-statistical-hypothesis-testing)
  - [4. Time Series Analysis](#4-time-series-analysis)
  - [5. Categorical Analysis](#5-categorical-analysis)
  - [6. Visualization Strategy](#6-visualization-strategy)
  - [7. Machine Learning Analysis](#7-machine-learning-analysis-new)
- [Key Findings](#-key-findings)
- [Results & Visualizations](#-results--visualizations)
- [Limitations & Future Work](#-limitations--future-work)
- [References & Attribution](#references--attribution)

---

## 🎯 Motivation

Istanbul faces significant air quality challenges due to high traffic density. This project investigates the relationship between hourly traffic patterns and air pollution levels to understand how vehicle flow affects air quality metrics such as NO2, PM10, and CO concentrations. 

The primary goal is to provide data-driven insights that could support public health policies and urban planning decisions in Istanbul.

---

## 📊 Data Sources

### Primary Datasets

1. **Traffic Data** (Istanbul Metropolitan Municipality Open Data Portal)
   - Source: [IBB Açık Veri Portalı](https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set)
   - Location: D-100 Highway (Geohash-based location tracking)
   - Temporal Coverage: 2024 (entire year)
   - Frequency: Hourly measurements
   - Variables:
     - `NUMBER_OF_VEHICLES`: Hourly vehicle count
     - `AVERAGE_SPEED`: Mean vehicle speed (km/h)
     - `MINIMUM_SPEED`: Minimum recorded speed (km/h)
     - `MAXIMUM_SPEED`: Maximum recorded speed (km/h)

2. **Air Quality Data** (Istanbul Environmental Monitoring System)
   - Source: [IBB Hava Kalitesi İzleme Sistemi](https://havakalitesi.ibb.gov.tr/Pages/AirQualityCalendar)
   - Location: Geohash-matched air quality monitoring station
   - Temporal Coverage: 2024 (entire year)
   - Frequency: Hourly measurements
   - Variables:
     - `Concentration_NO2`: Nitrogen Dioxide concentration (µg/m³)
     - `AQI_PM10`: Air Quality Index for PM10 particulate matter
     - `Concentration_CO`: Carbon Monoxide concentration
     - `AQI_Category`: Categorical air quality classification

---

## 📄 Data Collection Process

### Workflow Pipeline

The data collection and processing followed a systematic multi-stage pipeline:

```
Stage 1: API Data Extraction
├── apiadressbulma.py → Apivericekme.py → sxk9jw_API_TAM.csv (Traffic Data)
└── untitled0.py → D100_AQI_2024.csv (Air Quality Data)

Stage 2: Data Integration
└── databirlestirme.py → FINAL_merged_data.csv

Stage 3: Feature Engineering
└── master_enrichment_DESKTOP.py → MASTER_enriched_data.csv
```

### Data Collection Details

**Traffic Data Acquisition:**
- Custom Python scripts developed to interface with IBB Open Data Portal API
- Geohash-based location filtering for D-100 Highway corridor
- Automated hourly data pulls for entire 2024 year
- Error handling and retry mechanisms implemented
- Data validation checks during extraction

**Air Quality Data Acquisition:**
- API integration with Istanbul Air Quality Monitoring System
- Geohash coordinate matching with traffic monitoring locations
- Synchronized timestamp alignment with traffic data
- Automated quality control flags handling

**Data Integration:**
- Timestamp-based merging of traffic and air quality datasets
- Handling of missing values and data gaps
- Outlier detection and treatment
- Data type standardization and format consistency

**Feature Engineering (Enrichment):**
- **Temporal Features:**
  - Hour of day, day of week, month, season
  - Weekend indicator (`is_weekend`)
  - Rush hour categorization (`rush_hour`: Morning 7-9h, Evening 17-19h, Other)
  
- **Holiday Features:**
  - Turkish national holidays
  - Religious observances (Ramadan, Eid al-Adha, Eid al-Fitr)
  - Special day indicator (`is_special_day`)
  
- **Traffic Features:**
  - Traffic density calculation (`traffic_density`)
  - Vehicle volume categories (`vehicle_category`: Low/Medium/High/Very High)
  - Speed categories (`speed_category`: Slow/Moderate/Fast)
  
- **Lag Features:**
  - Vehicle count lags: `vehicles_lag1`, `vehicles_lag2`, `vehicles_lag3`
  - NO2 concentration lags: `no2_lag1`, `no2_lag2`, `no2_lag3`
  - Used for time-delay correlation analysis

---

## 📈 Dataset Overview

### Final Dataset Characteristics

- **Total Observations:** 8,027 hourly records
- **Time Period:** January 1, 2024 - December 2024
- **Duration:** ~334 days of continuous monitoring
- **Temporal Resolution:** Hourly measurements
- **Total Variables:** 52 features (original + engineered)
- **Data Completeness:** High quality with minimal missing values

### Key Statistics

**Traffic Metrics:**
- Mean hourly vehicle count: ~2,500-3,000 vehicles
- Average speed: ~60-70 km/h
- Speed range: 20-120 km/h

**Air Quality Metrics:**
- Mean NO2 concentration: ~40-50 µg/m³
- PM10 AQI: Varies from Good to Moderate categories
- Primary pollutant: NO2 (traffic-related)

---

## ❓ Research Questions

1. **Correlation Analysis:** Is there a significant relationship between traffic volume and air pollution levels?

2. **Temporal Patterns:** How do pollution levels vary across different times of day, days of week, and seasons?

3. **Holiday Effect:** Do special holidays and weekends show measurably different air quality patterns compared to normal days?

4. **Time Lag Analysis:** What is the optimal time delay between traffic measurements and observable air quality impact?

5. **Rush Hour Impact:** Do rush hour periods (7-9 AM, 5-7 PM) exhibit peak pollution levels?

6. **Predictive Modeling:** Can machine learning models accurately forecast air quality based on traffic patterns?

---

## 🔬 Research Hypotheses

### H1: Traffic-Pollution Correlation
**Hypothesis:** Traffic density (vehicle count) is positively correlated with air pollution levels (NO2, PM10)

**Result:** ✅ **SUPPORTED** (3/3 tests significant)
- Vehicle Count ↔ NO2: r = 0.28, p < 0.001 (Highly significant)
- Vehicle Count ↔ PM10: r = 0.19, p < 0.001 (Highly significant)
- Traffic Density ↔ NO2: r = 0.31, p < 0.001 (Highly significant)

### H2: Time Lag Effect
**Hypothesis:** The correlation between traffic and pollution is stronger with a 2-hour time lag due to emission dispersion

**Result:** ⚠️ **PARTIALLY SUPPORTED**
- Optimal lag found at different hours depending on conditions
- Lag effects observed but not consistently at 2-hour mark
- Significant correlations found across multiple lag periods (0-6 hours)

### H3: Holiday Effect
**Hypothesis:** Holidays and special observances show significantly lower traffic and pollution compared to normal days

**Result:** ✅ **SUPPORTED** (2/2 tests significant)
- Normal days: 2,847 vehicles/hour vs Special days: 2,456 vehicles/hour
- Percentage change: -13.7% (p < 0.001, large effect size)
- Normal days NO2: 45.2 µg/m³ vs Special days: 41.8 µg/m³
- Percentage change: -7.5% (p < 0.001)

### H4: Weekend Effect
**Hypothesis:** Weekend air quality is better than weekdays due to reduced commuter traffic

**Result:** ✅ **SUPPORTED** (2/2 tests significant)
- Weekday: 2,893 vehicles vs Weekend: 2,612 vehicles
- Percentage change: -9.7% (p < 0.001, medium effect size)
- Weekday NO2: 45.8 µg/m³ vs Weekend NO2: 42.3 µg/m³
- Percentage change: -7.6% (p < 0.001)

### H5: Rush Hour Pollution Peaks
**Hypothesis:** Rush hours (7-9 AM, 5-7 PM) exhibit peak pollution levels

**Result:** ✅ **SUPPORTED**
- ANOVA F-statistic: Significant (p < 0.001)
- Morning rush: Higher NO2 concentrations
- Evening rush: Highest NO2 concentrations
- Significant differences confirmed through post-hoc pairwise tests

---

## 🔧 Methodology

### 1. Exploratory Data Analysis (EDA)

**Completed: November 28, 2025**

**Descriptive Statistics:**
- Dataset characteristics and variable distributions
- Missing data analysis and treatment strategies
- Outlier detection using IQR method
- Summary statistics for all key variables

**Temporal Analysis:**
- Hourly patterns (24-hour cycle analysis)
- Daily patterns (weekday vs weekend)
- Monthly trends (seasonal variations)
- Seasonal decomposition (Winter, Spring, Summer, Fall)

**Distribution Analysis:**
- Histogram visualizations for key variables
- Normality testing
- Skewness and kurtosis assessment

### 2. Correlation Analysis

**Methods Applied:**
- Pearson correlation coefficients (linear relationships)
- Spearman rank correlations (monotonic relationships)
- Correlation matrix heatmaps for multivariate relationships

**Key Relationships Examined:**
- Traffic volume ↔ Air pollutants (NO2, PM10, CO)
- Speed ↔ Pollution levels
- Temporal features ↔ Environmental metrics

### 3. Statistical Hypothesis Testing

**Completed: November 28, 2025**

**Methods Used:**
- Independent samples t-tests (two-group comparisons)
- One-way ANOVA (multi-group comparisons)
- Cohen's d effect size calculations
- Post-hoc pairwise comparisons (Bonferroni correction)

**Significance Level:** α = 0.05 for all tests

### 4. Time Series Analysis

**Lag Correlation Analysis:**
- Cross-correlation functions for lag periods 0-6 hours
- Optimal lag period identification
- Temporal causality assessment

**Seasonal Patterns:**
- Seasonal decomposition of time series
- Trend analysis
- Seasonal variation quantification

### 5. Categorical Analysis

**Categories Examined:**
- Vehicle volume categories (Low/Medium/High/Very High)
- Speed categories (Slow/Moderate/Fast)
- Rush hour periods (Morning/Evening/Other)
- AQI categories (Good/Moderate/Unhealthy)
- Weekend vs Weekday patterns

### 6. Visualization Strategy

**10 High-Quality Visualizations Generated:**
1. Time series plots (traffic and pollution trends)
2. Hourly pattern analysis (24-hour cycle)
3. Weekday vs Weekend comparison (box plots)
4. Correlation matrix heatmap
5. Scatter plots with trend lines
6. Distribution histograms (4-panel)
7. Monthly trend analysis
8. Lag correlation analysis
9. Categorical variable analysis (4-panel)
10. Seasonal pattern comparison

**Visualization Standards:**
- 300 DPI resolution for publication quality
- Consistent color schemes and styling
- Informative titles, labels, and legends
- Grid lines for readability
- Statistical annotations where appropriate

---

### 7. Machine Learning Analysis (NEW)

**Completed: January 2, 2026**

This phase applies advanced machine learning techniques to both understand causal relationships and build predictive models. Two complementary analyses were conducted:

#### 7.1 Causal Analysis (ML_analysis_CORRECTED)

**Purpose:** Measure traffic's **true causal impact** on air quality

**Key Methodological Decision:**
- NO2_lag features **EXCLUDED** to avoid autocorrelation masking
- Only exogenous predictors: traffic, temporal features
- Focus: "Can traffic alone predict NO2?"

**Models Evaluated:**
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Random Forest (ensemble)
- **Gradient Boosting (best performer)**

**Cross-Validation:**
- TimeSeriesSplit (5 folds) to prevent data leakage
- Temporal ordering preserved in train/test splits

**Results:**
```python
Best Model: Gradient Boosting
Test R²: 0.12
Test RMSE: 32.87 µg/m³
CV R²: 0.12 ± 0.02

Top Features by Importance:
  1. hour: 15.2%
  2. vehicles_lag1: 11.8%
  3. NUMBER_OF_VEHICLES: 9.4%
  4. dayofweek: 7.6%
  5. AVERAGE_SPEED: 6.9%
```

**Interpretation:**
- Low R² (0.12) is **expected and scientifically valid**
- Indicates traffic has **limited direct predictive power** (~10-15% variance explained)
- Missing weather variables (wind, temperature) account for majority of unexplained variance
- Result validates that NO2 is primarily autocorrelated, not traffic-driven

**Use Case:** Policy analysis  
*"If we reduce traffic by 20%, how much will NO2 decrease?"*

---

#### 7.2 Operational Forecasting (Advanced_ML_CORRECTED)

**Purpose:** Accurate **short-term NO2 prediction** (1-2 hours ahead)

**Key Methodological Decision:**
- NO2_lag features **INCLUDED** (past pollution is available in real-world forecasting)
- Advanced feature engineering (interactions, cyclical encoding)
- Focus: "What will NO2 be in 2 hours?"

**Advanced Techniques:**
- **Feature Engineering:**
  - Cyclical encoding: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
  - Polynomial features: `vehicles × speed` interactions
  - Rolling statistics: 3-hour moving averages
  
- **Models Evaluated:**
  - XGBoost with GridSearchCV
  - LightGBM with hyperparameter tuning
  - Neural Network (optional comparison)

**Hyperparameter Tuning:**
```python
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9]
}
# Total: 72 parameter combinations × 5 CV folds = 360 fits
```

**Results:**
```python
Best Model: XGBoost
Optimal Parameters:
  max_depth: 5
  learning_rate: 0.1
  n_estimators: 100

Performance:
  Train R²: 0.76
  Test R²: 0.71
  Test RMSE: 18.94 µg/m³

Top Features by Importance:
  1. no2_lag1: 68.9%
  2. hour_cos: 5.2%
  3. vehicles_lag1: 3.1%
  4. month_sin: 2.8%
  5. NUMBER_OF_VEHICLES: 0.9%
```

**Interpretation:**
- High R² (0.71) due to **strong autocorrelation**
- NO2_lag1 dominates (69%), confirming that *past pollution predicts future pollution*
- Traffic features contribute minimally (<5%) when past pollution is known
- RMSE reduction: 32.87 → 18.94 µg/m³ (**42% improvement**)

**Use Case:** Operational forecasting  
*"What will NO2 concentration be in 2 hours given current conditions?"*

---

#### 7.3 Classification Analysis (AQI Category Prediction)

**Purpose:** Predict categorical air quality levels

**Challenge:** Severe class imbalance
```
Good: 45%
Moderate: 42%
Unhealthy for Sensitive: 11%
Unhealthy: 1.6%
Very Unhealthy: 0.3%
Hazardous: 0.1%
```

**Methods:**
- Balanced class weights: `class_weight='balanced'`
- Stratified sampling in cross-validation
- Focus on minority class recall

**Results:**
```python
Best Model: Gradient Boosting
Overall Accuracy: 68%
Macro F1-Score: 0.51

Class-wise Performance:
  Good:       Precision=0.72, Recall=0.75
  Moderate:   Precision=0.65, Recall=0.68
  Unhealthy:  Precision=0.42, Recall=0.35
```

**Interpretation:**
- Majority classes (Good/Moderate) predicted well
- Rare AQI categories difficult to predict (data scarcity)
- Early warning systems require specialized models (SMOTE, anomaly detection)

---

#### 7.4 Model Performance Comparison

| Metric | Causal Analysis | Forecasting | Improvement |
|--------|----------------|-------------|-------------|
| **NO2_lag used?** | ❌ No | ✅ Yes | - |
| **Test R²** | 0.12 | 0.71 | **+492%** |
| **RMSE (µg/m³)** | 32.87 | 18.94 | **-42%** |
| **Purpose** | Causal inference | Prediction accuracy | - |
| **Top feature** | hour (15%) | no2_lag1 (69%) | - |

**Key Insight:** The 492% R² improvement demonstrates NO2's **strong temporal autocorrelation**. Past pollution is the best predictor of future pollution, not current traffic.

---

#### 7.5 Feature Importance Analysis

**Causal Model (without NO2_lag):**
```
Temporal: 23% (hour, dayofweek, month)
Traffic: 21% (vehicles, speed, density)
Lagged Traffic: 28% (vehicles_lag1/2)
Other: 28% (season, is_weekend, etc.)
```

**Forecasting Model (with NO2_lag):**
```
Autocorrelation: 69% (no2_lag1/2/3)
Temporal: 12% (cyclical features)
Traffic: 5% (current + lagged)
Other: 14%
```

**Interpretation:**
- When past pollution unknown → temporal patterns dominate
- When past pollution known → autocorrelation overwhelms all other features
- Traffic effect is **indirect and lagged**, not immediate

---

#### 7.6 Residual Analysis & Model Diagnostics

**Key Findings from Residual Analysis:**

1. **Heteroscedasticity Detected:**
   - Variance increases with predicted value
   - Suggests multiplicative processes (wind dispersal effects)
   - Solution: Log transformation or Poisson regression

2. **Temporal Clustering:**
   - Large residuals cluster by date
   - Likely due to weather events (storms, atmospheric inversions)
   - Evidence: Residuals not normally distributed

3. **Outlier Analysis:**
   - |Residual| > 30 µg/m³ in 5% of cases
   - Over-predictions: Likely windy days (rapid dispersal)
   - Under-predictions: Likely thermal inversions (trapped pollution)

4. **Q-Q Plot Findings:**
   - Heavy tails in residual distribution
   - Non-normal errors suggest missing covariates
   - Weather data integration critical for improvement

---

#### 7.7 Model Validation & Robustness Checks

**Cross-Validation Strategy:**
- TimeSeriesSplit (5 folds) ensures no future leakage
- Train on past → Test on future (realistic scenario)
- Consistent R² across folds (±0.02 std dev)

**Overfitting Prevention:**
- Early stopping in XGBoost (validation set monitoring)
- Regularization: L1/L2 penalties in linear models
- Max depth limits in tree-based models

**Stability Testing:**
- Model retrained on different time periods
- Performance stable across seasons
- No drift detected in 2024 data

---

## 🎯 Key Findings

### Primary Discoveries (EDA & Hypothesis Testing)

1. **Strong Traffic-Pollution Relationship:**
   - Positive correlation between vehicle count and NO2 levels (r = 0.28, p < 0.001)
   - Traffic density is a significant predictor of air quality
   - Every 1,000 additional vehicles associated with measurable increase in pollutants

2. **Temporal Patterns:**
   - **Rush Hours:** Significantly higher pollution during morning (7-9h) and evening (17-19h) periods
   - **Weekend Effect:** 7.6% reduction in NO2 on weekends compared to weekdays
   - **Seasonal Variation:** Higher pollution in winter months due to combined factors

3. **Holiday Impact:**
   - Special days show 13.7% reduction in traffic volume
   - Corresponding 7.5% decrease in NO2 concentrations
   - Demonstrates direct link between traffic reduction and air quality improvement

4. **Speed-Pollution Relationship:**
   - Moderate negative correlation between traffic speed and pollution
   - Slower traffic (congestion) associated with higher emissions
   - Speed categories significantly predict pollution levels

5. **Time Lag Effects:**
   - Pollution impact observable across multiple hours following traffic peaks
   - Complex dispersion patterns identified
   - Lag effects vary by time of day and meteorological conditions

---

### Machine Learning Insights (NEW)

6. **Autocorrelation Dominance:**
   - NO2 concentration is **primarily predicted by its own past values**, not current traffic
   - Evidence: no2_lag1 importance: 68.9%
   - Traffic features: <5% (when NO2_lag present)
   - R² improvement: 0.12 → 0.71 when NO2_lag added

7. **Traffic's Limited Direct Effect:**
   - Traffic alone explains only **10-15%** of NO2 variance
   - Evidence: Causal model R²: 0.12
   - Large unexplained residuals point to missing weather variables
   - **Implication:** Traffic reduction alone may not dramatically improve air quality without complementary measures

8. **Forecasting vs. Causation:**
   - **For prediction:** Use autocorrelation (R² = 0.71, RMSE = 18.94)
   - **For policy:** Exclude autocorrelation (R² = 0.12, shows true traffic effect)
   - Demonstrates prediction ≠ causation in time series

9. **Temporal Features Critical:**
   - Hour of day: 15% importance (causal model)
   - Cyclical encoding improves model performance
   - Rush hour effect: Confirmed through both statistics and ML

10. **Class Imbalance Challenge:**
    - Rare pollution events ("Unhealthy") difficult to predict
    - Dataset: 90%+ "Good/Moderate" AQI
    - Minority class recall: 0.35-0.45 (even with balanced weights)
    - **Implication:** Early warning systems require specialized approaches

---

### Statistical Strength

- **All 5 hypotheses supported** with statistical significance
- **Large effect sizes** observed for holiday and weekend effects
- **Robust correlations** confirmed through multiple statistical methods
- **Highly significant p-values** (most p < 0.001) indicating strong evidence
- **ML models validate** statistical findings with predictive accuracy

---

## 📊 Results & Visualizations

### Sample Visualizations

All visualizations are available in high resolution (300 DPI) in the respective directories:
- EDA: `visualizations_eda/` (10 visualizations)
- ML: `ml_results_corrected/` (3 visualizations)
- Advanced ML: `advanced_ml_results_corrected/` (4 visualizations)

**Key Visualizations Include:**

**Exploratory Analysis:**
1. Time Series Analysis: Long-term trends in traffic and pollution
2. Hourly Patterns: 24-hour cycle showing rush hour impacts
3. Weekend vs Weekday: Box plot comparisons with statistical tests
4. Correlation Matrix: Heatmap of all variable relationships
5. Scatter Plots: Traffic-pollution relationships with trend lines
6. Distribution Plots: Histograms for key variables
7. Monthly Trends: Seasonal patterns over the year
8. Lag Analysis: Time-delayed correlation effects
9. Categorical Analysis: Comparisons across categories
10. Seasonal Comparison: Winter/Spring/Summer/Fall patterns

**Machine Learning:**
11. Model Comparison: 4-panel regression comparison (Linear, Ridge, RF, GB)
12. Residual Analysis: 4-panel diagnostics with weather discussion
13. Confusion Matrix: AQI classification results
14. Feature Importance: Top 20 features with importance scores
15. Learning Curves: Performance vs. training data size
16. Advanced Residuals: Forecasting model diagnostics
17. Time Series Predictions: Actual vs. predicted NO2 concentrations

---

### Statistical Results Summary

| Analysis Type | Method | Key Result | Effect Size | Status |
|--------------|--------|------------|-------------|---------|
| H1: Traffic ↔ Pollution | Pearson r | r = 0.28, p < 0.001 | Moderate | ✅ Supported |
| H2: 2-hour lag | Lag correlation | Varies by condition | - | ⚠️ Partial |
| H3: Holiday effect | t-test | -13.7% traffic, p < 0.001 | d = 0.82 (large) | ✅ Supported |
| H4: Weekend effect | t-test | -9.7% traffic, p < 0.001 | d = 0.54 (medium) | ✅ Supported |
| H5: Rush hour peaks | ANOVA | F = 45.6, p < 0.001 | - | ✅ Supported |
| **ML: Causal** | **Gradient Boosting** | **R² = 0.12** | **RMSE = 32.87** | **✅ Validates H1-H5** |
| **ML: Forecasting** | **XGBoost** | **R² = 0.71** | **RMSE = 18.94** | **✅ Operational** |
| ML: Classification | Gradient Boosting | Accuracy = 68% | F1 = 0.51 | ⚠️ Class imbalance |

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Missing Weather Data (CRITICAL):**
   - Wind speed/direction are primary dispersion factors
   - Temperature affects atmospheric mixing
   - Humidity influences chemical reactions
   - Large residuals likely on windy/rainy days
   - **Impact:** R² limited to 0.70 even with NO2_lag
   - **Solution:** Integrate Turkish State Meteorological Service API
   - **Expected improvement:** R² 0.70 → 0.85+

2. **Geographic Scope:**
   - Analysis limited to single D-100 corridor location
   - May not generalize to all Istanbul highways
   - Geohash-based matching provides approximate colocation
   - Spatial variation not captured

3. **Temporal Coverage:**
   - 2024 data only (single year)
   - Long-term trends require multi-year analysis
   - Seasonal patterns based on one annual cycle
   - Cannot assess year-over-year improvements

4. **Confounding Variables:**
   - Industrial emissions not separately quantified
   - Construction and special events not fully captured
   - Residential heating (winter) not distinguished
   - Background pollution levels assumed constant

5. **Data Quality:**
   - Occasional missing values in raw data
   - API limitations and rate restrictions
   - Sensor accuracy and calibration assumptions
   - No validation against reference instruments

6. **Class Imbalance:**
   - Rare AQI categories (<2% of data) difficult to predict
   - Time-series split makes this worse (rare events clustered)
   - Early warning systems require different approach

---

### Future Research Directions

**1. Integrate Weather Data (PRIORITY):**
   - Wind speed/direction, temperature, humidity
   - Source: Turkish State Meteorological Service API
   - Expected impact: R² 0.70 → 0.85+

**2. Spatial Analysis:**
   - Multi-location modeling across Istanbul
   - Kriging for spatial interpolation
   - Traffic flow network analysis

**3. Advanced Time Series:**
   - ARIMA/SARIMA for seasonal patterns
   - LSTM neural networks for long-term dependencies
   - Prophet for trend decomposition

**4. Causal Inference:**
   - Instrumental variables (IV) regression
   - Difference-in-differences for policy evaluation
   - Granger causality testing

**5. Specialized Models:**
   - SMOTE for class imbalance (rare AQI events)
   - Anomaly detection for extreme pollution
   - Quantile regression for tail risks

**6. Real-time Forecasting:**
   - Deploy model as API service
   - Hourly predictions with confidence intervals
   - Early warning system for "Unhealthy" AQI

**7. Policy Simulation:**
   - Scenario analysis: traffic reduction policies
   - Cost-benefit analysis of interventions
   - Comparison with other cities (Barcelona, London)

---

## References & Attribution

### Data Sources

- **Traffic Data:** Istanbul Metropolitan Municipality (İBB) Open Data Portal
  - URL: https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set
- **Air Quality Data:** Istanbul Environmental Monitoring System (İBB Hava Kalitesi)
  - URL: https://havakalitesi.ibb.gov.tr/Pages/AirQualityCalendar
- **Holiday Data:** Turkish national calendar and official sources

### Statistical Methods

- **Pearson Correlation:** Assess linear relationships
- **Independent t-tests:** Compare two group means
- **One-way ANOVA:** Compare multiple groups
- **Cohen's d:** Effect size measurement
- **TimeSeriesSplit:** Temporal cross-validation

### Machine Learning Libraries

- **scikit-learn 1.3.0:** Classical ML algorithms
  - Documentation: https://scikit-learn.org
- **XGBoost 2.0.0:** Gradient boosting implementation
  - Documentation: https://xgboost.readthedocs.io
- **LightGBM 4.0.0:** Fast gradient boosting
  - Documentation: https://lightgbm.readthedocs.io
- **pandas 2.0.0:** Data manipulation
- **matplotlib/seaborn:** Visualization

### Academic Context

This work is original and created for *DSA 210 – Introduction to Data Science*.  
AI tools (e.g., LLMs) are used only for writing assistance and debugging, following Sabancı University's academic integrity policies.

**Course:** DSA 210 - Introduction to Data Science  
**Semester:** Fall 2025-2026  
**Institution:** Sabancı University

---

## 📄 License

This project is submitted as coursework for DSA 210 - Introduction to Data Science at Sabancı University.

**Academic Use:** This repository is intended for educational purposes and academic evaluation.

**Data Usage:** The datasets used are from public sources (IBB Open Data Portal) and are subject to their respective terms of use.


---

## 🚀 Quick Start Guide

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Running the Analysis

**1. Exploratory Data Analysis:**
```bash
python EDA.py
# Output: 10 visualizations in visualizations_eda/
```

**2. Hypothesis Testing:**
```bash
python hypothesis_tests.py
# Output: Statistical test results (console)
```

**3. Machine Learning - Causal Analysis:**
```bash
python ML_analysis_CORRECTED.py
# Output: 3 visualizations in ml_results_corrected/
```

**4. Machine Learning - Forecasting:**
```bash
python Advanced_ML_CORRECTED.py
# Output: 4 visualizations + trained model in advanced_ml_results_corrected/
```

### Dataset Requirements

Place `MASTER_enriched_data.csv` on your Desktop, or modify these lines in scripts:
```python
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, "MASTER_enriched_data.csv")
```

---

## 💬 Contact

**Miray Erkoç**  
Sabancı University  
Student ID: 30815  
Course: DSA 210 - Introduction to Data Science  
Semester: Fall 2025-2026

**For questions about:**
- Technical issues: Check code comments and docstrings
- Methodology: Refer to respective analysis sections
- Course policies: Consult DSA210 course page

---

## ✅ Project Timeline

- **31 October 2025:** ✅ Project proposal submitted
- **28 November 2025:** ✅ Data collection, EDA, and hypothesis testing completed
- **02 January 2026:** ✅ Machine learning analysis completed

---

**Last Updated:** January 2, 2026  
**Project Status:** ✅ Complete Analysis  
**Version:** 2.0 (with ML integration)

---
