# DSA210_Project
# Traffic Density and Air Quality Analysis in Istanbul

**DSA210 - Introduction to Data Science | Fall 2025-2026**

**Student:** Miray Erkoc

---

## Motivation

Istanbul faces significant air quality challenges due to high traffic density. This project investigates the relationship between hourly traffic patterns and air pollution levels to understand how vehicle flow affects air quality metrics such as NO2, PM10, and CO concentrations. The insights aim to support data-driven public health policies.

---

## Data Sources

This or similar data (that will support the hypothesis) will be collected.

### Primary Datasets
- **Traffic Data:** Hourly traffic volume and speed measurements from Istanbul Metropolitan Municipality's open data portal (IBB Açık Veri Portalı) for 2024
  - Metrics: Vehicle count, average speed, min/max speed
-**Air Quality Data:** Hourly air pollution measurements retrieved via Turkish Air Quality Monitoring API for 2024
  - Metrics: NO2, PM10, CO, AQI (Air Quality Index)

### Data Enrichment
- Turkish national holidays and religious observances (Ramadan, Eid al-Adha)
- Temporal features: hour, day of week, season, weekend indicators
- AQI category classifications (Good, Moderate, Unhealthy, etc.)
- Lag features (1-3 hour delayed traffic measurements)
- Further enrichment may be made if deemed appropriate.

---

## Methodology

Data Collection Plan

This or a similar data collection phase is planned.

Phase 1: Data Acquisition

API Integration: Develop Python scripts to connect to IBB Open Data Portal and Air Quality Monitoring APIs 
Automated Collection: Schedule hourly data pulls for the entire 2024 year for designated location(s)
Data Validation: Implement error handling and data quality checks
Storage: Store raw data in CSV format with timestamps

Phase 2: Data Preparation

Merging: Combine traffic and air quality datasets by timestamp
Cleaning: Handle missing values, remove outliers, standardize formats
Feature Engineering: Create temporal features, holiday indicators, and lag variables
Enrichment: Add calculated metrics (traffic density, AQI categories)

Phase 3: Quality Assurance

Check for data completeness (target: >95% coverage)
Validate timestamp alignment between datasets
Document data collection challenges and limitation

---
Research Questions

These or similar questions are planned to be investigated.

Correlation Analysis: Is there a significant relationship between traffic volume and air pollution levels?
Temporal Patterns: How do pollution levels vary across different times of day, days of week, and seasons?
Holiday Effect: Do special holidays and weekends show measurably different air quality patterns?
Time Lag: What is the optimal time delay between traffic measurements and air quality impact?
AQI Prediction: Can traffic patterns predict air quality category transitions?


Planned Methodology

It is envisaged that these or similar methodologies will be used

1. Exploratory Data Analysis (EDA)

Descriptive statistics for all variables
Time series visualization of traffic and pollution trends
Distribution analysis and outlier detection

2. Statistical Analysis

Correlation analysis (Pearson, Spearman) between traffic and pollution metrics
Hypothesis testing for weekend vs. weekday differences
Time lag analysis to find optimal correlation delay
Seasonal decomposition of time series

3. Comparative Analysis

Holiday vs. normal day traffic and pollution comparison
Rush hour vs. off-peak hour analysis
AQI category distribution across different conditions

4. Predictive Modeling (Advanced)

It is envisaged that these or similar methodologies will be used

Feature selection based on correlation and domain knowledge
Machine learning models (Random Forest, XGBoost, etc.) for pollution prediction
Time series forecasting (ARIMA, LSTM, etc.) for air quality trends
Model evaluation and validation
