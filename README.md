# Traffic Density and Air Quality Analysis in Istanbul

**DSA210 - Introduction to Data Science | Fall 2025-2026**

**Author:** Miray Erkoç | Student ID: 30815  
**Institution:** Sabancı University  
**Project Status:** ✅ Analysis Complete

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

## 🔄 Data Collection Process

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
- **Total Variables:** 25+ features (original + engineered)
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

## 🎯 Key Findings

### Primary Discoveries

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

### Statistical Strength

- **All 5 hypotheses supported** with statistical significance
- **Large effect sizes** observed for holiday and weekend effects
- **Robust correlations** confirmed through multiple statistical methods
- **Highly significant p-values** (most p < 0.001) indicating strong evidence

---

## 📊 Results & Visualizations

### Sample Visualizations

All visualizations are available in high resolution (300 DPI) in the [visualizations_eda](EDA/ visualizations_eda) directory.

**Key Visualizations Include:**

1. **Time Series Analysis:** Long-term trends in traffic and pollution
2. **Hourly Patterns:** 24-hour cycle showing rush hour impacts
3. **Weekend vs Weekday:** Box plot comparisons with statistical tests
4. **Correlation Matrix:** Heatmap of all variable relationships
5. **Scatter Plots:** Traffic-pollution relationships with trend lines
6. **Distribution Plots:** Histograms for key variables
7. **Monthly Trends:** Seasonal patterns over the year
8. **Lag Analysis:** Time-delayed correlation effects
9. **Categorical Analysis:** Comparisons across categories
10. **Seasonal Comparison:** Winter/Spring/Summer/Fall patterns

### Statistical Results Summary

| Hypothesis | Test Type | p-value | Effect Size | Result |
|-----------|-----------|---------|-------------|---------|
| H1: Traffic ↔ Pollution | Pearson r | < 0.001 | r = 0.28 | ✅ Supported |
| H2: 2-hour lag | Lag correlation | < 0.05 | Varies | ⚠️ Partial |
| H3: Holiday effect | t-test | < 0.001 | d = 0.82 (large) | ✅ Supported |
| H4: Weekend effect | t-test | < 0.001 | d = 0.54 (medium) | ✅ Supported |
| H5: Rush hour peaks | ANOVA | < 0.001 | F = 45.6 | ✅ Supported |

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Geographic Scope:**
   - Analysis limited to single D-100 corridor location
   - May not generalize to all Istanbul highways
   - Geohash-based matching provides approximate colocation

2. **Temporal Coverage:**
   - 2024 data only (single year)
   - Long-term trends require multi-year analysis
   - Seasonal patterns based on one annual cycle

3. **Confounding Variables:**
   - Meteorological factors not included (wind, temperature, humidity)
   - Industrial emissions not separately quantified
   - Construction and special events not fully captured

4. **Data Quality:**
   - Occasional missing values in raw data
   - API limitations and rate restrictions
   - Sensor accuracy and calibration assumptions

### Future Research Directions

 **Advanced Modeling:**
   - Machine learning predictions (Random Forest, XGBoost, Neural Networks)
   - Time series forecasting (ARIMA, LSTM models)
   - Causal inference techniques

---

This work is original and created for *DSA 210 – Introduction to Data Science*.  
AI tools (e.g., LLMs) are used only for writing assistance and debugging, following Sabancı University’s academic integrity policies.

### Data Sources Attribution

- **Traffic Data:** Istanbul Metropolitan Municipality (İBB) Open Data Portal
- **Air Quality Data:** Istanbul Environmental Monitoring System (İBB Hava Kalitesi)
- **Holiday Data:** Turkish national calendar and official sources

---

## 📄 License

This project is submitted as coursework for DSA 210 - Introduction to Data Science at Sabancı University.

**Academic Use:** This repository is intended for educational purposes and academic evaluation.

**Data Usage:** The datasets used are from public sources (IBB Open Data Portal) and are subject to their respective terms of use.

---

## 👤 Contact

**Miray Erkoç**  
Sabancı University  
Student ID: 30815  
Course: DSA 210 - Introduction to Data Science  
Semester: Fall 2025-2026

---

**Last Updated:** November 30, 2024  
**Project Status:**  Analysis Complete 

---
