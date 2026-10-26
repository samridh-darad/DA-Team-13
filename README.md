# DA-Team-13

## Air Quality Analysis of 26 Indian Cities (2015–2020)

This repository contains a reproducible data analytics project investigating how particulate matter, gaseous pollutants, and seasonal patterns influence urban air quality across 26 Indian cities.

### Project objective

The analysis explores the relationship between air quality indicators and environmental conditions by combining:

- data quality auditing and preprocessing
- descriptive statistics and confidence intervals
- normalization and discretization
- exploratory data analysis
- PCA and regression modeling
- clustering and association rule mining
- confirmatory hypothesis testing
- publication-style figure generation

### Dataset

The project uses the `city_day.csv` dataset, which includes daily air quality observations for multiple Indian cities. The notebook/script performs cleaning, missing-value treatment, seasonal feature engineering, and city-wise analysis to evaluate pollution trends and AQI determinants.

### Repository contents

- `analysis.py` — main data analytics pipeline
- `city_day.csv` — source dataset
- `README.md` — project overview and usage guide
- generated outputs such as figures and `results.json` after running the script

### Key analytical methods

- Missing value audit and imputation
- AQI distribution and seasonal trend analysis
- Correlation analysis and PCA
- Multiple linear and log-linear regression
- K-Means clustering
- Apriori / association rule mining
- One-way and two-way ANOVA, t-tests, and chi-square tests
- Multi-panel figure generation for reporting

### Requirements

Install the required Python libraries:

```bash
python -m pip install pandas numpy scipy matplotlib seaborn
```

### Run the analysis

From the repository root:

```bash
python analysis.py
```

This script will:

1. load and preprocess the dataset
2. run the statistical analyses
3. generate report figures in the project directory
4. save result summaries in `results.json`

### Outputs

The workflow generates visual outputs including:

- PM2.5 distribution and seasonal cycle plots
- PCA scree plot and clustering visualizations
- AQI comparisons across zones and seasons
- Apriori rule summaries

### Project theme

The study is titled:

> "Do Particulate Matter, Gaseous Emissions and Seasonal Cycles Drive Urban Air Quality? An Exploratory, Confirmatory and Association Rule Analysis of 26 Indian Cities (2015–2020)"

### Notes

This project is designed for academic / learning-oriented data analytics work and demonstrates a full end-to-end exploratory and confirmatory analysis workflow in Python.
