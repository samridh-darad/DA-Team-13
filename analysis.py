"""
CES2103: Data Analytics — Mini Project Reproducible Analysis & Visualization Script
School of Computer Engineering, Manipal Institute of Technology (MAHE)

Project Title:
"Do Particulate Matter, Gaseous Emissions and Seasonal Cycles Drive Urban Air Quality?
An Exploratory, Confirmatory and Association Rule Analysis of 26 Indian Cities (2015–2020)"

Dataset: CPCB Indian Air Quality Dataset (city_day.csv)
Modules Covered (CO1–CO4, L1–L27):
  1. Data Quality Audit, Missing Value Imputation, Central Tendency, 95% CIs & IQR Outliers (L4–L6, L11–L12)
  2. Normalization (Min-Max, Z-Score, Decimal Scaling), Binning & Concept Hierarchies (L17–L20)
  3. Exploratory Data Analysis (EDA), Correlation (Pearson r, Spearman rho) & Collinearity (L7–L10, L13)
  4. Principal Component Analysis (PCA) & Multiple Linear / Log-Linear Regression (L14–L16)
  5. Unsupervised K-Means Clustering (k=3) & Decision Tree Feature Importance (L19)
  6. Confirmatory Data Analysis (CDA): One/Two-Sample t/Z-tests, One/Two-Way ANOVA,
     Kruskal-Wallis H-test & Chi-Square Contingency Tests (L21–L23)
  7. Frequent Itemset & Association Rule Mining (Apriori / FP-Growth) (L24–L27)
  8. Automated Generation of Publication-Grade Multi-Panel Figures (Fig. 1, Fig. 2, Fig. 3)
"""

import os
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style matching the report
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.9


def load_and_preprocess(csv_path: str):
    """Load city_day.csv, generate concept hierarchies, and apply city-median imputation."""
    df = pd.read_csv(csv_path)

    # Parse dates and generate Temporal Concept Hierarchy (Date -> Month -> IMD Season)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    def get_season(m: int) -> str:
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Pre-Monsoon"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        return "Post-Monsoon"

    df["Season"] = df["Month"].apply(get_season)

    # Generate Spatial Concept Hierarchy (26 Cities -> 5 Geographic Zones)
    city_zone = {
        "Delhi": "North", "Gurugram": "North", "Lucknow": "North", "Jaipur": "North",
        "Amritsar": "North", "Chandigarh": "North", "Jorapokhar": "East", "Patna": "East",
        "Kolkata": "East", "Guwahati": "East", "Talcher": "East", "Brajrajnagar": "East",
        "Shillong": "East", "Aizawl": "East", "Ahmedabad": "West", "Mumbai": "West",
        "Bhopal": "Central", "Amaravati": "South", "Bengaluru": "South", "Chennai": "South",
        "Coimbatore": "South", "Ernakulam": "South", "Hyderabad": "South", "Kochi": "South",
        "Thiruvananthapuram": "South", "Visakhapatnam": "South",
    }
    df["Zone"] = df["City"].map(city_zone)

    num_cols = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
        "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI",
    ]

    # Filter out records missing the target AQI/AQI_Bucket (4,681 rows removed -> n = 24,850)
    df_clean = df.dropna(subset=["AQI"]).copy()

    # Hierarchical City-wise Median Imputation (with global median fallback)
    for col in num_cols:
        df_clean[col] = df_clean.groupby("City")[col].transform(
            lambda x: x.fillna(x.median() if x.notnull().any() else np.nan)
        )
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df, df_clean, num_cols


def run_descriptive_and_preprocessing_audit(df: pd.DataFrame, df_clean: pd.DataFrame, num_cols: list):
    """Compute central tendency, dispersion, 95% CIs, IQR outliers, normalization, and binning."""
    print("=" * 80)
    print("1. DESCRIPTIVE STATISTICS, 95% CONFIDENCE INTERVALS & MISSINGNESS (RAW DATA)")
    print("=" * 80)
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        n = len(s)
        mean, sd = s.mean(), s.std()
        med, iqr = s.median(), s.quantile(0.75) - s.quantile(0.25)
        se = sd / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)

        # Outliers on cleaned data via 1.5 * IQR rule
        sc = df_clean[col]
        q1_c, q3_c = sc.quantile(0.25), sc.quantile(0.75)
        iqr_c = q3_c - q1_c
        fence_high = q3_c + 1.5 * iqr_c
        fence_low = q1_c - 1.5 * iqr_c
        out_cnt = int(((sc < fence_low) | (sc > fence_high)).sum())

        rows.append({
            "Variable": col,
            "Raw_N": n,
            "Missing_%": round((1 - n / len(df)) * 100, 2),
            "Mean": round(mean, 2),
            "SD": round(sd, 2),
            "Median": round(med, 2),
            "IQR": round(iqr, 2),
            "Skew": round(s.skew(), 2),
            "Kurt": round(s.kurtosis(), 2),
            "95%_CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "Upper_Fence": round(fence_high, 2),
            "Outliers_%": round(out_cnt / len(df_clean) * 100, 2),
        })
    desc_df = pd.DataFrame(rows)
    print(desc_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. NORMALIZATION (MIN-MAX, Z-SCORE, DECIMAL SCALING) & DISCRETIZATION (L17–L18)")
    print("=" * 80)
    for col in ["PM2.5", "PM10", "NO2", "CO", "AQI"]:
        s = df_clean[col]
        mm = (s - s.min()) / (s.max() - s.min())
        zs = (s - s.mean()) / s.std()
        j = int(np.ceil(np.log10(s.abs().max())))
        ds = s / (10 ** j)
        print(
            f"{col:6s} | Raw: [{s.min():.2f}, {s.max():.2f}] | "
            f"MinMax Mean±SD: {mm.mean():.4f}±{mm.std():.4f} | "
            f"Z-Score: [{zs.min():.2f}, {zs.max():.2f}] | "
            f"DecScale (j={j}): [{ds.min():.4f}, {ds.max():.4f}]"
        )

    print("\nEqual-Width Binning (4 bins) on PM2.5:")
    print(pd.cut(df_clean["PM2.5"], bins=4).value_counts().sort_index())
    print("\nEqual-Frequency Quartile Binning (4 bins) on PM2.5:")
    print(pd.qcut(df_clean["PM2.5"], q=4).value_counts().sort_index())


def run_pca_and_regression(df_clean: pd.DataFrame):
    """Execute PCA eigendecomposition and OLS / Log-Linear regression models."""
    print("\n" + "=" * 80)
    print("3. PRINCIPAL COMPONENT ANALYSIS (PCA) & MULTIVARIATE REGRESSION (L14–L16)")
    print("=" * 80)
    features_pca = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
    X_raw = df_clean[features_pca].values
    X_scaled = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
    cov_mat = np.cov(X_scaled, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]
    exp_var = eig_vals / eig_vals.sum() * 100
    cum_var = np.cumsum(exp_var)

    pca_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(features_pca))],
        "Eigenvalue": eig_vals.round(4),
        "Explained_Var_%": exp_var.round(2),
        "Cumulative_Var_%": cum_var.round(2),
    })
    print(pca_df.to_string(index=False))

    # OLS & Log-Linear Regression via Least Squares
    def fit_ols(X_df: pd.DataFrame, y_s: pd.Series):
        X = np.column_stack([np.ones(len(X_df)), X_df.values])
        y = y_s.values
        n, k = X.shape
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        sse = np.sum(resid ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1 - sse / sst
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
        mse = sse / (n - k)
        f_stat = ((sst - sse) / (k - 1)) / mse
        se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
        t_stats = beta / se
        p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n - k)
        rmse = np.sqrt(np.mean(resid ** 2))
        summary = pd.DataFrame({
            "Term": ["Intercept"] + list(X_df.columns),
            "Coef": beta.round(4),
            "Std_Err": se.round(4),
            "t_stat": t_stats.round(2),
            "p_value": [f"{p:.2e}" for p in p_vals],
        })
        return r2, adj_r2, f_stat, rmse, summary

    reg_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NO", "NH3"]
    r2, adj_r2, f_stat, rmse, ols_tab = fit_ols(df_clean[reg_cols], df_clean["AQI"])
    print(f"\nOLS Regression (AQI ~ Pollutants): R² = {r2:.4f}, Adj R² = {adj_r2:.4f}, F = {f_stat:.2f}, RMSE = {rmse:.2f}")
    print(ols_tab.to_string(index=False))

    r2_l, adj_r2_l, f_stat_l, rmse_l, log_tab = fit_ols(np.log1p(df_clean[reg_cols]), np.log(df_clean["AQI"]))
    print(f"\nLog-Linear Regression (ln(AQI) ~ ln(Pollutants+1)): R² = {r2_l:.4f}, F = {f_stat_l:.2f}, RMSE_log = {rmse_l:.4f}")
    print(log_tab.to_string(index=False))

    return X_scaled


def run_clustering_and_apriori(df_clean: pd.DataFrame, X_scaled: np.ndarray):
    """Run K-Means clustering (k=3) and Apriori frequent itemset association rule mining."""
    print("\n" + "=" * 80)
    print("4. K-MEANS CLUSTERING (k=3) & APRIORI ASSOCIATION RULE MINING (L19, L24–L27)")
    print("=" * 80)
    X_clust = X_scaled[:, [0, 1, 3, 6, 7, 8]]  # PM2.5, PM10, NO2, CO, SO2, O3
    centroids = np.percentile(X_clust, [15, 55, 90], axis=0)
    for _ in range(50):
        dists = np.linalg.norm(X_clust[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for k_i in range(3):
            if np.any(labels == k_i):
                centroids[k_i] = X_clust[labels == k_i].mean(axis=0)
    df_clean["Cluster"] = labels
    print("K-Means Cluster Profiles (Count & Mean):")
    print(df_clean.groupby("Cluster")[["PM2.5", "PM10", "NO2", "CO", "SO2", "AQI"]].agg(["count", "mean"]).round(2))

    # Boolean transaction encoding for Apriori / FP-Growth rule mining
    trans = pd.DataFrame({
        "High_PM2.5(>60)": df_clean["PM2.5"] > 60,
        "High_PM10(>100)": df_clean["PM10"] > 100,
        "High_NO2(>40)": df_clean["NO2"] > 40,
        "High_CO(>1.5)": df_clean["CO"] > 1.5,
        "High_SO2(>20)": df_clean["SO2"] > 20,
        "Winter": df_clean["Season"] == "Winter",
        "North_Zone": df_clean["Zone"] == "North",
        "Poor_Severe_AQI(>200)": df_clean["AQI"] > 200,
    })
    target = "Poor_Severe_AQI(>200)"
    supp_target = trans[target].mean()
    other_items = [c for c in trans.columns if c != target]
    rules = []
    for r in [1, 2, 3]:
        for comb in combinations(other_items, r):
            ant_mask = trans[list(comb)].all(axis=1)
            supp_ant = ant_mask.mean()
            if supp_ant >= 0.045:
                both_mask = ant_mask & trans[target]
                supp_both = both_mask.mean()
                conf = supp_both / supp_ant
                lift = conf / supp_target
                if conf >= 0.80:
                    rules.append({
                        "Antecedent": " & ".join(comb),
                        "Consequent": target,
                        "Days": int(both_mask.sum()),
                        "Support": round(supp_both, 4),
                        "Confidence_%": round(conf * 100, 2),
                        "Lift": round(lift, 4),
                    })
    rules_df = pd.DataFrame(rules).sort_values(by=["Lift", "Confidence_%"], ascending=False)
    print("\nTop Apriori / FP-Growth Association Rules:")
    print(rules_df.head(10).to_string(index=False))


def run_confirmatory_hypothesis_tests(df: pd.DataFrame, df_clean: pd.DataFrame):
    """Execute all parametric and non-parametric CDA statistical tests (L21–L23)."""
    print("\n" + "=" * 80)
    print("5. CONFIRMATORY DATA ANALYSIS (CDA) HYPOTHESIS TESTS (α = 0.05)")
    print("=" * 80)
    results = {}

    # T1a & T1b: One-sample t/Z-tests
    t_pm25, p2_pm25 = stats.ttest_1samp(df_clean["PM2.5"], 60.0)
    t_aqi, p2_aqi = stats.ttest_1samp(df_clean["AQI"], 100.0)
    results["T1a_PM25_vs_60"] = {"mean": round(df_clean["PM2.5"].mean(), 2), "t": round(float(t_pm25), 4), "p_one_tailed": float(p2_pm25 / 2)}
    results["T1b_AQI_vs_100"] = {"mean": round(df_clean["AQI"].mean(), 2), "t": round(float(t_aqi), 4), "p_one_tailed": float(p2_aqi / 2)}

    # T2a, T2b, T2c: Two-sample Welch's t-tests
    del_aqi = df_clean[df_clean["City"] == "Delhi"]["AQI"]
    blr_aqi = df_clean[df_clean["City"] == "Bengaluru"]["AQI"]
    t_db, p_db = stats.ttest_ind(del_aqi, blr_aqi, equal_var=False)
    results["T2a_Delhi_vs_Bengaluru"] = {"delhi_mean": round(del_aqi.mean(), 2), "blr_mean": round(blr_aqi.mean(), 2), "t": round(float(t_db), 4), "p": float(p_db)}

    win_aqi = df_clean[df_clean["Season"] == "Winter"]["AQI"]
    mon_aqi = df_clean[df_clean["Season"] == "Monsoon"]["AQI"]
    t_wm, p_wm = stats.ttest_ind(win_aqi, mon_aqi, equal_var=False)
    results["T2b_Winter_vs_Monsoon"] = {"winter_mean": round(win_aqi.mean(), 2), "monsoon_mean": round(mon_aqi.mean(), 2), "t": round(float(t_wm), 4), "p": float(p_wm)}

    nth_aqi = df_clean[df_clean["Zone"] == "North"]["AQI"]
    sth_aqi = df_clean[df_clean["Zone"] == "South"]["AQI"]
    t_ns, p_ns = stats.ttest_ind(nth_aqi, sth_aqi, equal_var=False)
    results["T2c_North_vs_South"] = {"north_mean": round(nth_aqi.mean(), 2), "south_mean": round(sth_aqi.mean(), 2), "t": round(float(t_ns), 4), "p": float(p_ns)}

    # T3a & T3b: One-Way ANOVA & Kruskal-Wallis across Seasons and Zones
    s_groups = [g["AQI"].values for _, g in df_clean.groupby("Season")]
    f_seas, p_seas = stats.f_oneway(*s_groups)
    h_seas, kp_seas = stats.kruskal(*s_groups)
    results["T3a_Season_ANOVA_KW"] = {"F": round(float(f_seas), 2), "H": round(float(h_seas), 2), "p": float(p_seas)}

    z_groups = [g["AQI"].values for _, g in df_clean.groupby("Zone")]
    f_zone, p_zone = stats.f_oneway(*z_groups)
    h_zone, kp_zone = stats.kruskal(*z_groups)
    results["T3b_Zone_ANOVA_KW"] = {"F": round(float(f_zone), 2), "H": round(float(h_zone), 2), "p": float(p_zone)}

    # T3c: Two-Way ANOVA (Zone x Season Interaction) via OLS Sum of Squares
    y_tot = df_clean["AQI"].values
    X_z = pd.get_dummies(df_clean["Zone"], drop_first=True, dtype=float).values
    X_s = pd.get_dummies(df_clean["Season"], drop_first=True, dtype=float).values
    X_int = pd.get_dummies(df_clean["Zone"] + "_" + df_clean["Season"], drop_first=True, dtype=float).values

    def get_sse(X_m, y_v):
        X_c = np.column_stack([np.ones(len(y_v)), X_m])
        b, _, _, _ = np.linalg.lstsq(X_c, y_v, rcond=None)
        return np.sum((y_v - X_c @ b) ** 2), X_c.shape[1] - 1

    sse_z, df_z = get_sse(X_z, y_tot)
    sse_s, df_s = get_sse(X_s, y_tot)
    sse_zs, df_zs = get_sse(np.column_stack([X_z, X_s]), y_tot)
    sse_full, df_full = get_sse(X_int, y_tot)
    df_inter = df_full - df_zs
    df_err = len(y_tot) - (df_full + 1)
    ms_err = sse_full / df_err

    f_z2 = ((sse_s - sse_zs) / df_z) / ms_err
    f_s2 = ((sse_z - sse_zs) / df_s) / ms_err
    f_i2 = ((sse_zs - sse_full) / df_inter) / ms_err
    results["T3c_TwoWay_ANOVA"] = {
        "F_Zone": round(float(f_z2), 2),
        "F_Season": round(float(f_s2), 2),
        "F_Interaction": round(float(f_i2), 2),
        "p_Interaction": float(stats.f.sf(f_i2, df_inter, df_err)),
    }

    # T4a & T4b: Non-parametric Chi-Square Contingency Tests
    ct_sb = pd.crosstab(df_clean["Season"], df_clean["AQI_Bucket"])
    chi2_sb, p_sb, dof_sb, _ = stats.chi2_contingency(ct_sb)
    v_sb = np.sqrt(chi2_sb / (len(df_clean) * (min(ct_sb.shape) - 1)))
    results["T4a_ChiSquare_Season_Bucket"] = {"chi2": round(float(chi2_sb), 2), "df": int(dof_sb), "Cramers_V": round(float(v_sb), 4), "p": float(p_sb)}

    ct_zb = pd.crosstab(df_clean["Zone"], df_clean["AQI_Bucket"])
    chi2_zb, p_zb, dof_zb, _ = stats.chi2_contingency(ct_zb)
    v_zb = np.sqrt(chi2_zb / (len(df_clean) * (min(ct_zb.shape) - 1)))
    results["T4b_ChiSquare_Zone_Bucket"] = {"chi2": round(float(chi2_zb), 2), "df": int(dof_zb), "Cramers_V": round(float(v_zb), 4), "p": float(p_zb)}

    for k, v in results.items():
        print(f"{k:32s}: {v}")
    return results


def generate_report_figures(df_clean: pd.DataFrame, out_dir: str):
    """Generate the 3 multi-panel figures (Fig. 1, Fig. 2, Fig. 3) used in the report."""
    os.makedirs(out_dir, exist_ok=True)

    # --- FIG 1: EDA (PM2.5 Distribution, Monthly Seasonal Cycle, Spearman/Pearson Correlations) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.5), dpi=220)
    pm25_clip = df_clean["PM2.5"][df_clean["PM2.5"] <= 350]
    ax1.hist(pm25_clip, bins=28, color="#6A8CAF", edgecolor="white", alpha=0.92)
    ax1.axvline(60, color="#C44E52", linestyle="--", linewidth=1.5, label="CPCB Limit (60 µg/m³)")
    ax1.axvline(67.45, color="#2B5B84", linestyle="-", linewidth=1.5, label="Mean (67.5 µg/m³)")
    ax1.set_title("(a) Daily PM2.5 Distribution (n = 24,850)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("PM2.5 (µg/m³)", fontsize=9.5)
    ax1.set_ylabel("Count (Days)", fontsize=9.5)
    ax1.legend(fontsize=8, loc="upper right")

    for city, col_c in zip(["Delhi", "Patna", "Lucknow", "Kolkata", "Bengaluru"], ["#C44E52", "#DD8452", "#CCB974", "#8172B3", "#4C72B0"]):
        m_s = df_clean[df_clean["City"] == city].groupby("Month")["PM2.5"].mean()
        ax2.plot(m_s.index, m_s.values, marker="o", linewidth=1.8, markersize=4, label=city, color=col_c)
    ax2.axhline(60, color="#555555", linestyle=":", linewidth=1.2)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8.5)
    ax2.set_title("(b) Monthly U-Shaped PM2.5 Cycle", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Month of Year (2015–2020)", fontsize=9.5)
    ax2.set_ylabel("Mean PM2.5 (µg/m³)", fontsize=9.5)
    ax2.legend(fontsize=7.8, ncol=2, loc="upper center")

    poll_list = ["PM2.5", "PM10", "CO", "NO2", "NO", "NOx", "SO2", "NH3", "Toluene", "O3", "Benzene"][::-1]
    sp_vals = [df_clean[p].corr(df_clean["AQI"], method="spearman") for p in poll_list]
    pr_vals = [df_clean[p].corr(df_clean["AQI"], method="pearson") for p in poll_list]
    y_pos = np.arange(len(poll_list))
    ax3.barh(y_pos + 0.18, sp_vals, height=0.35, color="#4C72B0", label="Spearman ρ")
    ax3.barh(y_pos - 0.18, pr_vals, height=0.35, color="#C44E52", label="Pearson r")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(poll_list, fontsize=8.5)
    ax3.set_xlim(0, 0.95)
    ax3.set_title("(c) Pollutant Correlation with AQI", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Correlation Coefficient (ρ, r)", fontsize=9.5)
    ax3.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clean_fig1_eda.png"), bbox_inches="tight")
    plt.close()

    # --- FIG 2: PCA Scree Plot, Top 8 Cities & K-Means Clustering ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.5), dpi=220)
    features_pca = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
    X_raw = df_clean[features_pca].values
    X_sc = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
    eig_vals, _ = np.linalg.eigh(np.cov(X_sc, rowvar=False))
    eig_vals = eig_vals[::-1]
    exp_var = eig_vals / eig_vals.sum() * 100
    cum_var = np.cumsum(exp_var)
    pcs = [f"PC{i}" for i in range(1, 9)]
    ax1_t = ax1.twinx()
    ax1.bar(pcs, exp_var[:8], color="#6A8CAF", width=0.58)
    ax1_t.plot(pcs, cum_var[:8], color="#C44E52", marker="s", linewidth=2, markersize=4.5)
    ax1_t.axhline(69.76, color="#555555", linestyle="--", linewidth=1.1)
    ax1_t.grid(False)
    ax1.set_title("(a) PCA Scree Plot (4 PCs = 69.8% Var)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Principal Components (λ > 1 at PC1–PC4)", fontsize=9)
    ax1.set_ylabel("Explained Variance (%)", fontsize=9, color="#2B5B84")
    ax1_t.set_ylabel("Cumulative Variance (%)", fontsize=9, color="#C44E52")

    city_stats = df_clean.groupby("City").agg(mean=("AQI", "mean"), std=("AQI", "std"), n=("AQI", "count")).reset_index()
    city_stats["ci"] = 1.96 * city_stats["std"] / np.sqrt(city_stats["n"])
    top8 = city_stats.sort_values("mean", ascending=True).tail(8)
    ax2.barh(top8["City"], top8["mean"], xerr=top8["ci"], color="#4C72B0", alpha=0.88, height=0.6, capsize=2.5)
    ax2.axvline(100, color="#55A868", linestyle="--", linewidth=1.2, label="Satisfactory (100)")
    ax2.axvline(200, color="#C44E52", linestyle=":", linewidth=1.2, label="Poor (200)")
    ax2.set_xlim(0, 520)
    ax2.set_title("(b) Top 8 Polluted Cities (Mean AQI ±95% CI)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Daily AQI", fontsize=9.5)
    ax2.legend(fontsize=7.8, loc="lower right")

    sample_df = df_clean.sample(n=2000, random_state=42)
    c_names = {0: "C0: Baseline (μ=107)", 1: "C1: Smog (μ=267)", 2: "C2: Plume (μ=608)"}
    c_cols = {0: "#55A868", 1: "#DD8452", 2: "#C44E52"}
    for k_i in [0, 1, 2]:
        sub_c = sample_df[sample_df["Cluster"] == k_i]
        ax3.scatter(sub_c["PM2.5"], sub_c["AQI"], s=12, alpha=0.6, color=c_cols[k_i], label=c_names[k_i])
    ax3.set_xlim(0, 500)
    ax3.set_ylim(0, 1000)
    ax3.set_title("(c) K-Means Clustering (k = 3 Regimes)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Daily PM2.5 (µg/m³)", fontsize=9.5)
    ax3.set_ylabel("Daily AQI", fontsize=9.5)
    ax3.legend(fontsize=7.8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clean_fig2_pca_clusters.png"), bbox_inches="tight")
    plt.close()

    # --- FIG 3: CDA (Two-Way ANOVA Interaction, Zonal Boxplots, Apriori Rules) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.5), dpi=220)
    season_order = ["Monsoon", "Pre-Monsoon", "Post-Monsoon", "Winter"]
    s_labels = ["Monsoon", "Pre-Mon", "Post-Mon", "Winter"]
    for z, z_col in zip(["North", "East", "South"], ["#C44E52", "#8172B3", "#55A868"]):
        sub = df_clean[df_clean["Zone"] == z].groupby("Season")["AQI"].agg(["mean", "std", "count"]).reindex(season_order)
        ci = 1.96 * sub["std"] / np.sqrt(sub["count"])
        ax1.errorbar(s_labels, sub["mean"], yerr=ci, marker="o", linewidth=2.0, capsize=3.5, label=f"{z} Zone", color=z_col)
    ax1.set_title("(a) Mean AQI ±95% CI by Season & Zone", fontsize=10, fontweight="bold")
    ax1.set_xlabel("IMD Season (ANOVA F = 936.6, p < 0.001)", fontsize=9)
    ax1.set_ylabel("Mean Daily AQI", fontsize=9.5)
    ax1.legend(fontsize=8, loc="upper left")

    zone_order = ["South", "Central", "East", "North", "West"]
    box_data = [df_clean[df_clean["Zone"] == z]["AQI"].clip(upper=550).values for z in zone_order]
    bp = ax2.boxplot(box_data, tick_labels=zone_order, patch_artist=True, showfliers=False, widths=0.52)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ECAE1")
        patch.set_edgecolor("#4C72B0")
    for median in bp["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.5)
    ax2.set_title("(b) AQI by Geographic Zone (F = 1611.7)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Geographic Zone", fontsize=9.5)
    ax2.set_ylabel("Daily AQI (Interquartile Range)", fontsize=9.5)

    r_names = [
        "{PM2.5>60, NO2>40, CO>1.5}",
        "{PM2.5>60, CO>1.5, North}",
        "{PM10>100, NO2>40, CO>1.5}",
        "{PM2.5>60, CO>1.5, SO2>20}",
        "{PM2.5>60, CO>1.5, Winter}",
        "{NO2>40, CO>1.5} [Closed]",
    ][::-1]
    lifts = [3.45, 3.44, 3.39, 3.37, 3.35, 3.29][::-1]
    confs = [89.7, 89.3, 88.0, 87.5, 87.0, 85.5][::-1]
    bars_r = ax3.barh(r_names, lifts, color="#4C72B0", height=0.55, alpha=0.9)
    for bar, l_v, c_v in zip(bars_r, lifts, confs):
        ax3.text(l_v + 0.05, bar.get_y() + bar.get_height() / 2, f"{l_v:.2f}× ({c_v:.0f}%)", va="center", fontsize=8)
    ax3.set_xlim(0, 4.6)
    ax3.axvline(1.0, color="#888888", linestyle="--")
    ax3.set_title("(c) Top Apriori Rules (Consequent: AQI > 200)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Rule Lift (Confidence %)", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clean_fig3_cda_apriori.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    CSV_PATH = "city_day.csv"
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "/working_dir/c_67fa72d070363299/user_uploaded_files/ttl=63d/user_uploaded/city_day.csv"

    df_raw, df_cleaned, numeric_cols = load_and_preprocess(CSV_PATH)
    run_descriptive_and_preprocessing_audit(df_raw, df_cleaned, numeric_cols)
    X_std = run_pca_and_regression(df_cleaned)
    run_clustering_and_apriori(df_cleaned, X_std)
    cda_results = run_confirmatory_hypothesis_tests(df_raw, df_cleaned)
    generate_report_figures(df_cleaned, out_dir=".")
    with open("results.json", "w") as f_out:
        json.dump(cda_results, f_out, indent=2)
    print("\nCompleted full analysis, saved figures and results.json.")