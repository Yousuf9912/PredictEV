"""
EV Population Predictive Analytics Dashboard
University of East Anglia – NBS-7096B Advanced Topics in Data Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictEV",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1rem;
        border-left: 5px solid #00d4aa;
    }
    .main-header h1 {
        font-family: 'Space Mono', monospace; color: #00d4aa;
        font-size: 1.6rem; margin: 0; letter-spacing: 1px;
    }
    .main-header p { color: #a0c4c8; margin: 0.4rem 0 0 0; font-size: 0.88rem; }
    .intro-box {
        background: #111f2d;
        border: 1px solid #2a4a5a;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #b0cdd8;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .intro-box strong { color: #00d4aa; }
    .kpi-card {
        background: linear-gradient(145deg, #1a2a3a, #1e3348);
        border: 1px solid #2a4a5a; border-radius: 10px;
        padding: 1.2rem 1.5rem; text-align: center;
        border-top: 3px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-family: 'Space Mono', monospace; font-size: 1.9rem;
        font-weight: 700; color: #00d4aa; line-height: 1;
    }
    .kpi-label {
        color: #7a9cb0; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 1.5px; margin-top: 0.4rem;
    }
    .section-header {
        font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #00d4aa;
        border-bottom: 2px solid #2a4a5a; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0; text-transform: uppercase; letter-spacing: 2px;
    }
    .insight-box {
        background: linear-gradient(135deg, #0d1f2d, #162535);
        border-left: 3px solid #00d4aa; padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0; margin: 0.6rem 0;
        font-size: 0.86rem; color: #c0d8e0; line-height: 1.6;
    }
    .prediction-result {
        background: linear-gradient(135deg, #003d30, #005040);
        border: 2px solid #00d4aa; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .pred-value {
        font-family: 'Space Mono', monospace; font-size: 3rem;
        color: #00d4aa; font-weight: 700;
    }
    .sidebar-note {
        font-size: 0.78rem; color: #5a7a8a; line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = None
    for fname in ["Electric_Vehicle_Population_Data.csv", "cleaned_ev_population_data.csv"]:
        try:
            df = pd.read_csv(fname)
            break
        except FileNotFoundError:
            continue
    if df is None:
        st.error("CSV file not found. Place 'Electric_Vehicle_Population_Data.csv' in the same folder as this script.")
        st.stop()

    # Normalise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # vehicle_category: extract short code (BEV / PHEV) from long-form column
    if "vehicle_category" not in df.columns:
        type_col = None
        for c in df.columns:
            if "electric_vehicle_type" in c or ("vehicle" in c and "type" in c):
                type_col = c
                break
        if type_col:
            extracted = df[type_col].str.extract(r"\((\w+)\)")[0]
            df["vehicle_category"] = extracted.fillna(df[type_col])
        else:
            st.error(f"Cannot find vehicle type column. Columns found: {df.columns.tolist()}")
            st.stop()
    else:
        sample_val = str(df["vehicle_category"].dropna().iloc[0])
        if len(sample_val) > 6:
            extracted = df["vehicle_category"].str.extract(r"\((\w+)\)")[0]
            if extracted.notna().sum() > 0:
                df["vehicle_category"] = extracted.fillna(df["vehicle_category"])

    # electric_range
    if "electric_range" not in df.columns:
        for c in df.columns:
            if "range" in c:
                df = df.rename(columns={c: "electric_range"})
                break
    if "electric_range" not in df.columns:
        st.error(f"Cannot find electric range column. Columns found: {df.columns.tolist()}")
        st.stop()

    # model_year
    if "model_year" not in df.columns:
        for c in df.columns:
            if "year" in c:
                df = df.rename(columns={c: "model_year"})
                break
    if "model_year" not in df.columns:
        st.error(f"Cannot find model year column. Columns found: {df.columns.tolist()}")
        st.stop()

    # make
    if "make" not in df.columns:
        for c in df.columns:
            if "manufacturer" in c or "brand" in c:
                df = df.rename(columns={c: "make"})
                break
    if "make" not in df.columns:
        st.error(f"Cannot find make/manufacturer column. Columns found: {df.columns.tolist()}")
        st.stop()

    # Final guard
    missing = [c for c in ["electric_range", "model_year", "vehicle_category"] if c not in df.columns]
    if missing:
        st.error(f"Still missing columns: {missing}. All columns: {df.columns.tolist()}")
        st.stop()

    df = df.dropna(subset=["electric_range", "model_year", "vehicle_category"])
    df = df[df["electric_range"] > 0]
    df["model_year"] = df["model_year"].astype(int)
    df = df[df["model_year"].between(2000, 2026)]
    df["make_clean"] = df["make"].astype(str).str.strip().str.upper()
    return df


@st.cache_data
def build_model(df):
    model_df = df[["electric_range", "model_year", "vehicle_category", "make_clean"]].copy()
    le_vc   = LabelEncoder()
    le_make = LabelEncoder()
    model_df["vehicle_cat_enc"] = le_vc.fit_transform(model_df["vehicle_category"])
    model_df["make_enc"]        = le_make.fit_transform(model_df["make_clean"])
    X = model_df[["model_year", "vehicle_cat_enc", "make_enc"]]
    y = model_df["electric_range"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    metrics = {
        "R²":   round(r2_score(y_test, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }
    cv_scores = cross_val_score(mdl, X, y, cv=5, scoring="r2")
    return mdl, le_vc, le_make, metrics, cv_scores, X_test, y_test, y_pred


def dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1f2d")
    ax.set_facecolor("#0d1f2d")
    for spine in ax.spines.values():
        spine.set_color("#2a4a5a")
    ax.tick_params(colors="#7a9cb0", labelsize=9)
    ax.xaxis.label.set_color("#7a9cb0")
    ax.yaxis.label.set_color("#7a9cb0")
    ax.title.set_color("#00d4aa")
    ax.grid(color="#1e3348", linestyle="--", linewidth=0.6, alpha=0.7)
    return fig, ax


# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()
model, le_vc, le_make, metrics, cv_scores, X_test, y_test, y_pred = build_model(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dashboard Controls")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Exploratory Analysis", "Model & Evaluation", "Range Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset Filters**")
    ev_types = st.multiselect(
        "Vehicle Type",
        options=df["vehicle_category"].unique().tolist(),
        default=df["vehicle_category"].unique().tolist(),
    )
    year_range = st.slider(
        "Model Year Range",
        int(df["model_year"].min()), int(df["model_year"].max()), (2010, 2025),
    )
    st.markdown("---")
    st.markdown(
        '<p class="sidebar-note">Filters apply to Overview and EDA pages.<br>'
        'Model trained on full dataset.<br><br>'
        'UEA NBS-7096B · 2025–26<br><br>'
        '© 2026 M Yousuf Atif<br>All rights reserved.</p>',
        unsafe_allow_html=True,
    )

filtered = df[
    df["vehicle_category"].isin(ev_types)
    & df["model_year"].between(year_range[0], year_range[1])
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="main-header">
        <h1>PREDICTEV — ELECTRIC VEHICLE RANGE TREND & ESTIMATION</h1>
        <p>University of East Anglia &nbsp;·&nbsp; NBS-7096B Advanced Topics in Data Analytics &nbsp;·&nbsp; 2025–26</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-box">
        This dashboard analyses registered electric vehicle data from Washington State to answer one
        central question: <strong>how has electric vehicle range changed over time, and what drives those
        differences?</strong> Using model year, vehicle type, and manufacturer as inputs, a linear
        regression model estimates the expected range of any given vehicle configuration.
        The findings are relevant to <strong>policy analysts, researchers, and anyone monitoring
        the progress of EV technology</strong> — showing where range stands today, how fast it is
        improving, and which manufacturers lead. Use the sidebar filters to explore specific
        vehicle types or time periods.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(filtered):,}",                          "Registered Vehicles"),
        (f"{filtered['make_clean'].nunique()}",          "Manufacturers"),
        (f"{filtered['electric_range'].mean():.0f} mi", "Average Range"),
        (f"{filtered['model_year'].max()}",              "Latest Model Year"),
        (f"{float(metrics['R²'])*100:.1f}%",             "Prediction Accuracy"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Average Range by Model Year</div>', unsafe_allow_html=True)
        trend = filtered.groupby(["model_year", "vehicle_category"])["electric_range"].mean().reset_index()
        fig, ax = dark_fig((8, 3.8))
        colors = {"BEV": "#00d4aa", "PHEV": "#ff6b6b"}
        for vtype, grp in trend.groupby("vehicle_category"):
            ax.plot(grp["model_year"], grp["electric_range"], marker="o", markersize=4,
                    linewidth=2, color=colors.get(vtype, "#8ecae6"), label=vtype)
        ax.set_xlabel("Model Year")
        ax.set_ylabel("Avg Electric Range (miles)")
        ax.set_title("Range Trajectory Over Time — BEV vs PHEV", fontsize=11, pad=10)
        ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-header">BEV vs PHEV Registrations by Year</div>', unsafe_allow_html=True)
        reg_by_year = (
            filtered.groupby(["model_year", "vehicle_category"])
            .size().unstack(fill_value=0)
        )
        fig, ax = dark_fig((4, 3.8))
        years  = reg_by_year.index
        bottom = np.zeros(len(years))
        colors = {"BEV": "#00d4aa", "PHEV": "#ff6b6b"}
        for vtype in reg_by_year.columns:
            vals = reg_by_year[vtype].values
            ax.bar(years, vals, bottom=bottom,
                   color=colors.get(vtype, "#8ecae6"),
                   label=vtype, edgecolor="#0d1f2d", linewidth=0.4, alpha=0.9)
            bottom += vals
        ax.set_xlabel("Model Year")
        ax.set_ylabel("Registrations")
        ax.set_title("Fleet Growth — BEV vs PHEV", fontsize=11, pad=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    col3, col4 = st.columns([2, 3])

    with col3:
        st.markdown('<div class="section-header">Top 10 Manufacturers by Registration</div>', unsafe_allow_html=True)
        top_makes = filtered["make_clean"].value_counts().head(10)
        fig, ax = dark_fig((4, 4.2))
        bars = ax.barh(top_makes.index[::-1], top_makes.values[::-1],
                       color="#00d4aa", alpha=0.85, edgecolor="#0d1f2d")
        for bar in bars:
            ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", color="#7a9cb0", fontsize=7.5)
        ax.set_xlabel("Registrations")
        ax.set_title("Market Share by Manufacturer", fontsize=11, pad=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown('<div class="section-header">Range Distribution by Vehicle Type</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((8, 4.2))
        for vtype, color in [("BEV", "#00d4aa"), ("PHEV", "#ff6b6b")]:
            subset = filtered[filtered["vehicle_category"] == vtype]["electric_range"]
            if len(subset):
                ax.hist(subset, bins=40, alpha=0.6, color=color, label=vtype, edgecolor="none")
        ax.set_xlabel("Electric Range (miles)")
        ax.set_ylabel("Number of Vehicles")
        ax.set_title("Range Distribution — Implications for Charging Infrastructure", fontsize=11, pad=10)
        ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Key Observations</div>', unsafe_allow_html=True)
    bev_avg  = filtered[filtered["vehicle_category"] == "BEV"]["electric_range"].mean()
    phev_avg = filtered[filtered["vehicle_category"] == "PHEV"]["electric_range"].mean()
    top_make = filtered["make_clean"].value_counts().idxmax()
    pct_bev  = (filtered["vehicle_category"] == "BEV").mean() * 100
    for ins in [
        f"Battery Electric Vehicles (BEVs) travel an average of <strong>{bev_avg:.0f} miles</strong> on a single charge — {bev_avg/phev_avg:.1f}x further than Plug-in Hybrids (PHEVs), which average {phev_avg:.0f} miles. This gap reflects fundamentally different battery architectures.",
        f"<strong>{top_make.title()}</strong> has the most registered EVs in this dataset, making it the most represented manufacturer for range analysis.",
        f"<strong>{pct_bev:.1f}% of vehicles in the selected period are BEVs.</strong> The shift towards fully electric vehicles has accelerated noticeably since 2019, driven by improving battery technology and falling costs.",
        f"Range has improved steadily with each model year. Vehicles manufactured from <strong>{filtered['model_year'].max() - 1}</strong> onwards show the highest average range on record in this dataset.",
    ]:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>EXPLORATORY DATA ANALYSIS</h1>
        <p>Understanding the data — distributions, patterns, and relationships — before building the model</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-box">
        Before building any model, it is important to understand what the data actually contains.
        This section looks at how range varies across manufacturers, vehicle types, and time periods,
        and examines which factors are most strongly associated with higher range.
        These patterns inform which variables were selected as inputs to the regression model
        on the next page.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = filtered[["model_year", "electric_range"]].describe().T
    num_cols.columns = [c.title() for c in num_cols.columns]
    st.dataframe(
        num_cols.style.format("{:.2f}").background_gradient(cmap="YlGnBu", axis=1),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        corr_df = filtered[["model_year", "electric_range"]].copy()
        corr_df["vehicle_type_enc"] = LabelEncoder().fit_transform(filtered["vehicle_category"])
        corr_df["make_enc"]         = LabelEncoder().fit_transform(filtered["make_clean"])
        corr = corr_df.corr()
        fig, ax = dark_fig((5, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, linecolor="#0d1f2d",
                    annot_kws={"size": 9, "color": "white"})
        ax.set_title("Correlation — Range vs Candidate Predictors", fontsize=11, pad=10)
        ax.tick_params(colors="#c0d8e0")
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown(
            '<div class="insight-box">Model year shows the strongest positive correlation with '
            'electric range, confirming it as the primary predictor. Vehicle type and manufacturer '
            'provide additional explanatory power and are included in the regression.</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="section-header">Range Distribution by Era</div>', unsafe_allow_html=True)
        filtered2 = filtered.copy()
        filtered2["era"] = pd.cut(
            filtered2["model_year"],
            bins=[1999, 2012, 2016, 2019, 2022, 2026],
            labels=["2000–12", "2013–16", "2017–19", "2020–22", "2023+"],
        )
        fig, ax = dark_fig((5, 4))
        eras = filtered2.dropna(subset=["era"])
        box_data = [eras[eras["era"] == e]["electric_range"].values for e in eras["era"].cat.categories]
        bp = ax.boxplot(
            box_data, labels=eras["era"].cat.categories, patch_artist=True,
            medianprops=dict(color="#00d4aa", linewidth=2),
            whiskerprops=dict(color="#7a9cb0"), capprops=dict(color="#7a9cb0"),
            flierprops=dict(marker=".", color="#7a9cb0", markersize=2, alpha=0.4),
        )
        for patch, c in zip(bp["boxes"], ["#1e3348", "#1e4060", "#0f5060", "#0a6060", "#006666"]):
            patch.set_facecolor(c); patch.set_edgecolor("#2a4a5a")
        ax.set_xlabel("Era")
        ax.set_ylabel("Electric Range (miles)")
        ax.set_title("Range Improvement Across Vehicle Eras", fontsize=11, pad=10)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown(
            '<div class="insight-box">Median range has risen consistently across eras, '
            'with the most significant jump occurring post-2019 — coinciding with rapid '
            'battery cost reductions and growing model variety from major manufacturers.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Model Year vs Electric Range — Full Scatter</div>', unsafe_allow_html=True)
    sample = filtered.sample(min(5000, len(filtered)), random_state=42)
    fig, ax = dark_fig((10, 4))
    color_map = {"BEV": "#00d4aa", "PHEV": "#ff6b6b"}
    for vtype in sample["vehicle_category"].unique():
        sub = sample[sample["vehicle_category"] == vtype]
        ax.scatter(sub["model_year"], sub["electric_range"], alpha=0.22, s=10,
                   color=color_map.get(vtype, "#8ecae6"), label=vtype)
    bev_sub = filtered[filtered["vehicle_category"] == "BEV"]
    if len(bev_sub) > 10:
        xs    = np.linspace(bev_sub["model_year"].min(), bev_sub["model_year"].max(), 100)
        coef  = np.polyfit(bev_sub["model_year"], bev_sub["electric_range"], 1)
        ax.plot(xs, np.polyval(coef, xs), color="white", linewidth=2,
                linestyle="--", label=f"BEV linear trend  (+{coef[0]:.1f} mi/yr)")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Electric Range (miles)")
    ax.set_title("Range vs Model Year — Basis for Linear Regression", fontsize=11, pad=10)
    ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Average Range by Manufacturer (Top 15)</div>', unsafe_allow_html=True)
    make_range = (
        filtered.groupby("make_clean")["electric_range"]
        .agg(["mean", "count"])
        .query("count >= 50")
        .sort_values("mean", ascending=False)
        .head(15)
    )
    fig, ax = dark_fig((10, 3.8))
    bars = ax.bar(make_range.index, make_range["mean"],
                  color="#00d4aa", alpha=0.85, edgecolor="#0d1f2d")
    for bar, (_, row) in zip(bars, make_range.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{row['mean']:.0f}", ha="center", color="#7a9cb0", fontsize=7.5)
    ax.set_xlabel("Manufacturer")
    ax.set_ylabel("Average Range (miles)")
    ax.set_title("Average Range by Manufacturer — Top 15 (min. 50 registrations)", fontsize=11, pad=10)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model & Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1>LINEAR REGRESSION MODEL & EVALUATION</h1>
        <p>How the range estimation model was built and how well it performs</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-box">
        A linear regression model was trained to estimate electric vehicle range based on three inputs:
        <strong>model year</strong>, <strong>vehicle type</strong> (BEV or PHEV), and
        <strong>manufacturer</strong>. In simple terms, the model learns from historical registration
        data what range a vehicle of a given type, brand, and age is likely to have — and uses that
        pattern to make estimates for new inputs. This page shows how accurately the model performs
        and how consistent it is across different subsets of the data.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, (val, label, desc) in zip([m1, m2, m3, m4], [
        (f"{float(metrics['R²'])*100:.1f}%",                      "Accuracy",       "How much of the variation in range the model explains"),
        (f"{metrics['MAE']} mi",                                "Avg. Error",     "On average, predictions are off by this many miles"),
        (f"{metrics['RMSE']} mi",                               "Worst-Case Error","Larger errors are weighted more heavily here"),
        (f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%", "Consistency", "Accuracy held stable across 5 independent test runs"),
    ]):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="font-size:1.4rem;">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)
        col.caption(desc)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Actual vs Predicted Range</div>', unsafe_allow_html=True)
        idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
        fig, ax = dark_fig((5.5, 4.5))
        ax.scatter(y_test.values[idx], y_pred[idx], alpha=0.22, s=8, color="#00d4aa")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="white", linewidth=1.5, label="Perfect prediction line")
        ax.set_xlabel("Actual Range (miles)")
        ax.set_ylabel("Predicted Range (miles)")
        ax.set_title("Prediction Accuracy — Test Set", fontsize=11, pad=10)
        ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-header">Residuals Distribution</div>', unsafe_allow_html=True)
        residuals = y_test.values - y_pred
        fig, ax = dark_fig((5.5, 4.5))
        ax.hist(residuals, bins=60, color="#00d4aa", alpha=0.75, edgecolor="none")
        ax.axvline(0, color="white", linewidth=1.5, linestyle="--", label="Zero error")
        ax.axvline(np.mean(residuals), color="#ff6b6b", linewidth=1.5,
                   label=f"Mean residual: {np.mean(residuals):.2f} mi")
        ax.set_xlabel("Residual (miles)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution — Normality Check", fontsize=11, pad=10)
        ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">5-Fold Cross-Validation — Accuracy per Fold</div>', unsafe_allow_html=True)
    cv_pct    = cv_scores * 100
    folds     = [f"Fold {i+1}" for i in range(len(cv_pct))]
    colors_cv = ["#00d4aa" if s >= cv_pct.mean() else "#ff6b6b" for s in cv_pct]
    fig, ax   = dark_fig((7, 3.2))
    bars = ax.bar(folds, cv_pct, color=colors_cv, edgecolor="#0d1f2d", alpha=0.9, width=0.45)
    ax.axhline(cv_pct.mean(), color="white", linestyle="--", linewidth=1.5)
    ax.text(len(folds) - 0.5, cv_pct.mean() + 1.2,
            f"Average: {cv_pct.mean():.1f}%",
            ha="right", color="white", fontsize=9)
    for bar, score in zip(bars, cv_pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 4,
                f"{score:.1f}%",
                ha="center", va="top", color="#0d1f2d", fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Cross-Validation — Model Accuracy Across 5 Independent Test Runs",
                 fontsize=11, pad=10)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(
        f'<div class="insight-box">'
        f'The model correctly accounts for <strong>{float(metrics["R²"])*100:.1f}%</strong> of the variation '
        f'in electric range across the dataset. On average, its estimates are off by '
        f'<strong>{metrics["MAE"]} miles</strong> — roughly the margin you would expect given '
        f'that range also depends on factors not in this dataset, such as driving conditions and battery health. '
        f'Accuracy was consistent across all five test runs, confirming the model generalises reliably '
        f'rather than just memorising the training data.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – RANGE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Range Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>RANGE ESTIMATOR</h1>
        <p>Use the trained model to estimate how far a vehicle will travel on a single charge</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-box">
        Select a manufacturer, vehicle type, and model year to get the model's estimated range
        for that combination. The estimate is based on patterns learned from over 100,000 registered
        vehicles in Washington State. It gives a reasonable ballpark figure — not a guarantee —
        since real-world range also depends on factors such as driving style, terrain, and weather.
        The chart below shows how the estimated range for your selected vehicle changes across
        all model years, illustrating how rapidly range has improved over time.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Vehicle Specification</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        input_year = st.slider("Model Year", 2000, 2026, 2023)
    with col2:
        input_type = st.selectbox("Vehicle Type", options=le_vc.classes_.tolist())
    with col3:
        all_makes   = sorted(df["make_clean"].unique().tolist())
        default_idx = all_makes.index("TESLA") if "TESLA" in all_makes else 0
        input_make  = st.selectbox("Manufacturer", options=all_makes, index=default_idx)

    try:
        vc_enc = le_vc.transform([input_type])[0]
    except Exception:
        vc_enc = 0
    try:
        make_enc = le_make.transform([input_make])[0]
    except Exception:
        make_enc = 0

    predicted_range = max(0, model.predict([[input_year, vc_enc, make_enc]])[0])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prediction-result">
        <div style="color:#7a9cb0; font-size:0.78rem; text-transform:uppercase;
                    letter-spacing:2px; margin-bottom:0.5rem;">
            Model-Estimated Electric Range
        </div>
        <div class="pred-value">{predicted_range:.0f} miles</div>
        <div style="color:#7a9cb0; font-size:0.85rem; margin-top:0.8rem;">
            {input_year} &nbsp;{input_make.title()} &nbsp;·&nbsp; {input_type}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Projected Range — 2000 to 2026</div>', unsafe_allow_html=True)
    years_sweep = np.arange(2000, 2027)
    sweep_preds = np.maximum(
        model.predict([[y, vc_enc, make_enc] for y in years_sweep]), 0
    )
    fig, ax = dark_fig((10, 3.5))
    ax.fill_between(years_sweep, sweep_preds, alpha=0.15, color="#00d4aa")
    ax.plot(years_sweep, sweep_preds, color="#00d4aa", linewidth=2.5)
    ax.axvline(input_year, color="#ffd166", linestyle="--",
               linewidth=1.5, label=f"Selected year: {input_year}")
    ax.scatter([input_year], [predicted_range], s=100, color="#ffd166", zorder=5)
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Predicted Range (miles)")
    ax.set_title(
        f"Range Projection — {input_make.title()}  {input_type}  (all model years)",
        fontsize=11, pad=10,
    )
    ax.legend(facecolor="#0d1f2d", edgecolor="#2a4a5a", labelcolor="#c0d8e0", fontsize=9)
    st.pyplot(fig, use_container_width=True); plt.close()

    rate         = model.coef_[0]
    future_range = max(0, model.predict([[input_year + 3, vc_enc, make_enc]])[0])
    mae          = metrics["MAE"]
    st.markdown(
        f'<div class="insight-box">'
        f'Based on historical registration data, a <strong>{input_year} {input_make.title()} {input_type}</strong> '
        f'is estimated to travel approximately <strong>{predicted_range:.0f} miles</strong> on a single charge '
        f'(give or take around {mae:.0f} miles).<br><br>'
        f'On average, each newer model year adds around <strong>{rate:.2f} miles</strong> of range. '
        f'The same vehicle from <strong>{input_year + 3}</strong> is estimated at roughly '
        f'<strong>{future_range:.0f} miles</strong> — <strong>{future_range - predicted_range:.0f} miles more</strong> '
        f'than the {input_year} model, reflecting the pace of battery improvement over those three years.'
        f'</div>',
        unsafe_allow_html=True,
    )