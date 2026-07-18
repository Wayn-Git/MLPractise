
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# ════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
# 📊 Machine Learning — Complete Step-by-Step Notebook

> **Covers:** Data Exploration → Data Analysis → EDA → Preprocessing → \
Classical ML (Classification & Regression) → Cross Validation → Model Evaluation → Hyperparameter Tuning → Pipelines → Saving

---

| # | Section |
|---|---------|
| 0 | Environment Setup & Imports |
| 1 | Data Exploration |
| 2 | Data Analysis |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Preprocessing & Feature Engineering |
| 5 | Classical ML — Classification |
| 6 | Classical ML — Regression |
| 7 | Cross Validation |
| 8 | Model Evaluation |
| 9 | Hyperparameter Tuning |
| 10 | End-to-End ML Pipeline |
| 11 | Model Saving & Loading |
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 0 — SETUP
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 🛠️ Section 0 — Environment Setup & Imports"))

cells.append(new_code_cell("""\
# ── Uncomment to install missing packages ──
# !pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm optuna shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, pickle, json, time
warnings.filterwarnings("ignore")

# ── Sklearn — Data ──
from sklearn.datasets import (load_iris, load_breast_cancer, load_diabetes,
                               make_classification, make_regression, make_blobs)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold, LeaveOneOut,
    GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve,
    cross_validate)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, PolynomialFeatures, PowerTransformer)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                        RFE, RFECV, mutual_info_classif)

# ── Sklearn — Metrics ──
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error)

# ── Sklearn — Classifiers ──
from sklearn.linear_model import (LogisticRegression, SGDClassifier, RidgeClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier)

# ── Sklearn — Regressors ──
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet, HuberRegressor)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor, ExtraTreesRegressor)

# ── Sklearn — Unsupervised ──
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOST = True
except ImportError:
    HAS_BOOST = False
    print("⚠️  XGBoost/LightGBM not found — boosting cells will be skipped.")

# ── Reproducibility ──
SEED = 42
np.random.seed(SEED)

# ── Plot Style ──
plt.rcParams.update({
    "figure.facecolor": "#111111",
    "axes.facecolor":   "#1c1c1c",
    "axes.edgecolor":   "#555",
    "axes.labelcolor":  "#ddd",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#ddd",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "lines.linewidth":  2,
    "font.family":      "monospace",
    "figure.dpi":       110,
})
ACCENT = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#e91e63"]

print("✅  All packages imported successfully")
print(f"   NumPy {np.__version__}  |  Pandas {pd.__version__}")
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA EXPLORATION
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 🔎 Section 1 — Data Exploration"))

cells.append(new_markdown_cell("""\
### 1.1 Load Datasets
We use **two real datasets** throughout this notebook:
- **Breast Cancer** → Classification task (malignant vs benign)
- **Diabetes** → Regression task (disease progression)
"""))

cells.append(new_code_cell("""\
# ── Classification Dataset: Breast Cancer ──
bc_data = load_breast_cancer(as_frame=True)
df_clf  = bc_data.frame.copy()
df_clf.rename(columns={"target": "label"}, inplace=True)

# ── Regression Dataset: Diabetes ──
diab    = load_diabetes(as_frame=True)
df_reg  = diab.frame.copy()
df_reg.rename(columns={"target": "disease_progress"}, inplace=True)

print("=" * 50)
print("CLASSIFICATION DATASET — Breast Cancer")
print("=" * 50)
print(f"  Shape   : {df_clf.shape}")
print(f"  Classes : {dict(zip(bc_data.target_names, [(df_clf.label==i).sum() for i in [0,1]]))}")
print(f"  Features: {list(df_clf.columns[:5])} ...")

print()
print("=" * 50)
print("REGRESSION DATASET — Diabetes")
print("=" * 50)
print(f"  Shape   : {df_reg.shape}")
print(f"  Target  : mean={df_reg.disease_progress.mean():.1f},",
      f"std={df_reg.disease_progress.std():.1f},",
      f"range=[{df_reg.disease_progress.min():.0f}, {df_reg.disease_progress.max():.0f}]")
print(f"  Features: {list(df_reg.columns[:5])} ...")
"""))

cells.append(new_markdown_cell("### 1.2 First Look"))

cells.append(new_code_cell("""\
print("── HEAD ──")
display(df_clf.head())

print("── TAIL ──")
display(df_clf.tail(3))

print("── SAMPLE (random) ──")
display(df_clf.sample(5, random_state=SEED))
"""))

cells.append(new_code_cell("""\
# ── Shape, dtypes, memory ──
print(f"Shape       : {df_clf.shape[0]:,} rows × {df_clf.shape[1]} columns")
print(f"Memory      : {df_clf.memory_usage(deep=True).sum() / 1024:.1f} KB")
print()
print("── Dtypes ──")
print(df_clf.dtypes.value_counts().to_string())
print()
print("── Column list ──")
for i, col in enumerate(df_clf.columns, 1):
    print(f"  {i:2d}. {col}")
"""))

cells.append(new_code_cell("""\
# ── Missing values audit ──
def missing_report(df, name="Dataset"):
    miss  = df.isnull().sum()
    pct   = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": miss, "Missing %": pct})
    report = report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    if report.empty:
        print(f"✅  [{name}] No missing values")
    else:
        print(f"⚠️  [{name}] Missing value report:")
        display(report)

missing_report(df_clf, "Breast Cancer")
missing_report(df_reg, "Diabetes")
"""))

cells.append(new_code_cell("""\
# ── Inject synthetic missing values so preprocessing has something to do ──
np.random.seed(SEED)
df_clf_messy = df_clf.copy()
df_reg_messy = df_reg.copy()

# 5% missingness in first 10 numeric columns
for df, cols in [(df_clf_messy, df_clf.columns[:10]),
                  (df_reg_messy, df_reg.columns[:5])]:
    for col in cols:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan

# Add a synthetic categorical column
df_clf_messy["tissue_type"] = np.random.choice(["Epithelial","Stromal","Mixed"],
                                                  size=len(df_clf_messy))
df_reg_messy["age_group"]   = np.random.choice(["Young","Middle","Senior"],
                                                  size=len(df_reg_messy))

missing_report(df_clf_messy, "Breast Cancer (messy)")
missing_report(df_reg_messy, "Diabetes (messy)")
"""))

cells.append(new_markdown_cell("### 1.3 Statistical Summary"))

cells.append(new_code_cell("""\
print("── Descriptive Statistics — Classification ──")
desc_clf = df_clf.describe().T
desc_clf["cv%"] = (desc_clf["std"] / desc_clf["mean"].abs() * 100).round(1)
display(desc_clf.style
        .background_gradient(subset=["mean","std"], cmap="plasma")
        .background_gradient(subset=["cv%"], cmap="RdYlGn_r")
        .format("{:.3f}"))
"""))

cells.append(new_code_cell("""\
print("── Descriptive Statistics — Regression ──")
display(df_reg.describe().T.style
        .background_gradient(cmap="viridis")
        .format("{:.3f}"))
"""))

cells.append(new_code_cell("""\
# ── Unique value counts for every column ──
print("── Unique Values per Column ──")
uniq = pd.DataFrame({
    "Unique":  df_clf.nunique(),
    "Example": [str(df_clf[c].dropna().unique()[:3].tolist()) for c in df_clf.columns]
})
display(uniq)
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 📐 Section 2 — Data Analysis"))

cells.append(new_markdown_cell("### 2.1 Target Analysis"))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── Classification: class balance ──
counts = df_clf["label"].value_counts()
axes[0].bar(bc_data.target_names, counts.values,
            color=[ACCENT[0], ACCENT[1]], edgecolor="k", linewidth=0.6)
axes[0].set_title("Class Distribution\\n(Breast Cancer)")
axes[0].set_ylabel("Count")
for bar, val in zip(axes[0].patches, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 f"{val}\\n({val/len(df_clf)*100:.1f}%)", ha="center", fontsize=9)

# ── Classification: pie ──
axes[1].pie(counts.values, labels=bc_data.target_names,
            autopct="%1.1f%%", startangle=90,
            colors=[ACCENT[0], ACCENT[1]], wedgeprops=dict(edgecolor="k"))
axes[1].set_title("Class Proportions")

# ── Regression: target distribution ──
axes[2].hist(df_reg["disease_progress"], bins=30, color=ACCENT[2],
             edgecolor="k", linewidth=0.4, alpha=0.85)
axes[2].axvline(df_reg["disease_progress"].mean(), color=ACCENT[3],
                linestyle="--", linewidth=2, label="Mean")
axes[2].axvline(df_reg["disease_progress"].median(), color=ACCENT[0],
                linestyle="--", linewidth=2, label="Median")
axes[2].set_title("Target Distribution\\n(Diabetes Progression)")
axes[2].set_xlabel("Disease Progress"); axes[2].set_ylabel("Count")
axes[2].legend()

plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 2.2 Feature-Target Relationship"))

cells.append(new_code_cell("""\
# ── Top correlated features vs. classification target ──
corr_with_target = df_clf.corr()["label"].drop("label").abs().sort_values(ascending=False)
top10_clf = corr_with_target.head(10)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(top10_clf.index[::-1], top10_clf.values[::-1],
               color=ACCENT[2], edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, top10_clf.values[::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("|Pearson Correlation with Target|")
ax.set_title("Top 10 Features Correlated with Cancer Label")
ax.set_xlim(0, 1)
plt.tight_layout(); plt.show()
"""))

cells.append(new_code_cell("""\
# ── Top correlated features vs. regression target ──
corr_reg = df_reg.corr()["disease_progress"].drop("disease_progress").abs().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# Bar of correlations
axes[0].barh(corr_reg.index[::-1], corr_reg.values[::-1],
             color=ACCENT[4], edgecolor="k", linewidth=0.5)
axes[0].set_xlabel("|Correlation with Target|")
axes[0].set_title("Feature Correlations — Diabetes")

# Scatter: top feature vs target
top_feat = corr_reg.index[0]
axes[1].scatter(df_reg[top_feat], df_reg["disease_progress"],
                alpha=0.5, s=20, color=ACCENT[1])
m, b = np.polyfit(df_reg[top_feat], df_reg["disease_progress"], 1)
x_line = np.linspace(df_reg[top_feat].min(), df_reg[top_feat].max(), 100)
axes[1].plot(x_line, m*x_line + b, color=ACCENT[0], linewidth=2, label=f"y={m:.1f}x+{b:.1f}")
axes[1].set_xlabel(top_feat); axes[1].set_ylabel("Disease Progress")
axes[1].set_title(f"Scatter: {top_feat} vs Target")
axes[1].legend()

plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 2.3 Statistical Tests"))

cells.append(new_code_cell("""\
from scipy import stats

print("── t-Test: Feature Means between Classes ──\\n")
results = []
feat_cols = [c for c in df_clf.columns if c not in ["label","tissue_type"]]
for col in feat_cols[:10]:
    grp0 = df_clf[df_clf["label"]==0][col].dropna()
    grp1 = df_clf[df_clf["label"]==1][col].dropna()
    t_stat, p_val = stats.ttest_ind(grp0, grp1)
    results.append({"Feature": col,
                    "Mean (Malignant)": grp0.mean(),
                    "Mean (Benign)":    grp1.mean(),
                    "t-statistic":      t_stat,
                    "p-value":          p_val,
                    "Significant":      "✅" if p_val < 0.05 else "❌"})

display(pd.DataFrame(results).set_index("Feature")
          .style.format({
              "Mean (Malignant)": "{:.3f}",
              "Mean (Benign)":    "{:.3f}",
              "t-statistic":      "{:.3f}",
              "p-value":          "{:.2e}"})
          .background_gradient(subset=["p-value"], cmap="RdYlGn"))
"""))

cells.append(new_code_cell("""\
# ── Normality test (Shapiro-Wilk on first 5 features) ──
print("── Shapiro-Wilk Normality Test (p>0.05 → normal) ──\\n")
norm_results = []
for col in feat_cols[:5]:
    stat, p = stats.shapiro(df_clf[col].dropna().sample(min(200, len(df_clf)), random_state=SEED))
    norm_results.append({"Feature": col, "W-stat": stat, "p-value": p,
                         "Normal?": "✅" if p > 0.05 else "❌ Skewed"})
display(pd.DataFrame(norm_results).set_index("Feature"))

# ── Correlation between features ──
print("\\n── Variance Inflation Factor (Multicollinearity) ──")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
try:
    X_vif = add_constant(df_clf[feat_cols[:8]].dropna())
    vif_df = pd.DataFrame({
        "Feature": feat_cols[:8],
        "VIF":     [variance_inflation_factor(X_vif.values, i+1) for i in range(8)]
    }).set_index("Feature").sort_values("VIF", ascending=False)
    display(vif_df.style.background_gradient(cmap="Reds").format("{:.2f}"))
except Exception:
    print("  (statsmodels not installed — skipping VIF)")
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 3 — EDA
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 📈 Section 3 — Exploratory Data Analysis (EDA)"))

cells.append(new_markdown_cell("### 3.1 Correlation Heatmap"))

cells.append(new_code_cell("""\
# ── Full correlation matrix ──
corr_matrix = df_clf[feat_cols].corr()

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.3,
            annot_kws={"size": 7}, ax=ax, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title("Full Feature Correlation Matrix — Breast Cancer", fontsize=14, pad=12)
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 3.2 Feature Distributions by Class"))

cells.append(new_code_cell("""\
# ── Histogram + KDE per class for top 12 features ──
top12 = corr_with_target.head(12).index.tolist()
class_colors = {0: ACCENT[0], 1: ACCENT[1]}
class_labels  = {0: "Malignant", 1: "Benign"}

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
for ax, col in zip(axes.flatten(), top12):
    for cls in [0, 1]:
        vals = df_clf[df_clf["label"]==cls][col].dropna()
        ax.hist(vals, bins=25, alpha=0.5, color=class_colors[cls],
                density=True, label=class_labels[cls], edgecolor="k", linewidth=0.2)
        vals.plot.kde(ax=ax, color=class_colors[cls], linewidth=2)
    ax.set_title(col, fontsize=8)
    ax.legend(fontsize=7)

plt.suptitle("Feature Distributions by Class (Top 12 Correlated Features)", fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 3.3 Box Plots & Violin Plots"))

cells.append(new_code_cell("""\
top6 = corr_with_target.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ax, col in zip(axes.flatten(), top6):
    data_by_class = [df_clf[df_clf["label"]==c][col].dropna().values for c in [0,1]]
    vp = ax.violinplot(data_by_class, positions=[0,1], showmedians=True,
                       showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor([ACCENT[0], ACCENT[1]][i])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("#f39c12"); vp["cmedians"].set_linewidth(2)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Malignant","Benign"])
    ax.set_title(col, fontsize=9)

plt.suptitle("Violin Plots — Top 6 Features", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(new_code_cell("""\
# ── Box plots ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), top6):
    data_by_class = [df_clf[df_clf["label"]==c][col].dropna().values for c in [0,1]]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color="#f39c12", linewidth=2.5),
                    whiskerprops=dict(color="#aaa", linewidth=1.5),
                    capprops=dict(color="#aaa"),
                    flierprops=dict(markerfacecolor=ACCENT[0], marker="o", ms=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], [ACCENT[0], ACCENT[1]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Malignant","Benign"])
    ax.set_title(col, fontsize=9)
plt.suptitle("Box Plots — Top 6 Features (notched = 95% CI of median)", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 3.4 Pair Plot"))

cells.append(new_code_cell("""\
top4 = corr_with_target.head(4).index.tolist()
pair_df = df_clf[top4 + ["label"]].copy()
pair_df["label"] = pair_df["label"].map({0:"Malignant", 1:"Benign"})

g = sns.pairplot(pair_df, hue="label",
                 palette={"Malignant": ACCENT[0], "Benign": ACCENT[1]},
                 diag_kind="kde", plot_kws={"alpha":0.5, "s":20})
g.fig.suptitle("Pair Plot — Top 4 Features vs Class", y=1.02, fontsize=12)
plt.show()
"""))

cells.append(new_markdown_cell("### 3.5 Outlier Detection"))

cells.append(new_code_cell("""\
# ── Z-score outlier detection ──
from scipy.stats import zscore

z_scores  = df_clf[feat_cols].apply(zscore)
outlier_mask = (z_scores.abs() > 3)
outlier_counts = outlier_mask.sum()

print("── Outliers per Feature (|Z| > 3) ──")
oc = outlier_counts[outlier_counts > 0].sort_values(ascending=False)
print(oc.to_string())
print(f"\\nTotal outlier cells : {outlier_mask.sum().sum()}")
print(f"Rows with ≥1 outlier: {outlier_mask.any(axis=1).sum()}")

# ── IQR method ──
def iqr_outliers(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    return ((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum()

iqr_counts = df_clf[feat_cols].apply(iqr_outliers).sort_values(ascending=False)
print("\\n── IQR Outliers per Feature ──")
print(iqr_counts.head(10).to_string())

# ── Heatmap of outliers ──
fig, ax = plt.subplots(figsize=(18, 4))
sns.heatmap(outlier_mask.T, cmap="Reds", cbar=False, ax=ax,
            yticklabels=True, xticklabels=False)
ax.set_xlabel("Sample index"); ax.set_title("Outlier Map (|Z|>3) — Red = Outlier")
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 3.6 Dimensionality Reduction Visuals"))

cells.append(new_code_cell("""\
from sklearn.preprocessing import StandardScaler

scaler_eda = StandardScaler()
X_scaled_eda = scaler_eda.fit_transform(df_clf[feat_cols].fillna(df_clf[feat_cols].median()))
y_eda = df_clf["label"].values

# ── PCA ──
pca_eda = PCA(n_components=2, random_state=SEED)
X_pca_eda = pca_eda.fit_transform(X_scaled_eda)

# ── t-SNE ──
tsne_eda = TSNE(n_components=2, perplexity=30, random_state=SEED)
X_tsne_eda = tsne_eda.fit_transform(X_scaled_eda)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (emb, title) in zip(axes, [
        (X_pca_eda,  f"PCA  (PC1={pca_eda.explained_variance_ratio_[0]*100:.1f}%, "
                     f"PC2={pca_eda.explained_variance_ratio_[1]*100:.1f}%)"),
        (X_tsne_eda, "t-SNE")]):
    sc = ax.scatter(emb[:,0], emb[:,1], c=y_eda,
                    cmap="RdYlGn", alpha=0.7, s=25, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    plt.colorbar(sc, ax=ax, label="0=Malignant 1=Benign")

plt.suptitle("Dimensionality Reduction Visualizations", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(new_code_cell("""\
# ── Scree plot ──
pca_full = PCA(random_state=SEED)
pca_full.fit(X_scaled_eda)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
n_comp = min(20, len(pca_full.explained_variance_ratio_))
axes[0].bar(range(1, n_comp+1), pca_full.explained_variance_ratio_[:n_comp],
            color=ACCENT[2], edgecolor="k", linewidth=0.4)
axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot")

axes[1].plot(range(1, n_comp+1),
             np.cumsum(pca_full.explained_variance_ratio_[:n_comp]),
             "o-", color=ACCENT[4])
axes[1].axhline(0.95, color=ACCENT[0], linestyle="--", label="95% threshold")
axes[1].axhline(0.99, color=ACCENT[3], linestyle="--", label="99% threshold")
axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()

plt.tight_layout(); plt.show()
n95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
n99 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.99) + 1
print(f"Components for 95% variance: {n95}")
print(f"Components for 99% variance: {n99}")
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 4 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## ⚙️ Section 4 — Preprocessing & Feature Engineering"))

cells.append(new_markdown_cell("### 4.1 Train / Validation / Test Split"))

cells.append(new_code_cell("""\
TARGET_CLF = "label"
TARGET_REG = "disease_progress"

# ── Classification split ──
X_c = df_clf_messy.drop(TARGET_CLF, axis=1)
y_c = df_clf_messy[TARGET_CLF]

X_c_train, X_c_temp, y_c_train, y_c_temp = train_test_split(
    X_c, y_c, test_size=0.30, random_state=SEED, stratify=y_c)
X_c_val, X_c_test, y_c_val, y_c_test = train_test_split(
    X_c_temp, y_c_temp, test_size=0.50, random_state=SEED, stratify=y_c_temp)

# ── Regression split ──
X_r = df_reg_messy.drop(TARGET_REG, axis=1)
y_r = df_reg_messy[TARGET_REG]

X_r_train, X_r_temp, y_r_train, y_r_temp = train_test_split(
    X_r, y_r, test_size=0.30, random_state=SEED)
X_r_val, X_r_test, y_r_val, y_r_test = train_test_split(
    X_r_temp, y_r_temp, test_size=0.50, random_state=SEED)

print("── Classification Splits ──")
print(f"  Train : {X_c_train.shape}  |  {y_c_train.mean():.3f} positive rate")
print(f"  Val   : {X_c_val.shape}    |  {y_c_val.mean():.3f} positive rate")
print(f"  Test  : {X_c_test.shape}   |  {y_c_test.mean():.3f} positive rate")

print("\\n── Regression Splits ──")
print(f"  Train : {X_r_train.shape}  |  mean target {y_r_train.mean():.2f}")
print(f"  Val   : {X_r_val.shape}    |  mean target {y_r_val.mean():.2f}")
print(f"  Test  : {X_r_test.shape}   |  mean target {y_r_test.mean():.2f}")
"""))

cells.append(new_markdown_cell("### 4.2 Build Preprocessing Pipelines"))

cells.append(new_code_cell("""\
# ── Identify column types ──
num_cols_c = X_c_train.select_dtypes(include=np.number).columns.tolist()
cat_cols_c = X_c_train.select_dtypes(exclude=np.number).columns.tolist()

num_cols_r = X_r_train.select_dtypes(include=np.number).columns.tolist()
cat_cols_r = X_r_train.select_dtypes(exclude=np.number).columns.tolist()

print(f"CLF numeric cols   ({len(num_cols_c)}): {num_cols_c[:4]} ...")
print(f"CLF categorical ({len(cat_cols_c)}): {cat_cols_c}")
print(f"REG numeric cols   ({len(num_cols_r)}): {num_cols_r}")
print(f"REG categorical ({len(cat_cols_r)}): {cat_cols_r}")
"""))

cells.append(new_code_cell("""\
# ── Numeric pipeline: median impute → robust scale ──
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  RobustScaler()),         # robust to outliers
])

# ── Categorical pipeline: mode impute → one-hot encode ──
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ── ColumnTransformer ──
def make_preprocessor(num_cols, cat_cols):
    steps = []
    if num_cols: steps.append(("num", num_pipe, num_cols))
    if cat_cols: steps.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(steps)

preprocessor_c = make_preprocessor(num_cols_c, cat_cols_c)
preprocessor_r = make_preprocessor(num_cols_r, cat_cols_r)

# ── Fit on train ONLY, transform val/test ──
X_c_tr_pp = preprocessor_c.fit_transform(X_c_train)
X_c_va_pp = preprocessor_c.transform(X_c_val)
X_c_te_pp = preprocessor_c.transform(X_c_test)

X_r_tr_pp = preprocessor_r.fit_transform(X_r_train)
X_r_va_pp = preprocessor_r.transform(X_r_val)
X_r_te_pp = preprocessor_r.transform(X_r_test)

print(f"CLF: Train {X_c_tr_pp.shape}, Val {X_c_va_pp.shape}, Test {X_c_te_pp.shape}")
print(f"REG: Train {X_r_tr_pp.shape}, Val {X_r_va_pp.shape}, Test {X_r_te_pp.shape}")
"""))

cells.append(new_markdown_cell("### 4.3 Feature Engineering"))

cells.append(new_code_cell("""\
# ── 1. Polynomial interaction features ──
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_c_poly = poly.fit_transform(X_c_tr_pp)
print(f"After polynomial interactions: {X_c_tr_pp.shape[1]} → {X_c_poly.shape[1]} features")

# ── 2. Power Transform (Yeo-Johnson) to normalize skewed features ──
pt = PowerTransformer(method="yeo-johnson", standardize=True)
X_c_pt = pt.fit_transform(X_c_tr_pp)
print(f"Power-transformed: {X_c_pt.shape}")

# ── 3. PCA reduction ──
pca = PCA(n_components=0.95, random_state=SEED)
X_c_pca = pca.fit_transform(X_c_tr_pp)
print(f"PCA (95% var): {X_c_tr_pp.shape[1]} → {X_c_pca.shape[1]} components")

# ── 4. SelectKBest ──
selector = SelectKBest(f_classif, k=15)
X_c_kbest = selector.fit_transform(X_c_tr_pp, y_c_train)
print(f"SelectKBest (k=15): {X_c_tr_pp.shape[1]} → {X_c_kbest.shape[1]}")

# ── 5. Mutual Information ──
mi_scores = mutual_info_classif(X_c_tr_pp, y_c_train, random_state=SEED)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(range(len(mi_scores)), np.sort(mi_scores)[::-1][:20][::-1],
        color=ACCENT[4], edgecolor="k", linewidth=0.4)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Top 20 Features by Mutual Information (Classification)")
plt.tight_layout(); plt.show()
"""))

cells.append(new_code_cell("""\
# ── 6. Recursive Feature Elimination (RFE) ──
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED),
          n_features_to_select=10)
rfe.fit(X_c_tr_pp, y_c_train)

print("── RFE: Selected feature indices ──")
print(f"  {np.where(rfe.support_)[0].tolist()}")
print(f"  Ranking (1=selected): {rfe.ranking_[:10].tolist()} ...")

# ── 7. KNN Imputer vs Median Imputer comparison ──
knn_imp  = KNNImputer(n_neighbors=5)
med_imp  = SimpleImputer(strategy="median")

X_raw_num = df_clf_messy[num_cols_c].copy()
X_knn  = knn_imp.fit_transform(X_raw_num)
X_med  = med_imp.fit_transform(X_raw_num)

col_idx = 0
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (data, title) in zip(axes, [(X_knn, "KNN Imputer"), (X_med, "Median Imputer")]):
    ax.hist(X_raw_num.iloc[:, col_idx].dropna(), bins=20, alpha=0.6,
            color=ACCENT[1], label="Original", density=True)
    ax.hist(data[:, col_idx], bins=20, alpha=0.5,
            color=ACCENT[0], label="Imputed", density=True)
    ax.set_title(f"{title} — {num_cols_c[col_idx]}")
    ax.legend()
plt.suptitle("Imputation Method Comparison", fontsize=12)
plt.tight_layout(); plt.show()
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 5 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 🎯 Section 5 — Classical ML: Classification"))

cells.append(new_markdown_cell("### 5.1 Logistic Regression"))

cells.append(new_code_cell("""\
# ── Standard ──
lr = LogisticRegression(max_iter=2000, random_state=SEED, C=1.0, solver="lbfgs")
lr.fit(X_c_tr_pp, y_c_train)

y_pred_lr = lr.predict(X_c_te_pp)
y_prob_lr = lr.predict_proba(X_c_te_pp)[:,1]

print("Logistic Regression")
print(f"  Accuracy  : {accuracy_score(y_c_test, y_pred_lr):.4f}")
print(f"  AUC-ROC   : {roc_auc_score(y_c_test, y_prob_lr):.4f}")
print(f"  F1-Score  : {f1_score(y_c_test, y_pred_lr):.4f}")
print(f"  Precision : {precision_score(y_c_test, y_pred_lr):.4f}")
print(f"  Recall    : {recall_score(y_c_test, y_pred_lr):.4f}")
"""))

cells.append(new_markdown_cell("### 5.2 Support Vector Machine (SVM)"))

cells.append(new_code_cell("""\
# ── Try multiple kernels ──
svm_results = {}
for kernel in ["linear", "rbf", "poly", "sigmoid"]:
    clf = SVC(kernel=kernel, probability=True, random_state=SEED, max_iter=2000)
    clf.fit(X_c_tr_pp, y_c_train)
    probs = clf.predict_proba(X_c_te_pp)[:,1]
    preds = clf.predict(X_c_te_pp)
    svm_results[kernel] = {
        "model": clf,
        "AUC":  roc_auc_score(y_c_test, probs),
        "Acc":  accuracy_score(y_c_test, preds),
        "F1":   f1_score(y_c_test, preds),
    }
    print(f"  SVM ({kernel:7s}) | Acc {svm_results[kernel]['Acc']:.4f}"
          f" | AUC {svm_results[kernel]['AUC']:.4f}"
          f" | F1 {svm_results[kernel]['F1']:.4f}")

best_kernel = max(svm_results, key=lambda k: svm_results[k]["AUC"])
svm = svm_results[best_kernel]["model"]
y_pred_svm = svm.predict(X_c_te_pp)
y_prob_svm = svm.predict_proba(X_c_te_pp)[:,1]
print(f"\\n→ Best SVM kernel: {best_kernel}")
"""))

cells.append(new_markdown_cell("### 5.3 K-Nearest Neighbors"))

cells.append(new_code_cell("""\
# ── K-sensitivity analysis ──
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_c_tr_pp, y_c_train)
    k_scores.append(roc_auc_score(y_c_val, knn.predict_proba(X_c_va_pp)[:,1]))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(k_range, k_scores, "o-", color=ACCENT[2])
best_k = k_range[np.argmax(k_scores)]
ax.axvline(best_k, color=ACCENT[0], linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k"); ax.set_ylabel("Val AUC-ROC")
ax.set_title("KNN — k-Sensitivity"); ax.legend()
plt.tight_layout(); plt.show()

knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(X_c_tr_pp, y_c_train)
y_pred_knn = knn.predict(X_c_te_pp)
y_prob_knn  = knn.predict_proba(X_c_te_pp)[:,1]
print(f"KNN (k={best_k}) | Acc {accuracy_score(y_c_test,y_pred_knn):.4f}"
      f" | AUC {roc_auc_score(y_c_test,y_prob_knn):.4f}")
"""))

cells.append(new_markdown_cell("### 5.4 Naive Bayes"))

cells.append(new_code_cell("""\
gnb = GaussianNB()
gnb.fit(X_c_tr_pp, y_c_train)
y_pred_gnb = gnb.predict(X_c_te_pp)
y_prob_gnb  = gnb.predict_proba(X_c_te_pp)[:,1]
print(f"Gaussian NB | Acc {accuracy_score(y_c_test,y_pred_gnb):.4f}"
      f" | AUC {roc_auc_score(y_c_test,y_prob_gnb):.4f}")
"""))

cells.append(new_markdown_cell("### 5.5 Decision Tree"))

cells.append(new_code_cell("""\
# ── Depth sensitivity ──
depth_scores = {}
for depth in [2, 3, 4, 5, 7, 10, None]:
    dt_ = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    dt_.fit(X_c_tr_pp, y_c_train)
    depth_scores[str(depth)] = roc_auc_score(y_c_val, dt_.predict_proba(X_c_va_pp)[:,1])

print("Depth vs Val AUC:")
for d, s in depth_scores.items():
    print(f"  depth={d:4s} → AUC {s:.4f}")

best_depth = int([k for k,v in depth_scores.items() if v==max(depth_scores.values())][0] or 0) or None

dt = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)
dt.fit(X_c_tr_pp, y_c_train)
y_pred_dt = dt.predict(X_c_te_pp)
y_prob_dt  = dt.predict_proba(X_c_te_pp)[:,1]

# Visualize
fig, ax = plt.subplots(figsize=(20, 7))
plot_tree(dt, max_depth=3, ax=ax, filled=True, rounded=True, fontsize=7,
          feature_names=[f"f{i}" for i in range(X_c_tr_pp.shape[1])],
          class_names=bc_data.target_names)
ax.set_title("Decision Tree (max_depth=3 shown for readability)")
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 5.6 Random Forest"))

cells.append(new_code_cell("""\
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
rf.fit(X_c_tr_pp, y_c_train)
y_pred_rf = rf.predict(X_c_te_pp)
y_prob_rf  = rf.predict_proba(X_c_te_pp)[:,1]
print(f"Random Forest | Acc {accuracy_score(y_c_test,y_pred_rf):.4f}"
      f" | AUC {roc_auc_score(y_c_test,y_prob_rf):.4f}"
      f" | F1 {f1_score(y_c_test,y_pred_rf):.4f}")

# ── Feature importances ──
n_show = 20
importances = rf.feature_importances_
idx = np.argsort(importances)[-n_show:]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(n_show), importances[idx], color=ACCENT[1], edgecolor="k", linewidth=0.4)
ax.set_yticks(range(n_show)); ax.set_yticklabels([f"f{i}" for i in idx], fontsize=8)
ax.set_xlabel("Importance"); ax.set_title(f"Random Forest — Top {n_show} Feature Importances")
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 5.7 Gradient Boosting"))

cells.append(new_code_cell("""\
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  random_state=SEED)
gb.fit(X_c_tr_pp, y_c_train)
y_pred_gb = gb.predict(X_c_te_pp)
y_prob_gb  = gb.predict_proba(X_c_te_pp)[:,1]
print(f"Gradient Boosting | Acc {accuracy_score(y_c_test,y_pred_gb):.4f}"
      f" | AUC {roc_auc_score(y_c_test,y_prob_gb):.4f}")
"""))

cells.append(new_markdown_cell("### 5.8 AdaBoost & ExtraTrees"))

cells.append(new_code_cell("""\
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=SEED)
ada.fit(X_c_tr_pp, y_c_train)
y_pred_ada = ada.predict(X_c_te_pp)

et = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
et.fit(X_c_tr_pp, y_c_train)
y_pred_et = et.predict(X_c_te_pp)

print(f"AdaBoost     | Acc {accuracy_score(y_c_test,y_pred_ada):.4f}"
      f" | AUC {roc_auc_score(y_c_test,ada.predict_proba(X_c_te_pp)[:,1]):.4f}")
print(f"ExtraTrees   | Acc {accuracy_score(y_c_test,y_pred_et):.4f}"
      f" | AUC {roc_auc_score(y_c_test,et.predict_proba(X_c_te_pp)[:,1]):.4f}")
"""))

cells.append(new_markdown_cell("### 5.9 XGBoost & LightGBM"))

cells.append(new_code_cell("""\
if HAS_BOOST:
    xgb_c = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=SEED, verbosity=0)
    xgb_c.fit(X_c_tr_pp, y_c_train,
               eval_set=[(X_c_va_pp, y_c_val)], verbose=False)
    y_pred_xgb = xgb_c.predict(X_c_te_pp)
    print(f"XGBoost  | Acc {accuracy_score(y_c_test,y_pred_xgb):.4f}"
          f" | AUC {roc_auc_score(y_c_test,xgb_c.predict_proba(X_c_te_pp)[:,1]):.4f}")

    lgb_c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 num_leaves=31, random_state=SEED, verbosity=-1)
    lgb_c.fit(X_c_tr_pp, y_c_train,
               eval_set=[(X_c_va_pp, y_c_val)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
    y_pred_lgb = lgb_c.predict(X_c_te_pp)
    print(f"LightGBM | Acc {accuracy_score(y_c_test,y_pred_lgb):.4f}"
          f" | AUC {roc_auc_score(y_c_test,lgb_c.predict_proba(X_c_te_pp)[:,1]):.4f}")
else:
    print("Skipping — install xgboost and lightgbm.")
"""))

cells.append(new_markdown_cell("### 5.10 Ensemble: Voting & Stacking"))

cells.append(new_code_cell("""\
# ── Soft Voting ──
voting = VotingClassifier(estimators=[
    ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ("gb", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
], voting="soft", n_jobs=-1)
voting.fit(X_c_tr_pp, y_c_train)
y_pred_vote = voting.predict(X_c_te_pp)
print(f"Voting   | Acc {accuracy_score(y_c_test,y_pred_vote):.4f}"
      f" | AUC {roc_auc_score(y_c_test,voting.predict_proba(X_c_te_pp)[:,1]):.4f}")

# ── Stacking ──
stacking = StackingClassifier(estimators=[
    ("lr",  LogisticRegression(max_iter=1000, random_state=SEED)),
    ("svm", SVC(probability=True, random_state=SEED)),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
], final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
stacking.fit(X_c_tr_pp, y_c_train)
y_pred_stack = stacking.predict(X_c_te_pp)
print(f"Stacking | Acc {accuracy_score(y_c_test,y_pred_stack):.4f}"
      f" | AUC {roc_auc_score(y_c_test,stacking.predict_proba(X_c_te_pp)[:,1]):.4f}")
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 6 — REGRESSION
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 📉 Section 6 — Classical ML: Regression"))

cells.append(new_code_cell("""\
def reg_report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<28} | RMSE {rmse:7.3f} | MAE {mae:7.3f} | R² {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

reg_scores = {}
"""))

cells.append(new_code_cell("""\
# ── Linear Regression ──
lin = LinearRegression()
lin.fit(X_r_tr_pp, y_r_train)
reg_scores["Linear"] = reg_report("Linear Regression", y_r_test, lin.predict(X_r_te_pp))

# ── Ridge (L2) ──
for alpha in [0.1, 1.0, 10.0]:
    m = Ridge(alpha=alpha).fit(X_r_tr_pp, y_r_train)
    reg_scores[f"Ridge(α={alpha})"] = reg_report(f"Ridge (α={alpha})", y_r_test, m.predict(X_r_te_pp))

# ── Lasso (L1) ──
for alpha in [0.01, 0.1, 1.0]:
    m = Lasso(alpha=alpha, max_iter=10000).fit(X_r_tr_pp, y_r_train)
    reg_scores[f"Lasso(α={alpha})"] = reg_report(f"Lasso (α={alpha})", y_r_test, m.predict(X_r_te_pp))

# ── ElasticNet ──
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000).fit(X_r_tr_pp, y_r_train)
reg_scores["ElasticNet"] = reg_report("ElasticNet", y_r_test, en.predict(X_r_te_pp))

# ── Polynomial Regression ──
poly_r = PolynomialFeatures(degree=2, include_bias=False)
X_r_poly = poly_r.fit_transform(X_r_tr_pp)
poly_lin  = Ridge(alpha=1.0).fit(X_r_poly, y_r_train)
pred_poly = poly_lin.predict(poly_r.transform(X_r_te_pp))
reg_scores["Poly Ridge"] = reg_report("Polynomial Ridge (deg=2)", y_r_test, pred_poly)
"""))

cells.append(new_code_cell("""\
# ── SVR ──
svr = SVR(kernel="rbf", C=1.0, epsilon=5.0)
svr.fit(X_r_tr_pp, y_r_train)
reg_scores["SVR"] = reg_report("SVR (RBF)", y_r_test, svr.predict(X_r_te_pp))

# ── Decision Tree Regressor ──
dtr = DecisionTreeRegressor(max_depth=5, random_state=SEED)
dtr.fit(X_r_tr_pp, y_r_train)
reg_scores["DecisionTree"] = reg_report("Decision Tree Reg", y_r_test, dtr.predict(X_r_te_pp))

# ── Random Forest Regressor ──
rfr = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)
rfr.fit(X_r_tr_pp, y_r_train)
reg_scores["RF"] = reg_report("Random Forest Reg", y_r_test, rfr.predict(X_r_te_pp))

# ── Gradient Boosting Regressor ──
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, subsample=0.8, random_state=SEED)
gbr.fit(X_r_tr_pp, y_r_train)
reg_scores["GBR"] = reg_report("Gradient Boosting Reg", y_r_test, gbr.predict(X_r_te_pp))

if HAS_BOOST:
    xgb_r = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                               random_state=SEED, verbosity=0)
    xgb_r.fit(X_r_tr_pp, y_r_train,
               eval_set=[(X_r_va_pp, y_r_val)], verbose=False)
    reg_scores["XGBoost"] = reg_report("XGBoost Reg", y_r_test, xgb_r.predict(X_r_te_pp))
"""))

cells.append(new_code_cell("""\
# ── Residual plots for best regressor ──
best_reg = rfr
y_pred_best = best_reg.predict(X_r_te_pp)
residuals   = y_r_test.values - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Actual vs Predicted
axes[0].scatter(y_r_test, y_pred_best, alpha=0.5, s=20, color=ACCENT[2])
lo, hi = y_r_test.min(), y_r_test.max()
axes[0].plot([lo,hi],[lo,hi],"--", color=ACCENT[0], linewidth=2, label="Ideal")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Actual vs Predicted (R²={r2_score(y_r_test,y_pred_best):.4f})")
axes[0].legend()

# Residual vs Predicted
axes[1].scatter(y_pred_best, residuals, alpha=0.5, s=20, color=ACCENT[4])
axes[1].axhline(0, color=ACCENT[0], linestyle="--", linewidth=2)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residual vs Predicted")

# Residual distribution
axes[2].hist(residuals, bins=30, color=ACCENT[1], edgecolor="k", linewidth=0.4, alpha=0.85)
from scipy.stats import norm
mu, std = norm.fit(residuals)
x_line = np.linspace(residuals.min(), residuals.max(), 100)
axes[2].plot(x_line, norm.pdf(x_line, mu, std) * len(residuals) * (residuals.max()-residuals.min())/30,
             color=ACCENT[0], linewidth=2, label=f"N({mu:.1f},{std:.1f})")
axes[2].set_xlabel("Residual"); axes[2].set_title("Residual Distribution")
axes[2].legend()

plt.suptitle("Regression Diagnostics — Random Forest Regressor", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(new_code_cell("""\
# ── Comparison bar chart ──
reg_df = pd.DataFrame(reg_scores).T.sort_values("R2", ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, metric, color in zip(axes, ["R2","RMSE","MAE"], [ACCENT[1],ACCENT[0],ACCENT[3]]):
    vals = reg_df[metric].sort_values(ascending=(metric!="R2"))
    ax.barh(vals.index, vals.values, color=color, edgecolor="k", linewidth=0.4)
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_xlabel(metric)
plt.tight_layout(); plt.show()
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 7 — CROSS VALIDATION
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 🔁 Section 7 — Cross Validation"))

cells.append(new_markdown_cell("### 7.1 K-Fold & Stratified K-Fold"))

cells.append(new_code_cell("""\
cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest":        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    "SVM (RBF)":            SVC(probability=True, random_state=SEED),
    "KNN":                  KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

print(f"{'Model':<25} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} {'Fold 4':>8} {'Fold 5':>8} {'Mean':>8} {'Std':>7}")
print("-" * 85)
for name, model in cv_models.items():
    scores = cross_val_score(model, X_c_tr_pp, y_c_train,
                              cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:<25}", " ".join(f"{s:8.4f}" for s in scores),
          f"{scores.mean():8.4f} {scores.std():7.4f}")
"""))

cells.append(new_code_cell("""\
# ── Box plot of CV scores ──
fig, ax = plt.subplots(figsize=(10, 5))
data_for_plot = [cv_results[name] for name in cv_models]
bp = ax.boxplot(data_for_plot, patch_artist=True, notch=False,
                labels=[n.replace(" ","\n") for n in cv_models],
                medianprops=dict(color="#f39c12", linewidth=2.5))
for patch, color in zip(bp["boxes"], ACCENT):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("AUC-ROC"); ax.set_title("5-Fold Stratified CV — AUC-ROC Distribution")
ax.grid(True, axis="y")
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 7.2 Cross Validate — Multiple Metrics Simultaneously"))

cells.append(new_code_cell("""\
multi_scoring = {
    "accuracy":  "accuracy",
    "roc_auc":   "roc_auc",
    "f1":        "f1",
    "precision": "precision",
    "recall":    "recall",
}
cv_multi = cross_validate(
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    X_c_tr_pp, y_c_train, cv=skf, scoring=multi_scoring, n_jobs=-1)

print("── Random Forest: 5-Fold CV Multi-Metric ──")
for metric in multi_scoring:
    vals = cv_multi[f"test_{metric}"]
    print(f"  {metric:<12} : {vals.mean():.4f} ± {vals.std():.4f}"
          f"  [{vals.min():.4f} – {vals.max():.4f}]")
"""))

cells.append(new_markdown_cell("### 7.3 Learning Curves"))

cells.append(new_code_cell("""\
train_sizes_frac = np.linspace(0.1, 1.0, 10)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

models_lc = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
    ("Random Forest",       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)),
    ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
]
for ax, (name, model) in zip(axes, models_lc):
    sizes, tr_scores, va_scores = learning_curve(
        model, X_c_tr_pp, y_c_train,
        train_sizes=train_sizes_frac, cv=skf,
        scoring="roc_auc", n_jobs=-1)
    tr_m, tr_s = tr_scores.mean(1), tr_scores.std(1)
    va_m, va_s = va_scores.mean(1), va_scores.std(1)

    ax.plot(sizes, tr_m, "o-", color=ACCENT[1], label="Train")
    ax.fill_between(sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.2, color=ACCENT[1])
    ax.plot(sizes, va_m, "o-", color=ACCENT[0], label="Val")
    ax.fill_between(sizes, va_m-va_s, va_m+va_s, alpha=0.2, color=ACCENT[0])
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Training Size"); ax.set_ylabel("AUC-ROC")
    ax.legend(); ax.grid(True)

plt.suptitle("Learning Curves — Bias-Variance Tradeoff", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 7.4 Validation Curves"))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF: n_estimators
param_range = [10, 30, 50, 100, 200, 300]
tr_sc, va_sc = validation_curve(
    RandomForestClassifier(n_jobs=-1, random_state=SEED),
    X_c_tr_pp, y_c_train,
    param_name="n_estimators", param_range=param_range,
    cv=skf, scoring="roc_auc", n_jobs=-1)

for ax, (scores_tr, scores_va, rng, label, xlog) in zip(axes, [
    (tr_sc, va_sc, param_range, "n_estimators (RF)", False),
]):
    tr_m, tr_s = scores_tr.mean(1), scores_tr.std(1)
    va_m, va_s = scores_va.mean(1), scores_va.std(1)
    ax.plot(rng, tr_m, "o-", color=ACCENT[1], label="Train"); ax.fill_between(rng, tr_m-tr_s, tr_m+tr_s, alpha=0.2, color=ACCENT[1])
    ax.plot(rng, va_m, "o-", color=ACCENT[0], label="Val");   ax.fill_between(rng, va_m-va_s, va_m+va_s, alpha=0.2, color=ACCENT[0])
    ax.set_xlabel(label); ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Validation Curve — {label}")
    ax.legend()

# LogReg: C parameter
c_range = [0.001, 0.01, 0.1, 1, 10, 100]
tr_sc2, va_sc2 = validation_curve(
    LogisticRegression(max_iter=1000, solver="lbfgs"),
    X_c_tr_pp, y_c_train,
    param_name="C", param_range=c_range,
    cv=skf, scoring="roc_auc", n_jobs=-1)
tr_m2, tr_s2 = tr_sc2.mean(1), tr_sc2.std(1)
va_m2, va_s2 = va_sc2.mean(1), va_sc2.std(1)
axes[1].semilogx(c_range, tr_m2, "o-", color=ACCENT[1], label="Train"); axes[1].fill_between(c_range, tr_m2-tr_s2, tr_m2+tr_s2, alpha=0.2, color=ACCENT[1])
axes[1].semilogx(c_range, va_m2, "o-", color=ACCENT[0], label="Val");   axes[1].fill_between(c_range, va_m2-va_s2, va_m2+va_s2, alpha=0.2, color=ACCENT[0])
axes[1].set_xlabel("C (log scale)"); axes[1].set_ylabel("AUC-ROC")
axes[1].set_title("Validation Curve — C (Logistic Regression)")
axes[1].legend()

plt.tight_layout(); plt.show()
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 8 — EVALUATION
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 📊 Section 8 — Model Evaluation"))

cells.append(new_markdown_cell("### 8.1 Classification Metrics Comparison"))

cells.append(new_code_cell("""\
clf_models_eval = {
    "Logistic Regression": (lr,      X_c_te_pp),
    "SVM":                 (svm,     X_c_te_pp),
    "KNN":                 (knn,     X_c_te_pp),
    "Naive Bayes":         (gnb,     X_c_te_pp),
    "Decision Tree":       (dt,      X_c_te_pp),
    "Random Forest":       (rf,      X_c_te_pp),
    "Gradient Boosting":   (gb,      X_c_te_pp),
    "AdaBoost":            (ada,     X_c_te_pp),
    "ExtraTrees":          (et,      X_c_te_pp),
    "Voting":              (voting,  X_c_te_pp),
    "Stacking":            (stacking, X_c_te_pp),
}

rows = []
for name, (model, Xte) in clf_models_eval.items():
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:,1]
    rows.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_c_test, preds),
        "AUC-ROC":   roc_auc_score(y_c_test, probs),
        "F1":        f1_score(y_c_test, preds),
        "Precision": precision_score(y_c_test, preds),
        "Recall":    recall_score(y_c_test, preds),
    })

leaderboard = pd.DataFrame(rows).set_index("Model").sort_values("AUC-ROC", ascending=False)
display(leaderboard.style
        .background_gradient(cmap="Greens")
        .format("{:.4f}"))
"""))

cells.append(new_code_cell("""\
# ── ROC Curves — all classifiers ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors_roc = plt.cm.tab20(np.linspace(0, 1, len(clf_models_eval)))
for (name, (model, Xte)), color in zip(clf_models_eval.items(), colors_roc):
    probs = model.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(y_c_test, probs)
    auc = roc_auc_score(y_c_test, probs)
    axes[0].plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color)

axes[0].plot([0,1],[0,1],"k--", linewidth=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curves — All Models")
axes[0].legend(fontsize=7, loc="lower right")

# ── Precision-Recall Curves ──
for (name, (model, Xte)), color in zip(clf_models_eval.items(), colors_roc):
    probs = model.predict_proba(Xte)[:,1]
    prec, rec, _ = precision_recall_curve(y_c_test, probs)
    ap = average_precision_score(y_c_test, probs)
    axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color)

axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves — All Models")
axes[1].legend(fontsize=7, loc="upper right")

plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 8.2 Confusion Matrices"))

cells.append(new_code_cell("""\
top5_names = leaderboard.head(5).index.tolist()
fig, axes  = plt.subplots(1, 5, figsize=(20, 4))

for ax, name in zip(axes, top5_names):
    model, Xte = clf_models_eval[name]
    cm   = confusion_matrix(y_c_test, model.predict(Xte))
    disp = ConfusionMatrixDisplay(cm, display_labels=bc_data.target_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name, fontsize=8)

plt.suptitle("Confusion Matrices — Top 5 Models", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 8.3 Regression Evaluation"))

cells.append(new_code_cell("""\
print("── Final Regression Leaderboard ──")
reg_leaderboard = pd.DataFrame(reg_scores).T.sort_values("R2", ascending=False)
display(reg_leaderboard.style
        .background_gradient(subset=["R2"], cmap="Greens")
        .background_gradient(subset=["RMSE","MAE"], cmap="Reds_r")
        .format("{:.4f}"))
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 9 — HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 🎛️ Section 9 — Hyperparameter Tuning"))

cells.append(new_markdown_cell("### 9.1 Grid Search CV"))

cells.append(new_code_cell("""\
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 3],
}
gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    param_grid_rf, cv=skf, scoring="roc_auc", n_jobs=-1, verbose=0)
gs.fit(X_c_tr_pp, y_c_train)

print(f"Best params : {gs.best_params_}")
print(f"Best CV AUC : {gs.best_score_:.4f}")
print(f"Test  AUC   : {roc_auc_score(y_c_test, gs.best_estimator_.predict_proba(X_c_te_pp)[:,1]):.4f}")

# ── Results heatmap ──
gs_df = pd.DataFrame(gs.cv_results_)
piv = gs_df.pivot_table(index="param_max_depth", columns="param_n_estimators",
                         values="mean_test_score", aggfunc="max")
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGn", ax=ax, linewidths=0.5)
ax.set_title("Grid Search AUC (max_depth × n_estimators)")
plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 9.2 Random Search CV"))

cells.append(new_code_cell("""\
from scipy.stats import randint, uniform

param_dist_rf = {
    "n_estimators":      randint(50, 400),
    "max_depth":         [None, 3, 5, 7, 10, 15, 20],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "max_features":      uniform(0.3, 0.7),
}
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    param_dist_rf, n_iter=40, cv=skf,
    scoring="roc_auc", n_jobs=-1, random_state=SEED)
rs.fit(X_c_tr_pp, y_c_train)

print(f"Best params : {rs.best_params_}")
print(f"Best CV AUC : {rs.best_score_:.4f}")
print(f"Test  AUC   : {roc_auc_score(y_c_test, rs.best_estimator_.predict_proba(X_c_te_pp)[:,1]):.4f}")

# ── Score distribution ──
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(rs.cv_results_["mean_test_score"], bins=20, color=ACCENT[4], edgecolor="k")
ax.axvline(rs.best_score_, color=ACCENT[0], linestyle="--", linewidth=2, label="Best")
ax.set_xlabel("AUC-ROC"); ax.set_title("Random Search — Score Distribution")
ax.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("### 9.3 Optuna (Bayesian Optimization)"))

cells.append(new_code_cell("""\
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_categorical("max_depth", [None,3,5,7,10,15]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_float("max_features", 0.2, 1.0),
        }
        model  = RandomForestClassifier(**p, n_jobs=-1, random_state=SEED)
        scores = cross_val_score(model, X_c_tr_pp, y_c_train,
                                  cv=skf, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print(f"\\nBest AUC   : {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # ── Optimization history ──
    vals = [t.value for t in study.trials]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(vals, "o", alpha=0.5, color=ACCENT[2], ms=4)
    axes[0].plot(np.maximum.accumulate(vals), "-", color=ACCENT[0], linewidth=2, label="Best")
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("AUC"); axes[0].set_title("Optuna History"); axes[0].legend()

    # Param importances
    importances_opt = optuna.importance.get_param_importances(study)
    axes[1].barh(list(importances_opt.keys())[::-1], list(importances_opt.values())[::-1],
                 color=ACCENT[4], edgecolor="k")
    axes[1].set_xlabel("Importance"); axes[1].set_title("Hyperparameter Importances")
    plt.tight_layout(); plt.show()

except ImportError:
    print("Install optuna: pip install optuna")
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 10 — PIPELINE
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 🔗 Section 10 — End-to-End ML Pipeline"))

cells.append(new_code_cell("""\
# ── Full no-leakage pipeline from raw data to prediction ──
clf_pipeline = Pipeline([
    ("preprocessor", preprocessor_c),
    ("selector",     SelectKBest(f_classif, k=20)),
    ("classifier",   RandomForestClassifier(
                         n_estimators=200, max_depth=None,
                         n_jobs=-1, random_state=SEED)),
])

reg_pipeline = Pipeline([
    ("preprocessor", preprocessor_r),
    ("regressor",    GradientBoostingRegressor(
                         n_estimators=200, learning_rate=0.05,
                         max_depth=4, random_state=SEED)),
])

# Fit on raw (un-preprocessed) training data
clf_pipeline.fit(X_c_train, y_c_train)
reg_pipeline.fit(X_r_train, y_r_train)

# Evaluate on raw test data (pipeline handles everything)
y_pipe_clf = clf_pipeline.predict(X_c_test)
y_pipe_reg = reg_pipeline.predict(X_r_test)

print("── Classification Pipeline ──")
print(f"  Accuracy : {accuracy_score(y_c_test, y_pipe_clf):.4f}")
print(f"  AUC-ROC  : {roc_auc_score(y_c_test, clf_pipeline.predict_proba(X_c_test)[:,1]):.4f}")
print(f"  F1       : {f1_score(y_c_test, y_pipe_clf):.4f}")

print("\\n── Regression Pipeline ──")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_r_test, y_pipe_reg)):.4f}")
print(f"  MAE  : {mean_absolute_error(y_r_test, y_pipe_reg):.4f}")
print(f"  R²   : {r2_score(y_r_test, y_pipe_reg):.4f}")

print("\\n── Pipeline Steps ──")
for name, step in clf_pipeline.steps:
    print(f"  {name:15s} → {type(step).__name__}")
"""))

cells.append(new_code_cell("""\
# ── Single-sample inference ──
def predict_sample(raw_row, pipeline, is_clf=True):
    if is_clf:
        pred  = pipeline.predict(raw_row)[0]
        proba = pipeline.predict_proba(raw_row)[0]
        return {"prediction": int(pred),
                "class_name": bc_data.target_names[pred],
                "proba_0":    float(proba[0]),
                "proba_1":    float(proba[1])}
    else:
        return {"prediction": float(pipeline.predict(raw_row)[0])}

sample = X_c_test.iloc[[0]]
result = predict_sample(sample, clf_pipeline, is_clf=True)
print("── Single Sample Prediction ──")
print(f"  True label : {y_c_test.iloc[0]} ({bc_data.target_names[y_c_test.iloc[0]]})")
print(f"  Result     : {result}")
"""))

# ════════════════════════════════════════════════════════════════════
#  SECTION 11 — SAVE/LOAD
# ════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("---\n## 💾 Section 11 — Model Saving & Loading"))

cells.append(new_code_cell("""\
import os, pickle, json
SAVE_DIR = "saved_ml_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Save sklearn Pipeline (recommended approach) ──
with open(f"{SAVE_DIR}/clf_pipeline.pkl", "wb") as f:
    pickle.dump(clf_pipeline, f)
with open(f"{SAVE_DIR}/reg_pipeline.pkl", "wb") as f:
    pickle.dump(reg_pipeline, f)

# ── Reload and verify ──
with open(f"{SAVE_DIR}/clf_pipeline.pkl", "rb") as f:
    loaded_clf = pickle.load(f)
with open(f"{SAVE_DIR}/reg_pipeline.pkl", "rb") as f:
    loaded_reg = pickle.load(f)

preds_check = loaded_clf.predict(X_c_test)
print(f"Loaded CLF accuracy: {accuracy_score(y_c_test, preds_check):.4f} ✅")
regs_check = loaded_reg.predict(X_r_test)
print(f"Loaded REG R²      : {r2_score(y_r_test, regs_check):.4f} ✅")

# ── Save metadata ──
metadata = {
    "clf_pipeline":  {
        "steps":     [n for n, _ in clf_pipeline.steps],
        "test_acc":  float(accuracy_score(y_c_test, preds_check)),
        "test_auc":  float(roc_auc_score(y_c_test, loaded_clf.predict_proba(X_c_test)[:,1])),
    },
    "reg_pipeline": {
        "steps":     [n for n, _ in reg_pipeline.steps],
        "test_r2":   float(r2_score(y_r_test, regs_check)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_r_test, regs_check))),
    }
}
with open(f"{SAVE_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\\n── Saved files ──")
for fn in os.listdir(SAVE_DIR):
    print(f"  {fn}")

print("\\n── Metadata ──")
print(json.dumps(metadata, indent=2))
"""))

cells.append(new_code_cell("""\
# ════════════════ FINAL LEADERBOARD ════════════════
print("=" * 65)
print(f"{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>8}")
print("=" * 65)
for _, row in leaderboard.iterrows():
    print(f"{row.name:<25} {row['Accuracy']:>10.4f} {row['AUC-ROC']:>10.4f} {row['F1']:>8.4f}")
print("=" * 65)
print("\\n✅  ML Notebook complete — all models trained, evaluated & saved!")
"""))

# ─── Write ───────────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

out = "ML_Complete_Notebook.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"[OK]  ML notebook -> {out}")
print(f"   {len(nb.cells)} total cells")