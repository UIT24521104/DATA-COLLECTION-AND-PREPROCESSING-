"""
=============================================================================
FINDING COUNTRIES WITH ECONOMIC TRAJECTORIES SIMILAR TO VIETNAM
Parallel Comparison: Vector Similarity (DTW/Cosine/Euclidean) vs JSD
=============================================================================
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from scipy.special import rel_entr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dtaidistance import dtw

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_PATH  = "../DATA_AFTER_PREPROCESSING/data_final_merged.csv"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = ["Agri_GDP", "Elec_Access", "FDI_GDP", "GDPC_2015",
            "Indus_GDP", "Internet_Usage", "Serv_GDP"]
EPS      = 1e-6
TOP_N    = 10

PALETTE = {
    "vnm":      "#000000",
    "v1":       "#2563eb",   # Blue – Vector
    "v2":       "#dc2626",   # Red  – JSD
    "combined": "#16a34a",   # Green – Combined
    "bars":     ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd",
                 "#bfdbfe","#dbeafe","#eff6ff","#f0fdf4","#dcfce7"],
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f9fafb",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

# ─── 1. DATA READING & VERIFICATION ─────────────────────────────────────────
df_raw   = pd.read_csv(DATA_PATH)
YEARS    = sorted(df_raw["Year"].unique())
COUNTRIES = sorted(df_raw["Country"].unique())

print(f"✓ Dataset: {df_raw.shape[0]:,} rows | "
      f"{df_raw['Country'].nunique()} countries | "
      f"Years {YEARS[0]}–{YEARS[-1]}")
print(f"  Missing: {df_raw[FEATURES].isnull().sum().sum()} → no imputation required\n")

# ─── 2. PREPROCESSING ───────────────────────────────────────────────────────
def build_panel(df):
    return {c: g.sort_values("Year").set_index("Year")[FEATURES]
            for c, g in df.groupby("Country")}

panel_raw = build_panel(df_raw)

# Z-score: balance units for Vector methods
scaler_z = StandardScaler().fit(df_raw[FEATURES])
panel_z  = {c: pd.DataFrame(scaler_z.transform(m.values),
                             index=m.index, columns=FEATURES)
            for c, m in panel_raw.items()}

# MinMax [ε,1]: ensure ≥ 0 for KLD/JSD
scaler_mm = MinMaxScaler(feature_range=(EPS, 1)).fit(df_raw[FEATURES])
panel_mm  = {c: pd.DataFrame(scaler_mm.transform(m.values),
                              index=m.index, columns=FEATURES)
             for c, m in panel_raw.items()}

print("✓ Preprocessing completed: Z-score (Vector) | MinMax[ε,1] (JSD)\n")

# ─── 3. METHOD 1 – VECTOR SIMILARITY ────────────────────────────────────────
def dtw_multi(a, b):
    return np.mean([dtw.distance_fast(a[:,f].astype(np.float64),
                                      b[:,f].astype(np.float64))
                    for f in range(a.shape[1])])

vnm_z_mat  = panel_z["VNM"].values
vnm_z_flat = vnm_z_mat.flatten()

print("── Method 1: Vector (DTW + Cosine + Euclidean) ──────────────")
rows_v1 = []
for c in COUNTRIES:
    if c == "VNM": continue
    m = panel_z[c].values
    rows_v1.append({"Country": c,
                    "DTW":      dtw_multi(vnm_z_mat, m),
                    "Cosine":   cosine(vnm_z_flat, m.flatten()),
                    "Euclidean":euclidean(vnm_z_flat, m.flatten())})

df_v1 = pd.DataFrame(rows_v1)
for col in ["DTW","Cosine","Euclidean"]:
    df_v1[f"r_{col}"] = df_v1[col].rank()
df_v1["Score_V1"] = df_v1[["r_DTW","r_Cosine","r_Euclidean"]].sum(axis=1)
df_v1 = df_v1.sort_values("Score_V1").reset_index(drop=True)
df_v1["Rank_V1"] = df_v1.index + 1

print(df_v1[["Rank_V1","Country","DTW","Cosine","Euclidean","Score_V1"]]
      .head(TOP_N).to_string(index=False))

# ─── 4. METHOD 2 – JSD ──────────────────────────────────────────────────────
def hist_prob(s, bins=10):
    h, _ = np.histogram(s, bins=bins)
    h = h.astype(float) + EPS
    return h / h.sum()

def jsd(p, q):
    m = 0.5*(p+q)
    return float(0.5*np.sum(rel_entr(p,m)) + 0.5*np.sum(rel_entr(q,m)))

def jsd_multi(a, b):
    return np.mean([jsd(hist_prob(a[f].values), hist_prob(b[f].values))
                    for f in FEATURES])

vnm_mm = panel_mm["VNM"]

print("\n── Method 2: Jensen-Shannon Divergence ───────────────────────")
rows_v2 = [{"Country": c,
             "JSD": jsd_multi(vnm_mm, panel_mm[c])}
           for c in COUNTRIES if c != "VNM"]

df_v2 = pd.DataFrame(rows_v2).sort_values("JSD").reset_index(drop=True)
df_v2["Rank_V2"] = df_v2.index + 1

print(df_v2[["Rank_V2","Country","JSD"]].head(TOP_N).to_string(index=False))

# ─── 5. COMBINED RANKING ─────────────────────────────────────────────────────
df_all = (df_v1[["Country","Score_V1","Rank_V1","DTW","Cosine","Euclidean"]]
          .merge(df_v2[["Country","JSD","Rank_V2"]], on="Country"))
df_all["Rank_Combined"] = (df_all["Rank_V1"] + df_all["Rank_V2"]) / 2
df_all = df_all.sort_values("Rank_Combined").reset_index(drop=True)
df_all["Final_Rank"] = df_all.index + 1

top = df_all.head(TOP_N).copy()

print(f"\n── Top {TOP_N} Combined ────────────────────────────────────────────")
print(top[["Final_Rank","Country","Rank_V1","Rank_V2","Rank_Combined"]]
      .to_string(index=False))

# Setup Top 5 for Visualizations
TOP5 = top["Country"].tolist()[:5]
COLORS5 = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261"]
C_MAP   = dict(zip(TOP5, COLORS5))


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PLOTS GENERATION (Each saved as .pdf)
# ══════════════════════════════════════════════════════════════════════════════

# ── CHART 1: Parallel Ranking Comparison ─────────────────────────────────────
fig_rank, ax_rank = plt.subplots(figsize=(14, 6))
x = np.arange(TOP_N)
width = 0.25
labels = top["Country"].values
r_v1 = top["Rank_V1"].values
r_v2 = top["Rank_V2"].values
r_comb = top["Rank_Combined"].values

b1 = ax_rank.bar(x - width, r_v1,  width, label="Vector Method (V1)", color=PALETTE["v1"],  alpha=0.85)
b2 = ax_rank.bar(x,         r_v2,  width, label="Distribution JSD (V2)",    color=PALETTE["v2"],  alpha=0.85)
b3 = ax_rank.bar(x + width, r_comb, width, label="Combined Rank",    color=PALETTE["combined"], alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax_rank.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:.0f}" if h.is_integer() else f"{h:.1f}", 
                     ha="center", va="bottom", fontsize=8)

ax_rank.set_xticks(x)
ax_rank.set_xticklabels(labels, fontsize=10)
ax_rank.set_ylabel("Rank (Lower = More Similar)")
ax_rank.set_title(f"Parallel Ranking Comparison: Vector vs JSD vs Combined (Top {TOP_N})", fontweight="bold", pad=12)
ax_rank.legend(fontsize=10)
ax_rank.invert_yaxis()

plt.savefig(f"{OUTPUT_DIR}/plot_rank_comparison.pdf", bbox_inches="tight")
plt.close()
print("✓ Saved: plot_rank_comparison.pdf")


# ── CHART 2: Method Convergence Scatter Plot ─────────────────────────────────
fig_sc, ax_sc = plt.subplots(figsize=(7, 6))
ax_sc.scatter(df_all["Rank_V1"], df_all["Rank_V2"], s=15, alpha=0.35, color="#9ca3af")

for _, row in top.iterrows():
    c = row["Country"]
    col = C_MAP.get(c, "#374151")
    ax_sc.scatter(row["Rank_V1"], row["Rank_V2"], s=80, color=col, zorder=5)
    ax_sc.annotate(c, (row["Rank_V1"], row["Rank_V2"]), textcoords="offset points", xytext=(4,3), fontsize=8)

lim = df_all[["Rank_V1","Rank_V2"]].max().max() + 8
ax_sc.plot([0,lim],[0,lim], "--", color="#d1d5db", lw=1)
ax_sc.set_xlim(0, lim)
ax_sc.set_ylim(0, lim)
ax_sc.set_xlabel("Rank – Vector Similarity (V1)")
ax_sc.set_ylabel("Rank – Jensen-Shannon Divergence (V2)")
ax_sc.set_title("Method Convergence\n(Bottom-Left Region = High Consistency)", fontweight="bold", fontsize=11, pad=10)

plt.savefig(f"{OUTPUT_DIR}/plot_method_convergence.pdf", bbox_inches="tight")
plt.close()
print("✓ Saved: plot_method_convergence.pdf")


# ── CHART 3: DTW Distance vs JSD Scatter Plot ────────────────────────────────
fig_dj, ax_dj = plt.subplots(figsize=(7, 6))
ax_dj.scatter(df_all["DTW"], df_all["JSD"], s=15, alpha=0.35, color="#9ca3af")

for _, row in top.iterrows():
    c = row["Country"]
    col = C_MAP.get(c, "#374151")
    ax_dj.scatter(row["DTW"], row["JSD"], s=80, color=col, zorder=5)
    ax_dj.annotate(c, (row["DTW"], row["JSD"]), textcoords="offset points", xytext=(4,2), fontsize=8)

ax_dj.set_xlabel("DTW Distance (Time-Series Shape)")
ax_dj.set_ylabel("Jensen-Shannon Divergence (Distribution)")
ax_dj.set_title("DTW (Vector) vs JSD (Distribution)\n(Bottom-Left Region = High Similarity in Both Dimensions)", fontweight="bold", fontsize=11, pad=10)

plt.savefig(f"{OUTPUT_DIR}/plot_dtw_vs_jsd.pdf", bbox_inches="tight")
plt.close()
print("✓ Saved: plot_dtw_vs_jsd.pdf")


# ── CHART 4: 20-Year Mean Comparison (Radar Chart) ──────────────────────────
fig_radar = plt.subplots(figsize=(8, 8))
ax_radar = plt.subplot(111, polar=True)
N = len(FEATURES)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

sc_r = MinMaxScaler()
all_mean = np.vstack([panel_raw[c][FEATURES].mean().values for c in ["VNM"]+TOP5])
sc_r.fit(all_mean)

for c, col in zip(["VNM"]+TOP5, ["black"]+COLORS5):
    v = sc_r.transform(panel_raw[c][FEATURES].mean().values.reshape(1,-1))[0].tolist()
    v += v[:1]
    ax_radar.plot(angles, v, color=col, lw=2.5 if c=="VNM" else 1.5, label=c)
    ax_radar.fill(angles, v, color=col, alpha=0.04)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels([f.replace("_","\n") for f in FEATURES], fontsize=8)
ax_radar.set_title("Radar Chart: 20-Year Features Mean\nVNM vs Top 5 Most Similar Countries", fontweight="bold", fontsize=11, pad=20)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=9)

plt.savefig(f"{OUTPUT_DIR}/plot_features_radar.pdf", bbox_inches="tight")
plt.close()
print("✓ Saved: plot_features_radar.pdf")


# ── CHART 5: Feature Time-Series Trajectories (Multi-panel) ──────────────────
fig_ts, axes_ts = plt.subplots(2, 4, figsize=(18, 9))
axes_ts = axes_ts.flatten()

for i, feat in enumerate(FEATURES):
    ax_f = axes_ts[i]
    vnm_v = panel_raw["VNM"][feat].values
    ax_f.plot(YEARS, vnm_v, color="black", lw=3.0, label="VNM (Vietnam)", zorder=5)
    
    for cn, col in C_MAP.items():
        ax_f.plot(YEARS, panel_raw[cn][feat].values, color=col, lw=1.5, alpha=0.85, label=cn)

    ax_f.set_title(feat, fontsize=10, fontweight="bold")
    ax_f.tick_params(labelsize=8)
    ax_f.set_xlim(YEARS[0], YEARS[-1])
    ax_f.set_xlabel("Year", fontsize=8)
    
    if i == 0:
        ax_f.legend(fontsize=8, ncol=2, loc="upper left")

# Hide any extra empty subplots (e.g., 7 features in an 8-panel grid)
for extra in range(len(FEATURES), len(axes_ts)):
    axes_ts[extra].set_visible(False)

plt.suptitle("Historical Trajectories (20-Year Time-Series) across Key Economic Indicators", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_time_series_trajectories.pdf", bbox_inches="tight")
plt.close()
print("✓ Saved: plot_time_series_trajectories.pdf")


# ── CHART 6: Top 20 Similarity Ranking Table ───────────────────────────────
top20 = df_all.head(20).copy()
top20_disp = top20[["Final_Rank","Country",
                    "Rank_V1","Rank_V2","Rank_Combined",
                    "DTW","Cosine","Euclidean","JSD"]].copy()

for col in ["DTW","Cosine","Euclidean","JSD"]:
    top20_disp[col] = top20_disp[col].map("{:.4f}".format)
for col in ["Rank_V1","Rank_V2"]:
    top20_disp[col] = top20_disp[col].astype(int)
top20_disp["Rank_Combined"] = top20_disp["Rank_Combined"].map("{:.1f}".format)

col_labels = ["Final\nRank", "Country",
              "Vector\nRank (V1)", "JSD\nRank (V2)", "Combined\nRank",
              "DTW\nDistance", "Cosine\nDist", "Euclidean\nDist", "JSD\nValue"]

fig2, ax2 = plt.subplots(figsize=(15, 10))
ax2.axis("off")
ax2.set_title("Comprehensive Similarity Ranking: Top 20 Countries vs Vietnam\n"
              "(Integrating Multidimensional Vector Proximity & Jensen-Shannon Divergence)",
              fontsize=13, fontweight="bold", pad=20)

tbl = ax2.table(
    cellText=top20_disp.values,
    colLabels=col_labels,
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)

# Header styling
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1e3a5f")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Dynamic column shading
col_bg = {2: "#dbeafe", 3: "#fee2e2", 4: "#dcfce7"}
for i in range(1, len(top20)+1):
    for j in range(len(col_labels)):
        cell = tbl[i, j]
        if j in col_bg:
            cell.set_facecolor(col_bg[j])
        elif i % 2 == 0:
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("white")
    # Highlight Top 5
    if i <= 5:
        tbl[i, 0].set_text_props(fontweight="bold", color=PALETTE["v2"])
        tbl[i, 1].set_text_props(fontweight="bold")

# Add informative table legends
fig2.text(0.15, 0.04, "■ Blue Accent = Vector Similarity Rank", color=PALETTE["v1"], fontsize=9, ha="left")
fig2.text(0.42, 0.04, "■ Red Accent = Distribution JSD Rank", color=PALETTE["v2"], fontsize=9, ha="left")
fig2.text(0.68, 0.04, "■ Green Accent = Final Combined Rank", color=PALETTE["combined"], fontsize=9, ha="left")

plt.savefig(f"{OUTPUT_DIR}/ranking_table.pdf", bbox_inches="tight")
plt.close()
print("✓ Saved: ranking_table.pdf")


# ── EXPORT DATA LAYER (CSV) ──────────────────────────────────────────────────
df_all[["Final_Rank","Country","Rank_V1","Rank_V2","Rank_Combined",
        "DTW","Cosine","Euclidean","JSD"]].to_csv(
    f"{OUTPUT_DIR}/similarity_ranking_combined.csv", index=False)
print("✓ Saved Data Log: similarity_ranking_combined.csv")