"""
═══════════════════════════════════════════════════════════════════════════════
STREAMLIT DASHBOARD: Quỹ đạo kinh tế tương đồng với Việt Nam
Import & visualize từ Models/Trajectory.py
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.special import rel_entr

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quỹ đạo kinh tế – VNM",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotly theme
import plotly.io as pio
pio.templates.default = "plotly_white"

# ─── MÀU SẮC NHẤT QUÁN ───────────────────────────────────────────────────────
COLOR_VNM      = "#000000"   # VNM đen
COLOR_V1       = "#2563eb"   # Vector → xanh dương
COLOR_V2       = "#dc2626"   # JSD    → đỏ
COLOR_COMBINED = "#16a34a"   # Tổng hợp → xanh lá
COLORS_TOP5    = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261"]

# ─── IMPORT DỮ LIỆU TỪ CSV VÀ DATASET_MERGED.CSV ──────────────────────────────
@st.cache_resource(ttl=3600)
def load_trajectory_data():
    """Load dữ liệu từ CSV output và dataset_merged.csv"""
    with st.spinner("⏳ Đang tải dữ liệu..."):
        # Determine base path (works in both local and Docker)
        base_path = Path(__file__).resolve().parents[1]
        
        # Load ranking từ CSV
        ranking_path = base_path / "MODELS" / "similarity_ranking_combined.csv"
        df_all = pd.read_csv(ranking_path)
        
        # Load dữ liệu gốc
        data_path = base_path / "DATA_AFTER_PREPROCESSING" / "dataset_merged.csv"
        df_raw = pd.read_csv(data_path)
        
        # Tạo panel_raw: dict[country → DataFrame(year × features)]
        FEATURES = ["Agri_GDP","Elec_Access","FDI_GDP","GDPC_2015",
                    "Indus_GDP","Internet_Usage","Serv_GDP"]
        YEARS = sorted(df_raw["Year"].unique())
        COUNTRIES = sorted(df_raw["Country"].unique())
        
        panel_raw = {
            c: g.sort_values("Year").set_index("Year")[FEATURES]
            for c, g in df_raw.groupby("Country")
        }
        
        # Tạo df_v1, df_v2 từ df_all
        df_v1 = df_all[["Country", "DTW", "Cosine", "Euclidean", "Rank_V1"]].sort_values("Rank_V1")
        df_v2 = df_all[["Country", "JSD", "Rank_V2"]].sort_values("Rank_V2")
        
        # TOP5
        TOP5 = df_all.head(5)["Country"].tolist()
        
        return {
            "df_all": df_all,
            "df_v1": df_v1,
            "df_v2": df_v2,
            "panel_raw": panel_raw,
            "FEATURES": FEATURES,
            "YEARS": YEARS,
            "COUNTRIES": COUNTRIES,
            "TOP5": TOP5,
        }

# Load dữ liệu
data = load_trajectory_data()
df_all = data["df_all"]
df_v1 = data["df_v1"]
df_v2 = data["df_v2"]
panel_raw = data["panel_raw"]
FEATURES = data["FEATURES"]
YEARS = data["YEARS"]
COUNTRIES = data["COUNTRIES"]
TOP5 = data["TOP5"]

# Tạo các hằng số helper
COUNTRIES_WITHOUT_VNM = [c for c in COUNTRIES if c != "VNM"]
VNM_FEATURES = panel_raw["VNM"].mean().values

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙ Bộ lọc")
st.sidebar.divider()

# Initialize session state
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = {
        "methods": ["Vector (V1)", "JSD (V2)", "Combined"],
        "top_n": 10,
        "period": "2004–2023",
        "selected_features": FEATURES.copy()
    }

# 1. Chọn phương pháp
methods = st.sidebar.multiselect(
    "Phương pháp so sánh",
    options=["Vector (V1)", "JSD (V2)", "Combined"],
    default=st.session_state.filters_applied["methods"],
    help="Chọn phương pháp(s) để hiển thị kết quả so sánh"
)

# 2. Top N
top_n = st.sidebar.slider(
    "Số quốc gia hiển thị",
    min_value=5,
    max_value=20,
    value=st.session_state.filters_applied["top_n"],
    step=1,
    help="Lọc và hiển thị Top N quốc gia tương đồng nhất"
)

# 3. Giai đoạn phân tích
period = st.sidebar.radio(
    "Giai đoạn phân tích",
    ["2004–2023", "2010–2023", "2015–2023"],
    index=["2004–2023", "2010–2023", "2015–2023"].index(st.session_state.filters_applied["period"]),
    help="Lọc năm khi vẽ time series"
)

# 4. Chỉ số kinh tế
selected_features = st.sidebar.multiselect(
    "Chỉ số kinh tế",
    options=FEATURES,
    default=st.session_state.filters_applied["selected_features"],
    help="Chọn chỉ số để hiển thị trong chuỗi thời gian"
)

# 5. Nút cập nhật
st.sidebar.divider()
col_update, col_reset = st.sidebar.columns(2)

with col_update:
    if st.button("🔄 Cập nhật kết quả", use_container_width=True, type="primary"):
        st.session_state.filters_applied = {
            "methods": methods,
            "top_n": top_n,
            "period": period,
            "selected_features": selected_features
        }
        st.rerun()

with col_reset:
    if st.button("↺ Đặt lại", use_container_width=True):
        st.session_state.filters_applied = {
            "methods": ["Vector (V1)", "JSD (V2)", "Combined"],
            "top_n": 10,
            "period": "2004–2023",
            "selected_features": FEATURES.copy()
        }
        st.rerun()

# Sử dụng giá trị từ session state
methods = st.session_state.filters_applied["methods"]
top_n = st.session_state.filters_applied["top_n"]
period = st.session_state.filters_applied["period"]
selected_features = st.session_state.filters_applied["selected_features"]
year_start = int(period.split("–")[0])

# 6. Download CSV
st.sidebar.divider()
top_df = df_all.head(top_n)
csv = top_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇ Tải CSV",
    csv,
    f"similarity_ranking_top{top_n}.csv",
    "text/csv",
    help="Tải bảng xếp hạng Top N dưới dạng CSV"
)

# ─── MAIN CONTENT ────────────────────────────────────────────────────────────
st.title("🌍 Quỹ đạo kinh tế tương đồng với Việt Nam")
st.caption("So sánh hai phương pháp: Vector Similarity (DTW/Cosine/Euclidean) vs Jensen-Shannon Divergence")

# Tạo tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan",
    "📈 Chuỗi thời gian",
    "🔍 So sánh chi tiết",
    "📉 Phân phối (JSD)"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Tổng quan xếp hạng")
    
    # Lấy top_n data
    top_data = df_all.head(top_n)
    
    # Phần 1: 4 KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    top1_combined = df_all.iloc[0]
    top1_v1 = df_v1.iloc[0]
    top1_v2 = df_v2.iloc[0]
    
    if "Combined" in methods:
        with col1:
            st.metric(
                "🥇 Top 1 – Tổng hợp",
                top1_combined["Country"],
                f"Rank V1={int(top1_combined['Rank_V1'])} · V2={int(top1_combined['Rank_V2'])}"
            )
    
    if "Vector (V1)" in methods:
        with col2 if "Combined" in methods else col1:
            st.metric(
                "🎯 Top 1 – Vector",
                top1_v1["Country"],
                f"DTW={top1_v1['DTW']:.4f}"
            )
    
    if "JSD (V2)" in methods:
        with col3 if "Combined" in methods else (col2 if "Vector (V1)" in methods else col1):
            st.metric(
                "📊 Top 1 – JSD",
                top1_v2["Country"],
                f"JSD={top1_v2['JSD']:.4f}"
            )
    
    # Đồng thuận: nước xuất hiện top_n ở cả V1 và V2
    if "Vector (V1)" in methods and "JSD (V2)" in methods:
        top_v1_set = set(df_v1.head(top_n)["Country"].values)
        top_v2_set = set(df_v2.head(top_n)["Country"].values)
        consensus = len(top_v1_set & top_v2_set)
        with col4 if len(methods) >= 3 else col3:
            st.metric(
                "🤝 Đồng thuận",
                f"{consensus} quốc gia",
                f"Xuất hiện top {top_n} ở cả V1 lẫn V2"
            )
    
    st.divider()
    
    # Phần 2: Grouped bar chart - so sánh rank
    methods_display = ", ".join(methods) if methods else "Không có phương pháp nào được chọn"
    st.subheader(f"Xếp hạng So Sánh: {methods_display}")
    
    if not methods:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 phương pháp so sánh")
    else:
        fig_bars = go.Figure()
        
        # Chỉ thêm traces cho phương pháp được chọn
        if "Vector (V1)" in methods:
            fig_bars.add_trace(go.Bar(
                x=top_data["Country"],
                y=top_data["Rank_V1"],
                name="Vector (V1)",
                marker_color=COLOR_V1,
                text=[f"{int(x)}" for x in top_data["Rank_V1"]],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{x}</b><br>Rank V1: %{y}<extra></extra>"
            ))
        
        if "JSD (V2)" in methods:
            fig_bars.add_trace(go.Bar(
                x=top_data["Country"],
                y=top_data["Rank_V2"],
                name="JSD (V2)",
                marker_color=COLOR_V2,
                text=[f"{int(x)}" for x in top_data["Rank_V2"]],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{x}</b><br>Rank V2: %{y}<extra></extra>"
            ))
        
        if "Combined" in methods:
            fig_bars.add_trace(go.Bar(
                x=top_data["Country"],
                y=top_data["Rank_Combined"],
                name="Combined",
                marker_color=COLOR_COMBINED,
                text=[f"{x:.1f}" for x in top_data["Rank_Combined"]],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{x}</b><br>Rank Combined: %{y:.1f}<extra></extra>"
            ))
        
        fig_bars.update_layout(
            barmode="group",
            yaxis=dict(autorange="reversed"),
            height=400,
            font_family="Arial",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig_bars, use_container_width=True)
        st.caption("💡 Rank thấp = tương đồng cao. Rank được tính lại từ scores của phương pháp được chọn.")
    
    st.divider()
    
    # Phần 3: Scatter + Radar
    col_scatter, col_radar = st.columns(2)
    
    with col_scatter:
        st.subheader("Đồng Thuận 2 Phương Pháp")
        st.caption("Vùng xanh lá = nước xuất hiện Top N ở cả V1 lẫn V2")
        
        fig_scatter = go.Figure()
        
        # Tất cả điểm: màu xám
        fig_scatter.add_trace(go.Scatter(
            x=df_all["Rank_V1"],
            y=df_all["Rank_V2"],
            mode="markers",
            marker=dict(
                color="#d1d5db",
                size=5,
                opacity=0.5
            ),
            name="Các nước khác",
            customdata=df_all["Country"],
            hovertemplate="<b>%{customdata}</b><br>Rank V1: %{x}<br>Rank V2: %{y}<extra></extra>"
        ))
        
        # Top N: màu riêng
        top_colors = [COLORS_TOP5[i] if i < len(COLORS_TOP5) else "#95a5a6"
                      for i in range(len(top_data))]
        
        fig_scatter.add_trace(go.Scatter(
            x=top_data["Rank_V1"],
            y=top_data["Rank_V2"],
            mode="markers+text",
            marker=dict(
                color=top_colors,
                size=10,
                opacity=0.85
            ),
            text=top_data["Country"],
            textposition="top center",
            textfont=dict(size=8),
            name=f"Top {top_n}",
            customdata=top_data["Country"],
            hovertemplate="<b>%{customdata}</b><br>Rank V1: %{x}<br>Rank V2: %{y}<extra></extra>"
        ))
        
        # Đường chéo y=x
        lim = max(df_all["Rank_V1"].max(), df_all["Rank_V2"].max()) + 5
        fig_scatter.add_trace(go.Scatter(
            x=[0, lim],
            y=[0, lim],
            mode="lines",
            line=dict(color="#d1d5db", dash="dash", width=1),
            name="y=x",
            hoverinfo="skip"
        ))
        
        # Highlight vùng consensus (vùng xanh lá)
        fig_scatter.add_shape(
            type="rect",
            x0=0.5, y0=0.5,
            x1=top_n+2, y1=top_n+2,
            fillcolor="rgba(22,163,74,0.07)",
            line=dict(color="#16a34a", dash="dot", width=1),
            name="Consensus zone"
        )
        
        fig_scatter.update_layout(
            xaxis_title="Rank – Vector (V1)",
            yaxis_title="Rank – JSD (V2)",
            height=450,
            font_family="Arial",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="v", yanchor="top", y=0.99, x=0.01)
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_radar:
        st.subheader("Radar: Trung Bình 20 Năm")
        st.caption("VNM (đen dày) vs Top 5 tương đồng")
        
        # Normalize features về [0,1]
        sc_r = MinMaxScaler()
        all_features = np.vstack([
            panel_raw[c][FEATURES].mean().values
            for c in ["VNM"] + TOP5
        ])
        sc_r.fit(all_features)
        
        fig_radar = go.Figure()
        
        # VNM
        vnm_vals = sc_r.transform(panel_raw["VNM"][FEATURES].mean().values.reshape(1, -1))[0]
        vnm_vals_plot = vnm_vals.tolist() + [vnm_vals[0]]
        angles = np.linspace(0, 2*np.pi, len(FEATURES), endpoint=False).tolist()
        angles += angles[:1]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=vnm_vals_plot,
            theta=[f.replace("_", "<br>") for f in FEATURES] + [FEATURES[0].replace("_", "<br>")],
            fill="toself",
            name="VNM",
            line=dict(color=COLOR_VNM, width=3),
            fillcolor="rgba(0,0,0,0.05)",
            opacity=0.8
        ))
        
        # Top 5
        for i, country in enumerate(TOP5):
            vals = sc_r.transform(panel_raw[country][FEATURES].mean().values.reshape(1, -1))[0]
            vals_plot = vals.tolist() + [vals[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_plot,
                theta=[f.replace("_", "<br>") for f in FEATURES] + [FEATURES[0].replace("_", "<br>")],
                fill="toself",
                name=country,
                line=dict(color=COLORS_TOP5[i], width=1.5),
                fillcolor=f"rgba({int(COLORS_TOP5[i][1:3],16)},{int(COLORS_TOP5[i][3:5],16)},{int(COLORS_TOP5[i][5:7],16)},0.1)",
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
            font_family="Arial",
            paper_bgcolor="white",
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.05)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CHUỖI THỜI GIAN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Chuỗi Thời Gian Các Chỉ Số Kinh Tế")
    st.caption("VNM (đen dày) vs các quốc gia tương đồng")
    
    # Multiselect chọn quốc gia
    default_countries = TOP5.copy() if TOP5 else []
    selected_countries = st.multiselect(
        "Chọn quốc gia để so sánh (VNM luôn hiển thị)",
        options=COUNTRIES_WITHOUT_VNM,
        default=default_countries,
        help="VNM luôn có trong danh sách"
    )
    
    # Cảnh báo nếu quá nhiều nước
    if len(selected_countries) > 8:
        st.warning(f"⚠️ Bạn chọn {len(selected_countries)} quốc gia. Giới hạn 8 để biểu đồ dễ đọc.")
        selected_countries = selected_countries[:8]
    
    # Lọc năm theo giai đoạn
    filtered_years = [y for y in YEARS if y >= year_start]
    
    # Lọc chỉ số
    active_features = [f for f in selected_features if f in FEATURES]
    
    if not active_features:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 chỉ số trong Bộ lọc")
    else:
        # Tạo grid subplots (2 hàng × 4 cột = 8 panels)
        n_features = len(active_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig_ts = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f.replace("_", " ") for f in active_features],
            shared_xaxes=False
        )
        
        # Vẽ line charts
        all_countries = ["VNM"] + selected_countries
        colors_map = {
            "VNM": COLOR_VNM,
            **{c: COLORS_TOP5[i % len(COLORS_TOP5)] for i, c in enumerate(selected_countries)}
        }
        widths = {c: 3 if c == "VNM" else 1.5 for c in all_countries}
        
        for feat_idx, feat in enumerate(active_features):
            row = feat_idx // n_cols + 1
            col = feat_idx % n_cols + 1
            
            for country in all_countries:
                if country not in panel_raw:
                    continue
                
                data_feat = panel_raw[country].loc[[y for y in filtered_years if y in panel_raw[country].index], feat]
                years_data = [y for y in filtered_years if y in panel_raw[country].index]
                
                if len(data_feat) == 0:
                    continue
                
                fig_ts.add_trace(
                    go.Scatter(
                        x=years_data,
                        y=data_feat.values,
                        name=country,
                        line=dict(
                            color=colors_map.get(country, "#9ca3af"),
                            width=widths.get(country, 1.5)
                        ),
                        showlegend=(feat_idx == 0),
                        hovertemplate=f"<b>{country}</b><br>{feat}<br>Năm: %{{x}}<br>Giá trị: %{{y:.2f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        # Cập nhật layout
        fig_ts.update_layout(
            height=200 * n_rows + 100,
            font_family="Arial",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified"
        )
        
        # Ẩn subplot thừa
        for extra_idx in range(n_features, n_rows * n_cols):
            row = extra_idx // n_cols + 1
            col = extra_idx % n_cols + 1
            fig_ts.update_xaxes(visible=False, row=row, col=col)
            fig_ts.update_yaxes(visible=False, row=row, col=col)
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    st.info("💡 Chọn tối đa 8 quốc gia cùng lúc để biểu đồ dễ đọc")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SO SÁNH CHI TIẾT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 So Sánh Chi Tiết")
    
    col_dt, col_table = st.columns([1, 1])
    
    with col_dt:
        st.subheader("Scatter: DTW vs JSD")
        st.caption("Vùng trái-dưới = tương đồng cả hai chiều")
        
        fig_dtj = go.Figure()
        
        # Tất cả điểm
        fig_dtj.add_trace(go.Scatter(
            x=df_all["DTW"],
            y=df_all["JSD"],
            mode="markers",
            marker=dict(
                color="#d1d5db",
                size=5,
                opacity=0.5
            ),
            name="Các nước khác",
            customdata=df_all["Country"],
            hovertemplate="<b>%{customdata}</b><br>DTW: %{x:.4f}<br>JSD: %{y:.4f}<extra></extra>"
        ))
        
        # Top N
        top_colors_dtj = [COLORS_TOP5[i] if i < len(COLORS_TOP5) else "#95a5a6"
                          for i in range(len(top_data))]
        
        fig_dtj.add_trace(go.Scatter(
            x=top_data["DTW"],
            y=top_data["JSD"],
            mode="markers+text",
            marker=dict(
                color=top_colors_dtj,
                size=10,
                opacity=0.85
            ),
            text=top_data["Country"],
            textposition="top center",
            textfont=dict(size=8),
            name=f"Top {top_n}",
            hovertemplate="<b>%{text}</b><br>DTW: %{x:.4f}<br>JSD: %{y:.4f}<extra></extra>"
        ))
        
        fig_dtj.update_layout(
            xaxis_title="DTW Distance (Vector)",
            yaxis_title="Jensen-Shannon Divergence",
            height=500,
            font_family="Arial",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=30, b=40)
        )
        
        st.plotly_chart(fig_dtj, use_container_width=True)
    
    with col_table:
        st.subheader("Bảng Xếp Hạng Top 20")
        
        top20 = df_all.head(20)[["Final_Rank","Country","Rank_V1","Rank_V2",
                                  "Rank_Combined","DTW","Cosine","Euclidean","JSD"]]
        
        # Format số liệu
        display_df = top20.copy()
        for col in ["DTW","Cosine","Euclidean","JSD"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_df,
            column_config={
                "Final_Rank": st.column_config.NumberColumn("#", format="%d"),
                "Country": "Quốc gia",
                "Rank_V1": st.column_config.NumberColumn("V1", format="%d"),
                "Rank_V2": st.column_config.NumberColumn("V2", format="%d"),
                "Rank_Combined": st.column_config.NumberColumn("Combined", format="%.1f"),
                "DTW": "DTW",
                "Cosine": "Cosine",
                "Euclidean": "Euclidean",
                "JSD": "JSD",
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PHÂN PHỐI (JSD)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📉 Phân Phối Các Chỉ Số (JSD)")
    st.caption("Minh họa tại sao JSD cho kết quả khác Vector – bằng cách so sánh phân phối xác suất")
    
    col_select, col_info = st.columns([2, 1])
    
    with col_select:
        # Dropdown chọn quốc gia
        compare_country = st.selectbox(
            "Chọn quốc gia để so sánh phân phối với VNM",
            options=top_data["Country"].tolist(),
            index=0,
            help="So sánh phân phối các chỉ số giữa VNM và quốc gia được chọn"
        )
    
    with col_info:
        # Hiển thị JSD score
        jsd_val = df_all[df_all["Country"] == compare_country]["JSD"].values
        jsd_val = jsd_val[0] if len(jsd_val) > 0 else 0
        st.metric("JSD Score", f"{jsd_val:.4f}", "Giá trị thấp = phân phối gần nhất")
    
    # Chọn chỉ số
    feat_choice = st.selectbox("Chỉ số kinh tế", FEATURES, help="Chọn chỉ số để hiển thị")
    
    # Lấy dữ liệu
    vnm_values = panel_raw["VNM"][feat_choice].values
    country_values = panel_raw[compare_country][feat_choice].values
    
    # Tạo histogram
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=vnm_values,
        name="VNM",
        opacity=0.6,
        nbinsx=10,
        marker_color=COLOR_VNM,
        hovertemplate="VNM<br>%{x:.2f}<br>Count: %{y}<extra></extra>"
    ))
    
    fig_hist.add_trace(go.Histogram(
        x=country_values,
        name=compare_country,
        opacity=0.6,
        nbinsx=10,
        marker_color=COLORS_TOP5[top_data["Country"].tolist().index(compare_country) % len(COLORS_TOP5)],
        hovertemplate=f"{compare_country}<br>%{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
    ))
    
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title=f"{feat_choice} (Giá trị thô, chưa scale)",
        yaxis_title="Tần suất",
        height=400,
        font_family="Arial",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.info(
        "💡 **Tại sao phân phối quan trọng?**\n"
        "Vector methods (DTW, Cosine) so sánh trực tiếp giá trị các chỉ số.\n"
        "JSD so sánh *phân phối* của từng chỉ số – có thể bắt được các mẫu vẫn tương đồng ngay cả khi giá trị tuyệt đối khác nhau."
    )


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🇻🇳 **Quỹ đạo kinh tế tương đồng với Việt Nam** | "
    "Dữ liệu: World Bank | "
    f"Phân tích: 2004–2023 ({len(YEARS)} năm) | "
    f"Quốc gia: {len(COUNTRIES)} | "
    f"Chỉ số: {len(FEATURES)}"
)
