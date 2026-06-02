"""
ĐỒ ÁN: PHÂN CỤM CẤU TRÚC VĨ MÔ TOÀN CẦU VÀ KHUYẾN NGHỊ CHÍNH SÁCH
Mô hình: K-Means Clustering + PCA + ANOVA + Euclidean Distance
Quy trình: 11 Bước tự động hóa hoàn toàn.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
import warnings

# Bỏ qua các cảnh báo thống kê không cần thiết
warnings.filterwarnings('ignore')

def main():
    print("=" * 75)
    print("--- KHỞI ĐỘNG HỆ THỐNG PHÂN TÍCH VĨ MÔ (MACROECONOMIC CLUSTERING) ---")
    print("=" * 75)

    # =========================================================
    # CẤU HÌNH ĐƯỜNG DẪN TUYỆT ĐỐI (THAY ĐỔI THEO ĐƯỜNG DẪN TRÊN MÁY CỦA BẠN)
    # =========================================================
    INPUT_DATA_PATH = r"../DATA_AFTER_PREPROCESSING/data_final_merged.csv"
    OUTPUT_FOLDER = r"./plots"

    # =========================================================
    # BƯỚC 1: KIỂM TRA & THIẾT LẬP THƯ MỤC OUTPUT TUYỆT ĐỐI
    # =========================================================
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"[*] Đã tạo thư mục lưu kết quả tại: {OUTPUT_FOLDER}")

    # =========================================================
    # BƯỚC 2: NẠP DỮ LIỆU TỪ ĐƯỜNG DẪN TUYỆT ĐỐI
    # =========================================================
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"[!] LỖI: Không tìm thấy file dữ liệu tại đường dẫn tuyệt đối: {INPUT_DATA_PATH}")
        return

    df = pd.read_csv(INPUT_DATA_PATH)
    print(f"[*] Nạp dữ liệu thành công. Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.")

    features = ['Agri_GDP', 'Elec_Access', 'FDI_GDP', 'GDPC_2015', 'Indus_GDP', 'Internet_Usage', 'Serv_GDP']
    X = df[features]

    # =========================================================
    # BƯỚC 3: TIỀN XỬ LÝ (STANDARD SCALER)
    # =========================================================
    print("[*] Đang chuẩn hóa ma trận 7 chiều (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================================================
    # BƯỚC 4: TÌM K TỐI ƯU & HUẤN LUYỆN K-MEANS
    # =========================================================
    print("[*] Đang đo lường WCSS và Silhouette Score...")
    wcss, silhouette_scores = [], []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # Vẽ biểu đồ Đánh giá K
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
    ax1.set_ylabel('WCSS (Inertia)', color='tab:blue', fontweight='bold')
    ax1.plot(K_range, wcss, marker='o', color='tab:blue', linewidth=2)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette Score', color='tab:orange', fontweight='bold')
    ax2.plot(K_range, silhouette_scores, marker='s', color='tab:orange', linewidth=2)
    plt.title('Optimal Cluster Analysis Using Elbow and Silhouette Scores', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_FOLDER, '1_kmeans_evaluation.png'), dpi=300)
    plt.close()

    k_optimal = 3
    print(f"---> Quyết định: Huấn luyện chính thức với K = {k_optimal} <---")
    kmeans_model = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
    df['Cluster'] = kmeans_model.fit_predict(X_scaled)

    # =========================================================
    # BƯỚC 5: KIỂM ĐỊNH ANOVA
    # =========================================================
    print("\n[*] Đang kiểm định ANOVA đa biến...")
    anova_results = []
    for feature in features:
        groups = [df[df['Cluster'] == i][feature].values for i in range(k_optimal)]
        f_stat, p_val = stats.f_oneway(*groups)
        anova_results.append({
            'Feature': feature, 
            'F-Statistic': round(f_stat, 2), 
            'p-value': p_val,
            'Significance': "Khác biệt có ý nghĩa" if p_val < 0.05 else "Không"
        })
    
    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(os.path.join(OUTPUT_FOLDER, '2_anova_results.csv'), index=False, encoding='utf-8-sig')
    print(" -> Đã lưu kết quả ANOVA (2_anova_results.csv)")

    # =========================================================
    # BƯỚC 6: GIẢM CHIỀU PCA & VẼ BẢN ĐỒ
    # =========================================================
    print("[*] Đang nén không gian 7D xuống 2D (PCA)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', alpha=0.7)
    plt.title(f'Macroeconomic Clustering Map (PCA Projection - K={k_optimal})', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_FOLDER, '3_pca_clusters_2d.png'), dpi=300)
    plt.close()

    # =========================================================
    # BƯỚC 7 & 8: PHÂN TÍCH TÂM CỤM & LƯU DATA LÕI
    # =========================================================
    centroids = df[features + ['Cluster']].groupby('Cluster').mean().round(2)
    centroids_t = centroids.T
    centroids_t.to_csv(os.path.join(OUTPUT_FOLDER, '4_cluster_centroids.csv'), index=True, encoding='utf-8-sig')
    df.to_csv(os.path.join(OUTPUT_FOLDER, '5_data_final_clustered.csv'), index=False)
    
    # Tự động tìm Cụm Đại diện cho "Nước Phát triển" (Cụm có GDP cao nhất)
    developed_cluster_id = centroids['GDPC_2015'].idxmax()

    # =========================================================
    # BƯỚC 9: GAP ANALYSIS (PHÂN TÍCH KHOẢNG CÁCH CHÍNH SÁCH)
    # =========================================================
    target_country = 'VNM'
    target_year = 2023
    print(f"\n[*] Đang phân tích Gap Analysis cho {target_country} (Năm {target_year})...")
    
    vn_data = df[(df['Country'] == target_country) & (df['Year'] == target_year)]
    if not vn_data.empty:
        X_vn = vn_data[features]
        vn_cluster = kmeans_model.predict(scaler.transform(X_vn))[0]
        target_centroid = centroids.loc[developed_cluster_id]

        # Vẽ biểu đồ Cột Kép
        pct_features = ['Agri_GDP', 'Elec_Access', 'FDI_GDP', 'Indus_GDP', 'Internet_Usage', 'Serv_GDP']
        curr_pct = [X_vn.iloc[0][f] for f in pct_features]
        tgt_pct = [target_centroid[f] for f in pct_features]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
        x = np.arange(len(pct_features))
        
        ax1.bar(x - 0.2, curr_pct, 0.4, label=f'{target_country} ({target_year})', color='tab:blue')
        ax1.bar(x + 0.2, tgt_pct, 0.4, label=f'Mục tiêu (Cụm {developed_cluster_id} - Phát triển)', color='tab:orange')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pct_features, rotation=15)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.set_title('Khuyến nghị Chuyển dịch Cơ cấu (%)', fontweight='bold')

        ax2.bar([0], [X_vn.iloc[0]['GDPC_2015']], 0.4, color='tab:blue')
        ax2.bar([1], [target_centroid['GDPC_2015']], 0.4, color='tab:orange')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([target_country, f'Mục tiêu'])
        ax2.set_title('Khoảng cách GDP', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'6_gap_analysis_{target_country}.png'), dpi=300)
        plt.close()

    # =========================================================
    # BƯỚC 10: TIME TRAJECTORY (QUỸ ĐẠO LỊCH SỬ)
    # =========================================================
    print(f"[*] Đang vẽ quỹ đạo di chuyển của {target_country} trên bản đồ PCA...")
    history_data = df[df['Country'] == target_country].sort_values(by='Year')
    
    if not history_data.empty:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', alpha=0.15)
        
        pca_x, pca_y, years = [], [], []
        for _, row in history_data.iterrows():
            coords = pca.transform(scaler.transform(pd.DataFrame([row[features]])))[0]
            pca_x.append(coords[0])
            pca_y.append(coords[1])
            years.append(int(row['Year']))
            
        plt.plot(pca_x, pca_y, color='red', linewidth=2, marker='o', markersize=5)
        plt.text(pca_x[0], pca_y[0]+0.1, f"Bắt đầu ({years[0]})", color='darkred', fontweight='bold')
        plt.text(pca_x[-1], pca_y[-1]+0.1, f"Hiện tại ({years[-1]})", color='red', fontweight='bold')
        plt.title(f'Quỹ đạo Tiến hóa Kinh tế: {target_country} ({years[0]}-{years[-1]})', fontweight='bold')
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'7_time_trajectory_{target_country}.png'), dpi=300)
        plt.close()

    # =========================================================
    # BƯỚC 12: ĐỘNG LỰC TĂNG TRƯGROWTH (GDP vs CLUSTER TRANSITION)
    # =========================================================
    print(f"\n[*] Đang vẽ biểu đồ Động lực Tăng trưởng GDP cho {target_country}...")
    
    if not history_data.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        years_list = history_data['Year'].values
        gdp_list = history_data['GDPC_2015'].values
        clusters_list = history_data['Cluster'].values
        
        ax.plot(years_list, gdp_list, color='grey', linestyle='--', alpha=0.5)
        scatter = ax.scatter(years_list, gdp_list, c=clusters_list, cmap='viridis', s=100, zorder=5)
        
        prev_c = clusters_list[0]
        for i in range(1, len(clusters_list)):
            curr_c = clusters_list[i]
            if curr_c != prev_c:
                ax.annotate(f'Nhảy sang Cụm {curr_c}\n({years_list[i]})', 
                            xy=(years_list[i], gdp_list[i]), 
                            xytext=(years_list[i]-2, gdp_list[i]+(gdp_list[-1]*0.1)),
                            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                            fontsize=10, fontweight='bold', color='red')
            prev_c = curr_c
            
        ax.set_title(f'Bằng chứng Tăng trưởng: Lịch sử Nhảy Cụm Cấu Trúc và GDP của {target_country}', fontweight='bold')
        ax.set_xlabel('Năm')
        ax.set_ylabel('GDP Bình quân (GDPC_2015 - USD)')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        legend1 = ax.legend(*scatter.legend_elements(), title="Thuộc Cụm", loc="upper left")
        ax.add_artist(legend1)
        
        gdp_cluster_path = os.path.join(OUTPUT_FOLDER, f'10_gdp_growth_vs_clusters_{target_country}.png')
        plt.savefig(gdp_cluster_path, dpi=300)
        plt.close()
        print(f" -> Đã xuất biểu đồ Động lực Tăng trưởng tại: {gdp_cluster_path}")

    # =========================================================
    # BƯỚC 11: PEER ANALYSIS & VISUALIZATION (TÌM QUỐC GIA TƯƠNG ĐỒNG NHẤT)
    # =========================================================
    print("\n" + "=" * 70)
    print("--- ỨNG DỤNG 3: NHẬN DIỆN VÀ TRỰC QUAN HÓA QUỐC GIA TƯƠNG ĐỒNG ---")
    
    target_country = 'VNM'  
    target_year = 2023      
    
    target_data = df[(df['Country'] == target_country) & (df['Year'] == target_year)]
    
    if not target_data.empty:
        print(f"[*] Đang dò tìm Top 5 quốc gia có cấu trúc giống {target_country} nhất...")
        
        X_target = target_data[features]
        X_target_scaled = scaler.transform(X_target)
        target_cluster = kmeans_model.predict(X_target_scaled)[0]
        
        peers_data = df[(df['Cluster'] == target_cluster) & 
                        (df['Year'] == target_year) & 
                        (df['Country'] != target_country)].copy()
        
        if not peers_data.empty:
            X_peers = peers_data[features]
            X_peers_scaled = scaler.transform(X_peers)
            distances = euclidean_distances(X_target_scaled, X_peers_scaled)[0]
            peers_data['Distance'] = distances
            
            top_5 = peers_data.sort_values(by='Distance').head(5)
            
            peers_out_path = os.path.join(OUTPUT_FOLDER, f'8_similar_peers_{target_country}.csv')
            top_5[['Country', 'Distance'] + features].to_csv(peers_out_path, index=False)
            
            print("\n" + "=" * 60)
            print(f"🏆 TOP 5 QUỐC GIA GIỐNG {target_country} NHẤT (NĂM {target_year}):")
            print("-" * 60)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"Top {i}: {row['Country']:<5} | Độ lệch không gian: {row['Distance']:.4f}")

            print(f"[*] Đang xuất biểu đồ trực quan hóa độ tương đồng...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]})
            
            plot_data = top_5.sort_values(by='Distance', ascending=False)
            sns.barplot(x='Distance', y='Country', data=plot_data, palette='Reds_r', ax=ax1)
            ax1.set_title(f'Khoảng cách Euclidean tới {target_country}\n(Càng ngắn càng giống)', fontweight='bold')
            ax1.set_xlabel('Độ lệch không gian (Distance)')
            ax1.set_ylabel('Quốc gia')
            
            for p in ax1.patches:
                width = p.get_width()
                ax1.text(width + 0.02, p.get_y() + p.get_height()/2. + 0.1, 
                         f'{width:.2f}', ha="left", fontsize=10)

            heatmap_df = pd.concat([target_data, top_5])
            heatmap_scaled = scaler.transform(heatmap_df[features])
            heatmap_plot_data = pd.DataFrame(heatmap_scaled, columns=features, index=heatmap_df['Country'])
            
            sns.heatmap(heatmap_plot_data, cmap='coolwarm', annot=True, fmt=".2f", 
                        linewidths=.5, ax=ax2, cbar_kws={'label': 'Z-score (Chuẩn hóa)'})
            ax2.set_title(f'Bản đồ Nhiệt Cấu trúc Vĩ mô: {target_country} vs Top 5 Tương đồng', fontweight='bold')
            ax2.set_ylabel('')
            ax2.tick_params(axis='x', rotation=15)
            
            for tick in ax2.get_yticklabels():
                if tick.get_text() == target_country:
                    tick.set_color('red')
                    tick.set_fontweight('bold')

            plt.tight_layout()
            peer_chart_path = os.path.join(OUTPUT_FOLDER, f'9_peer_analysis_visualization_{target_country}.png')
            plt.savefig(peer_chart_path, dpi=300)
            plt.close()
            print(f" -> Đã xuất biểu đồ Peer Analysis tại: {peer_chart_path}")

    print("\n" + "=" * 75)
    print(f"🎉 HOÀN TẤT! TOÀN BỘ FILE KẾT QUẢ ĐÃ ĐƯỢC XUẤT VÀO THƯ MỤC:")
    print(f"---> {OUTPUT_FOLDER}")
    print("=" * 75)

if __name__ == "__main__":
    main()