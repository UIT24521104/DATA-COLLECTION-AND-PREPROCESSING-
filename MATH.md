## 1. Phân đoạn Tiền xử lý (Preprocessing)

### 1.1. Chuẩn hóa Z-score (`StandardScaler`)

Sử dụng nhằm đưa các đặc trưng về cùng một quy mô (scale), triệt tiêu sự chênh lệch đơn vị.

- **Công thức toán học:**
  $$z = \frac{x - \mu}{\sigma}$$
- **Thành phần:** Với $x$ là giá trị gốc, $\mu$ là giá trị trung bình (mean) và $\sigma$ là độ lệch chuẩn (standard deviation) của đặc trưng đó trên toàn bộ dataset.

### 1.2. Chuẩn hóa Min-Max (`MinMaxScaler`)

Sử dụng để ép dữ liệu vào khoảng cố định nhằm phục vụ bài toán phân phối xác suất mà không làm thay đổi phân phối gốc.

- **Công thức toán học:**
  $$x_{scaled} = \epsilon + (1 - \epsilon) \times \frac{x - x_{min}}{x_{max} - x_{min}}$$
- **Thành phần:** Trong code, cấu hình `feature_range=(EPS, 1)` dịch chuyển miền giá trị về $[\epsilon, 1]$ với $\epsilon = 10^{-6}$ nhằm đảm bảo các giá trị sau chuẩn hóa luôn $> 0$.

### 1.3. Chuyển đổi Logarit (Log-Transform)

Được sử dụng để nén các đặc trưng có phân phối lệch phải mạnh (right-skewed) như GDP, kéo dữ liệu tiệm cận về phân phối chuẩn nhằm tối ưu hóa không gian cho các thuật toán dựa trên khoảng cách.

- **Công thức toán học:** Để tránh lỗi toán học khi tính $\log(0)$ hoặc với các số âm, một hằng số $c$ được cộng thêm vào (thường dùng biến thể $$\log(1+x)$$):
  $$x_{log} = \ln(x + c)$$

### 1.4. Nội suy đa biến dựa trên cây quyết định (Tree-based Imputation/ MICE)

Sử dụng mô hình học máy để dự đoán và điền khuyết giá trị bị thiếu dựa trên cấu trúc tương quan của tất cả các biến còn lại.
- **Cơ chế toán học (Iterative Regressor):** Tại bước lặp $t$, đặc trưng bị khuyết $X_j$ được dự đoán bởi hàm hồi quy cây $f$, với đầu vào là các đặc trưng khác đã được điền khuyết tạm thời:
  $$X_j^{(t)} = f\left(X_1^{(t)}, \dots, X_{j-1}^{(t)}, X_{j+1}^{(t-1)}, \dots, X_n^{(t-1)}\right)$$
---

## 2. Phương án 1: Vector Similarity (Hình thái chuỗi thời gian)

### 2.1. Thuật toán Dynamic Time Warping (`dtw.distance_fast`)

Dùng để đo lường khoảng cách hình thái giữa hai chuỗi thời gian có tính đến sự co giãn hoặc trễ pha theo thời gian.

- **Thuật toán:** Tìm một đường đi uốn lượn (warping path) $W = \{w_1, w_2, \dots, w_K\}$ trên ma trận khoảng cách giữa chuỗi $A$ và chuỗi $B$ sao cho tổng khoảng cách lũy kế là nhỏ nhất, thỏa mãn các điều kiện biên, tính liên tục và tính đơn điệu.
- **Công thức truy hồi (Quy hoạch động):**
  $$\gamma(i, j) = d(a_i, b_j) + \min\{\gamma(i-1, j), \gamma(i, j-1), \gamma(i-1, j-1)\}$$
- **Hàm đa biến trong code (`dtw_multi`):** Được tính bằng trung bình cộng khoảng cách DTW của từng đặc trưng đơn lẻ:
  
  $$\text{DTW}_{multi}(A, B) = \frac{1}{F} \sum_{f=1}^{F} \text{DTW}(A_{:, f}, B_{:, f})$$

### 2.2. Khoảng cách Cosine (`cosine`)

Đo hướng của hai vector phẳng (flattened vectors) đại diện cho toàn bộ ma trận chỉ số kinh tế qua các năm mà không phụ thuộc vào độ dài/quy mô tuyệt đối.

- **Công thức toán học:**

$$D_{\text{Cosine}}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2} = 1 - \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

### 2.3. Khoảng cách Euclidean (`euclidean`)

Đo lường khoảng cách thẳng (độ lệch tuyệt đối bình phương) giữa hai vector phẳng trong không gian đa chiều.

- **Công thức toán học:**

$$D_{\text{Euclidean}}(u, v) = \|u - v\|_2 = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

---

## 3. Phương án 2: Jensen-Shannon Divergence (Cấu trúc phân phối)

### 3.1. Phân phối xác suất thực nghiệm (`np.histogram` & `hist_prob`)

Chuyển đổi chuỗi giá trị liên tục thành một vector phân phối xác suất rời rạc thông qua kỹ thuật chia thùng (binning).

- **Công thức toán học:** Với $h_k$ là tần suất của thùng thứ $k$ và $\epsilon = 10^{-6}$, xác suất $p_k$ được làm mượt (smoothing) để tránh giá trị 0:
  $$p_k = \frac{h_k + \epsilon}{\sum_{j=1}^{\text{bins}} (h_j + \epsilon)}$$

### 3.2. Kiểm sai Kullback-Leibler (`rel_entr`)

Là nền tảng tính toán độ phân kỳ thông tin (Entropy tương đối) giữa hai phân phối xác suất $P$ và $Q$.

- **Công thức toán học:**
  $$D_{\text{KL}}(P \parallel Q) = \sum_{k=1}^{\text{bins}} p_k \log \left( \frac{p_k}{q_k} \right)$$

### 3.3. Độ phân kỳ Jensen-Shannon (`jsd`)

Sử dụng để đo độ lệch phân phối một cách đối xứng ($$JSD(P \parallel Q) = JSD(Q \parallel P)$$) và có giới hạn bảo toàn.

- **Công thức toán học:** Gọi $M$ là phân phối trung bình $M = \frac{1}{2}(P + Q)$, ta có:

  $$JSD(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$

- **Hàm đa biến trong code (`jsd_multi`):** Là trung bình cộng chỉ số JSD của phân phối thực nghiệm trên tất cả 7 đặc trưng.

---
## 4. Tích hợp kết quả (Ranking Fusion)

### 4.1. Gộp thứ hạng theo Borda Count (Phương án 1)

Do các khoảng cách DTW, Cosine và Euclidean có miền giá trị và tỷ lệ (scale) hoàn toàn khác nhau, thuật toán sử dụng cơ chế xếp hạng không tham số (tương tự phương pháp Borda Count) thay vì cộng trực tiếp giá trị tuyệt đối. Tổng các thứ hạng thành phần sẽ tạo ra điểm số đánh giá độ tương đồng hình thái $\text{Score}_{V1}$:

$$\text{Score}_{V1} = \text{Rank}(\text{DTW}) + \text{Rank}(\text{Cosine}) + \text{Rank}(\text{Euclidean})$$

*(Lưu ý: Quốc gia có $\text{Score}_{V1}$ càng nhỏ tức là xếp hạng càng cao, thể hiện hình thái chuỗi thời gian càng giống với quốc gia mục tiêu).*

### 4.2. Xếp hạng Tích hợp Toàn diện (Rank Combined)

Sử dụng trung bình toán học để dung hòa hai góc nhìn phân tích hoàn toàn độc lập: thứ hạng về mặt biến động hình thái theo thời gian (V1) và thứ hạng về bản chất cấu trúc phân phối nội tại (V2 - JSD). Phép toán này triệt tiêu độ lệch của từng phương pháp đơn lẻ, cho ra thứ hạng độ tương đồng (Similarity Ranking) cuối cùng:

$$\text{Rank}_{Combined} = \frac{\text{Rank}_{V1} + \text{Rank}_{V2}}{2}$$

---

## 5. Mô hình Phân cụm (Clustering)

### 5.1. Thuật toán K-Means ($K=3$)

Sử dụng để phân tách không gian quốc gia thành 3 dải kinh tế tách biệt dựa trên nguyên lý cực tiểu hóa phương sai nội cụm.

- **Hàm mục tiêu WCSS (Within-Cluster Sum of Squares):** Thuật toán tìm kiếm các tâm cụm $\mu_k$ sao cho tổng bình phương khoảng cách từ mỗi điểm dữ liệu $x_i$ đến tâm cụm tương ứng của nó $C_k$ là nhỏ nhất:
$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

### 5.2. Phân tích Thành phần Chính (PCA - Principal Component Analysis)

Được sử dụng để nén không gian đa chiều (7 đặc trưng) xuống còn 2 chiều (2D) nhằm trực quan hóa các cụm K-Means, đồng thời giữ lại tối đa lượng phương sai (thông tin) gốc.

- **Công thức toán học:** PCA tìm các vector riêng (eigenvectors) $w$ và giá trị riêng (eigenvalues) $\lambda$ từ ma trận hiệp phương sai $C$ của dữ liệu:

$$C w = \lambda w$$

Dữ liệu sau đó được chiếu lên các vector riêng có giá trị riêng lớn nhất để tạo ra các tọa độ mới $Z$:
$$Z = X W$$

### 6. Kiểm định Thống kê (Statistical Validation)

### 6.1. Kiểm định Phân tích Phương sai (ANOVA - Analysis of Variance)

Sử dụng làm bước Post-processing Validation nhằm xác nhận về mặt toán học ($p-value \approx 0$) rằng 3 cụm do K-Means sinh ra có sự khác biệt thực sự về bản chất, không phải do ngẫu nhiên.

Thống kê F (F-statistic): Là tỷ số giữa phương sai giữa các nhóm (Between-group variance) và phương sai trong nội bộ nhóm (Within-group variance):

$$F = \frac{\text{Variance Between}}{\text{Variance Within}} = \frac{\frac{SSG}{K-1}}{\frac{SSE}{N-K}}$$

Trong đó: $SSG$ là tổng bình phương độ lệch giữa các cụm, $SSE$ là tổng bình phương sai số nội cụm, $N$ là tổng số mẫu, và $K$ là số cụm.
