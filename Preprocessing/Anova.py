import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu
df = pd.read_csv('GPU_10_Features_Processed_Final.csv')

# 2. Xử lý dữ liệu trống và lọc các hãng chính (NVIDIA, AMD, Intel)
print("--- Số lượng giá trị thiếu trước khi xử lý ---")
print(df[['Manufacturer', 'Is_Notebook', 'Memory_Bandwidth_GBs']].isnull().sum())

df_filtered = df.dropna(subset=['Memory_Bandwidth_GBs', 'Manufacturer', 'Is_Notebook']).copy()
df_filtered = df_filtered[df_filtered['Manufacturer'].isin(['NVIDIA', 'AMD', 'Intel'])]

# Tạo cột định danh 'Type' cho Is_Notebook (0: Desktop, 1: Laptop)
df_filtered['Type'] = df_filtered['Is_Notebook'].map({0: 'Desktop', 1: 'Laptop'})

print(f"\nSố lượng bản ghi hợp lệ sau khi lọc: {len(df_filtered)}")

# 3. Thống kê mô tả (Descriptive Statistics)
print("\n" + "="*50)
print("=== 3.1. THỐNG KÊ MÔ TẢ RIÊNG THEO HÃNG SẢN XUẤT (MANUFACTURER) ===")
stats_mfg = df_filtered.groupby('Manufacturer')['Memory_Bandwidth_GBs'].agg(
    Count='count', Mean='mean', Std='std', Min='min', Median='median', Max='max'
).reset_index()
print(stats_mfg.to_string(index=False))

print("\n" + "="*50)
print("=== 3.2. THỐNG KÊ MÔ TẢ RIÊNG THEO LOẠI THIẾT BỊ (DESKTOP VS LAPTOP) ===")
stats_type = df_filtered.groupby('Type')['Memory_Bandwidth_GBs'].agg(
    Count='count', Mean='mean', Std='std', Min='min', Median='median', Max='max'
).reset_index()
print(stats_type.to_string(index=False))

print("\n" + "="*50)
print("=== 3.3. THỐNG KÊ MÔ TẢ TỔ HỢP (MANUFACTURER x IS_NOTEBOOK) ===")
stats_comb = df_filtered.groupby(['Manufacturer', 'Type'])['Memory_Bandwidth_GBs'].agg(
    Count='count', Mean='mean', Std='std', Min='min', Median='median', Max='max'
).reset_index()
print(stats_comb.to_string(index=False))

# 4. Kiểm tra tiền điều kiện (Assumptions Check)
print("\n" + "="*50)
print("=== 4. KIỂM TRA TIỀN ĐIỀU KIỆN ANOVA ===")

# 4.1. Đồng nhất phương sai giữa các nhóm tổ hợp (Levene's Test)
group_data = [group['Memory_Bandwidth_GBs'].values for _, group in df_filtered.groupby(['Manufacturer', 'Is_Notebook'])]
levene_stat, levene_p = stats.levene(*group_data)
print(f"1. Levene's Test (Đồng nhất phương sai): Statistic = {levene_stat:.4f}, p-value = {levene_p:.4e}")
if levene_p < 0.05:
    print("   -> Phương sai giữa các nhóm KHÔNG đồng nhất (ở mức alpha = 0.05).")
else:
    print("   -> Phương sai giữa các nhóm đồng nhất (ở mức alpha = 0.05).")

# 4.2. Chạy mô hình OLS để lấy phần dư
model = ols('Memory_Bandwidth_GBs ~ C(Manufacturer) * C(Is_Notebook)', data=df_filtered).fit()

# 4.3. Kiểm tra tính phân phối chuẩn của phần dư (Shapiro-Wilk Test)
residuals = model.resid
shapiro_stat, shapiro_p = stats.shapiro(residuals if len(residuals) <= 5000 else residuals.sample(5000, random_state=42))
print(f"2. Shapiro-Wilk Test (Phân phối chuẩn của phần dư): Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4e}")
if shapiro_p < 0.05:
    print("   -> Phần dư KHÔNG tuân theo phân phối chuẩn tuyệt đối (ở mức alpha = 0.05).")
else:
    print("   -> Phần dư tuân theo phân phối chuẩn (ở mức alpha = 0.05).")

# 5. Kết quả Phân tích phương sai 2 yếu tố (Two-Way ANOVA)
print("\n" + "="*50)
print("=== 5. KẾT QUẢ TWO-WAY ANOVA CHÍNH THỨC ===")
anova_table = sm.stats.anova_lm(model, typ=2)

# Tính toán Mean Square (MS)
anova_table['MS'] = anova_table['sum_sq'] / anova_table['df']

# Lấy bậc tự do của phần dư để tính F crit
df_resid = anova_table.loc['Residual', 'df']

# Tính F crit (với alpha = 0.05)
anova_table['Fcrit'] = [
    stats.f.ppf(0.95, df, df_resid) if idx != 'Residual' else float('nan') 
    for idx, df in anova_table['df'].items()
]

# Đổi tên các cột hiển thị theo đúng yêu cầu
anova_table = anova_table.rename(columns={
    'sum_sq': 'SS', 
    'F': 'Fqs', 
    'PR(>F)': 'p-value'
})

# Lọc lại các cột cần hiển thị
display_columns = ['SS', 'df', 'MS', 'Fqs', 'Fcrit', 'p-value']
print(anova_table[display_columns])

# =====================================================================
# 6. VẼ SƠ ĐỒ INTERACTION PLOT
# =====================================================================
print("\n" + "="*50)
print("=== 6. ĐANG VẼ SƠ ĐỒ INTERACTION PLOT ===")

# Thiết lập kích thước khung hình
plt.figure(figsize=(9, 6))

# Vẽ biểu đồ tương tác sử dụng Seaborn Pointplot
# Pointplot rất phù hợp cho ANOVA vì nó nối các điểm trung bình và hiển thị khoảng tin cậy
sns.pointplot(
    data=df_filtered, 
    x='Type', 
    y='Memory_Bandwidth_GBs', 
    hue='Manufacturer',
    dodge=True,            # Tách nhẹ các điểm đè lên nhau để dễ nhìn
    markers=['o', 's', 'D'], # Hình dáng điểm (Tròn, Vuông, Thoi)
    capsize=.1,            # Kích thước thanh khoảng tin cậy
    err_kws={'linewidth': 1.5},
    palette='Set1'         # Bảng màu
)

# Trang trí biểu đồ
plt.title('Interaction Plot: Ảnh hưởng của Hãng sản xuất và Loại thiết bị\nlên Băng thông bộ nhớ', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Loại thiết bị (Desktop vs Laptop)', fontsize=12, fontweight='bold')
plt.ylabel('Trung bình Băng thông bộ nhớ (GB/s)', fontsize=12, fontweight='bold')
plt.legend(title='Hãng sản xuất', title_fontsize='11', fontsize='10')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Căn chỉnh lề
plt.tight_layout()

# Lưu biểu đồ thành file ảnh PNG
plt.savefig('Interaction_Plot_GPU.png', dpi=300)
print(" -> Đã lưu biểu đồ thành công vào file 'Interaction_Plot_GPU.png'!")

# Hiển thị biểu đồ lên màn hình
plt.show()