from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

#  Khởi tạo
CSV_FILE = "../../GPU_10_Features_Processed_Final.csv"
GROUPS = ["DDR", "DDR2", "DDR3", "GDDR3", "GDDR5"]
VALUE, GROUP = "Memory_Bandwidth_GBs", "Memory_Type"
alpha = 0.05
# Bước 1 và 2: Lọc dữ liệu & Xếp hạng
df = pd.read_csv(CSV_FILE)
df = df[df[GROUP].isin(GROUPS)]

print(f"Số dòng 5 MMR Type trước khi xử lý băng thông thiếu: {len(df)}")
print("Dòng thiếu băng thông theo nhóm:",
      df[df[VALUE].isna()][GROUP].value_counts().to_dict())

df = df.dropna(subset=[VALUE]).copy()
N = len(df)
k = len(GROUPS)
n = df.groupby(GROUP)[VALUE].size().loc[GROUPS]
print(n.to_string())

# Xếp hạng chung, ties lấy hạng trung bình
df["rank"] = df[VALUE].rank(method="average")

#  Bước 3: Tổng hạng & Mean Rank
R = df.groupby(GROUP)["rank"].sum().loc[GROUPS]
mean_rank = R/n

print("\nTổng hạng và hạng trung bình")
print(pd.DataFrame({"n": n, "R_i": R, "mean_rank": mean_rank}).round(2).to_string())
#  Bước 4: Thống kê H chưa qua hiệu chỉnh
sum_term = (R**2/n).sum()
H_raw = 12 / (N*(N+ 1)) * sum_term - 3 * (N + 1)# công thức tính H
print(f"\nsum(R_i^2/n_i)={sum_term:,.2f}")
print(f"H nothc=12/({N}*{N + 1}) * {sum_term:,.2f} - 3*{N + 1} = {H_raw:.3f}")

#  Bước 5: Hiệu chỉnh Ties (C)
t = df[VALUE].value_counts()
T= (t**3-t).sum()
C= 1-T/(N ** 3 - N)
H =H_raw/C

print(f"\nTinh toan muc do trung sum(t^3 - t) = {T:,.0f}, N^3 - N = {N ** 3 - N:,}")
print(f"        C = {C:.8f} -> H hiệu chỉnh = {H_raw:.3f} / {C:.8f} = {H:.3f}")

#  Bước 6: Kiểm định Kruskal-Wallis
dof = k - 1
crit = stats.chi2.ppf(1 - alpha, dof)# xử dung chi bình phương
p_kw = stats.chi2.sf(H, dof)
eps2 = H * (N + 1) / (N ** 2 - 1)

print(f"\n df = {dof}, chi binh phuon tới hạn ({alpha}) = {crit:.3f}")
print(f"p-value = {p_kw:.3e}")
print(f"epsilon^2 = {eps2:.3f}")
print("        ", "BÁC BỎ H0: Có khác biệt tổng thể" if p_kw < alpha
      else "KHÔNG bác bỏ H0.")
# nói về cái này trong phần nhận xét kêt quả
# Đối chiếu với scipy
H_sp, p_sp = stats.kruskal(*[df.loc[df[GROUP] == g, VALUE] for g in GROUPS])
print(f"[Đối chiếu scipy] H = {H_sp:.3f}, p = {p_sp:.3e}")

#  Kiểm định Dunn (Bonferroni)
m = k * (k - 1) // 2
base = N * (N +1)/12-T/(12 *(N-1))

print(f"\n Dunn, số cặp m = {m}, phần chung = {base:,.3f}, "
      f"ngưỡng Bonferroni = {alpha}/{m} = {alpha / m}")

rows = []
for a, b in combinations(GROUPS, 2):
    diff = mean_rank[a] - mean_rank[b]
    se = np.sqrt(base * (1 / n[a] + 1 / n[b]))
    z = diff / se
    p_raw = 2 * stats.norm.sf(abs(z))# miền bác bỏ ppc của dunn
    p_adj = min(1.0, p_raw * m)
    rows.append({
        "Cap": f"{a} - {b}",
        "Hieu hang TB": diff,
        "SE": se,
        "z": z,
        "p tho": p_raw,
        "p Bonferroni": p_adj,
        "Khac biet": "CO" if p_adj < alpha else "khong"
    })

res = pd.DataFrame(rows)

pd.set_option("display.width", 200)
print(res.to_string(
    index=False,
    formatters={
        "SE": "{:.2f}".format,
        "z": "{:.3f}".format,
        "p tho": "{:.3e}".format,
        "p Bonferroni": "{:.3e}".format
    }
))

#  Kết luận
print("\nTRUNG VỊ (GB/s) để mô tả thêm:")
print(df.groupby(GROUP)[VALUE].median().loc[GROUPS].to_string())
