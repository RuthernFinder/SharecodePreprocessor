"""
Kruskal-Wallis + Post-hoc Dunn (Bonferroni) cho Memory_Bandwidth_GBs
giua 5 chuan bo nho: DDR, DDR2, DDR3, GDDR3, GDDR5.

Cach chay:   python kruskal_dunn_gpu.py GPU_10_Features_Processed_Final.csv
Can cai:     pip install pandas numpy scipy

Moi buoc deu tinh TAY theo cong thuc (khop voi loi giai), sau do doi chieu
voi scipy.stats.kruskal de kiem tra.
"""
import sys
from itertools import combinations # tổ hợp chỉnh hợp

import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "../../GPU_10_Features_Processed_Final.csv"
GROUPS = ["DDR", "DDR2", "DDR3", "GDDR3", "GDDR5"]
VALUE, GROUP = "Memory_Bandwidth_GBs", "Memory_Type"
alpha = 0.05

# Loc 5 chuan, bo dong thieu bang thong
df = pd.read_csv(CSV)
df = df[df[GROUP].isin(GROUPS)]
print(f"So dong 5 chuan truoc khi loc thieu: {len(df)}")
print("Dong thieu bang thong theo nhom:",
      df[df[VALUE].isna()][GROUP].value_counts().to_dict())
df = df.dropna(subset=[VALUE]).copy()

N = len(df)
k = len(GROUPS)
n = df.groupby(GROUP)[VALUE].size().loc[GROUPS]
print(f"\nBUOC 1: N = {N}")
print(n.to_string())


# Xep hang chung, ties lay hang trung binh
df["rank"] = df[VALUE].rank(method="average")

# ---------------------------------------------------------------- Buoc 3
# R_i = tong hang, mean rank = R_i / n_i
R = df.groupby(GROUP)["rank"].sum().loc[GROUPS]
mean_rank = R / n
print("\nBUOC 3: tong hang va hang trung binh")
print(pd.DataFrame({"n": n, "R_i": R, "mean_rank": mean_rank}).round(2).to_string())
print(f"Kiem tra: sum(R_i) = {R.sum():,.1f}  vs  N(N+1)/2 = {N * (N + 1) / 2:,.1f}")
print(f"Hang trung binh ky vong duoi H0 = (N+1)/2 = {(N + 1) / 2}")

# ---------------------------------------------------------------- Buoc 4
# H = 12/(N(N+1)) * sum(R_i^2 / n_i) - 3(N+1)
sum_term = (R ** 2 / n).sum()
H_raw = 12 / (N * (N + 1)) * sum_term - 3 * (N + 1)
print(f"\nBUOC 4: sum(R_i^2/n_i) = {sum_term:,.2f}")
print(f"        H tho = 12/({N}*{N + 1}) * {sum_term:,.2f} - 3*{N + 1} = {H_raw:.3f}")

# ---------------------------------------------------------------- Buoc 5
# He so hieu chinh ties: C = 1 - sum(t^3 - t) / (N^3 - N)
t = df[VALUE].value_counts()
T = (t ** 3 - t).sum()
C = 1 - T / (N ** 3 - N)
H = H_raw / C
print(f"\nBUOC 5: sum(t^3 - t) = {T:,.0f},  N^3 - N = {N ** 3 - N:,}")
print(f"        C = {C:.8f}  ->  H hieu chinh = {H_raw:.3f} / {C:.8f} = {H:.3f}")

# ---------------------------------------------------------------- Buoc 6
dof = k - 1
crit = stats.chi2.ppf(1 - alpha, dof)
p_kw = stats.chi2.sf(H, dof)
eps2 = H * (N + 1) / (N ** 2 - 1)
print(f"\nBUOC 6: df = {dof},  chi2 toi han ({alpha}) = {crit:.3f}")
print(f"        p-value = {p_kw:.3e}   (0 nghia la nho hon muc may tinh bieu dien duoc)")
print(f"        epsilon^2 = {eps2:.3f}")
print("        =>", "BAC BO H0: co khac biet tong the" if p_kw < alpha
      else "KHONG bac bo H0: dung lai, khong can Dunn")

# Doi chieu voi scipy
H_sp, p_sp = stats.kruskal(*[df.loc[df[GROUP] == g, VALUE] for g in GROUPS])
print(f"        [doi chieu scipy] H = {H_sp:.3f}, p = {p_sp:.3e}")

# ---------------------------------------------------------------- Buoc 7-9
# Dunn: z = (Rbar_i - Rbar_j) / SE
# SE = sqrt( [N(N+1)/12 - sum(t^3-t)/(12(N-1))] * (1/n_i + 1/n_j) )
m = k * (k - 1) // 2
base = N * (N + 1) / 12 - T / (12 * (N - 1))
print(f"\nBUOC 7-9: Dunn, so cap m = {m}, phan chung = {base:,.3f}, "
      f"nguong Bonferroni = {alpha}/{m} = {alpha / m}")

rows = []
for a, b in combinations(GROUPS, 2):
    diff = mean_rank[a] - mean_rank[b]
    se = np.sqrt(base * (1 / n[a] + 1 / n[b]))
    z = diff / se
    p_raw = 2 * stats.norm.sf(abs(z))        # hai phia
    p_adj = min(1.0, p_raw * m)              # Bonferroni
    rows.append({"Cap": f"{a} - {b}", "Hieu hang TB": diff, "SE": se, "z": z,
                 "p tho": p_raw, "p Bonferroni": p_adj,
                 "Khac biet": "CO" if p_adj < alpha else "khong"})
res = pd.DataFrame(rows)

pd.set_option("display.width", 200)
print(res.to_string(index=False,
                    formatters={"Hieu hang TB": "{:.1f}".format, "SE": "{:.2f}".format,
                                "z": "{:.3f}".format, "p tho": "{:.3e}".format,
                                "p Bonferroni": "{:.3e}".format}))

# ---------------------------------------------------------------- Ket luan
print("\nTRUNG VI (GB/s) de mo ta them:")
print(df.groupby(GROUP)[VALUE].median().loc[GROUPS].to_string())

print("\nCac buoc nhay giua the he lien ke:")
for a, b in zip(GROUPS[:-1], GROUPS[1:]):
    r = res[res["Cap"] == f"{a} - {b}"].iloc[0]
    print(f"  {b:6s} vs {a:6s}: hieu hang = {abs(r['Hieu hang TB']):7.1f}, "
          f"p Bonferroni = {r['p Bonferroni']:.3e}  -> {r['Khac biet']}")

res.to_csv("dunn_bonferroni_results.csv", index=False)
print("\nDa luu bang ket qua: dunn_bonferroni_results.csv")
