"""
Ve bieu do cho bai Kruskal-Wallis + Dunn (Bonferroni) - Memory_Bandwidth_GBs
giua 5 chuan bo nho: DDR, DDR2, DDR3, GDDR3, GDDR5.

Cach chay:   python ve_bieu_do_kruskal_dunn.py GPU_10_Features_Processed_Final.csv [thu_muc_xuat]
Can cai:     pip install pandas numpy scipy matplotlib

Xuat ra (PNG 300 dpi, san sang chen vao bao cao):
  hinh0_tong_quan.png          3 o canh nhau: so phan tu, tong hang, hang trung binh
  hinh1_hang_trung_binh.png    HINH CHINH: hang trung binh + duong ky vong duoi H0
  hinh2_boxplot_bang_thong.png du lieu goc (thang log) de nguoi doc thay hinh dang
  hinh3_dunn_bonferroni.png    luoi 10 cap: p Bonferroni + hieu hang trung binh
"""
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "../../GPU_10_Features_Processed_Final.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

GROUPS = ["DDR", "DDR2", "DDR3", "GDDR3", "GDDR5"]
VALUE, GROUP = "Memory_Bandwidth_GBs", "Memory_Type"
ALPHA = 0.05

# ----------------------------------------------------------------- mau & style
BLUE = "#2a78d6"          # 1 mau cho 1 dai luong (do lon) - khong can legend
BLUE_LIGHT = "#bcd6f5"
INK = "#0b0b0b"           # chu chinh
INK2 = "#52514e"          # chu phu
GRID = "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def vn(x, d=0):
    """Dinh dang so kieu Viet Nam: 1.234,5"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "#").replace(".", ",").replace("#", ".")


def style_axes(ax, ygrid=True):
    ax.spines["left"].set_color(GRID if ygrid else INK2)
    ax.spines["bottom"].set_color(INK2)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ----------------------------------------------------------------- tinh toan
df = pd.read_csv(CSV)
df = df[df[GROUP].isin(GROUPS)].dropna(subset=[VALUE]).copy()
N = len(df)
df["rank"] = df[VALUE].rank(method="average")
n = df.groupby(GROUP).size().loc[GROUPS]
R = df.groupby(GROUP)["rank"].sum().loc[GROUPS]
mr = R / n
expected = (N + 1) / 2

H_raw = 12 / (N * (N + 1)) * (R ** 2 / n).sum() - 3 * (N + 1)
t = df[VALUE].value_counts()
T = (t ** 3 - t).sum()
H = H_raw / (1 - T / (N ** 3 - N))
k = len(GROUPS)
eps2 = H / (N - 1)

base = N * (N + 1) / 12 - T / (12 * (N - 1))
m = k * (k - 1) // 2
dunn = {}
for a, b in combinations(GROUPS, 2):
    diff = mr[b] - mr[a]                                  # nhom sau - nhom truoc (duong)
    z = diff / np.sqrt(base * (1 / n[a] + 1 / n[b]))
    p = 2 * stats.norm.sf(abs(z))
    dunn[(a, b)] = (diff, z, min(1.0, p * m))

xt = [f"{g}\n(n = {vn(n[g])})" for g in GROUPS]
x = np.arange(k)


def bar_panel(ax, values, title, ylabel, fmt, headroom=1.15):
    ax.bar(x, values, width=0.58, color=BLUE, zorder=3)
    for xi, v in zip(x, values):
        ax.text(xi, v + max(values) * 0.015, fmt(v), ha="center", va="bottom",
                fontsize=10, color=INK)
    ax.set_xticks(x, GROUPS)
    ax.set_ylim(0, max(values) * headroom)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: vn(v)))
    style_axes(ax)


# ----------------------------------------------------------------- Hinh 0
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
bar_panel(axes[0], n.values, "Số phần tử mỗi nhóm (nᵢ)", "Số card đồ họa", vn)
bar_panel(axes[1], R.values, "Tổng hạng mỗi nhóm (Rᵢ)", "Tổng hạng", vn)
bar_panel(axes[2], mr.values, "Hạng trung bình (R̄ᵢ = Rᵢ / nᵢ)", "Hạng trung bình",
          lambda v: vn(v, 1))
axes[2].axhline(expected, color=INK2, linestyle=(0, (5, 4)), linewidth=1.2, zorder=4)
axes[2].text(-0.45, expected + 35, f"Kỳ vọng dưới H₀ = {vn(expected)}",
             color=INK2, fontsize=9.5, ha="left", va="bottom")
fig.suptitle(f"Số phần tử, tổng hạng và hạng trung bình theo chuẩn bộ nhớ (N = {vn(N)})",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.text(0.01, 0.005,
         "Tổng hạng bị ảnh hưởng mạnh bởi cỡ mẫu; hạng trung bình mới là thước đo công bằng để so sánh các nhóm.",
         fontsize=9.5, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(f"{OUT}/hinh0_tong_quan.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------- Hinh 1 (chinh)
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.bar(x, mr.values, width=0.58, color=BLUE, zorder=3)
for xi, g in zip(x, GROUPS):
    ax.text(xi, mr[g] + 35, vn(mr[g], 1), ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=INK)
ax.axhline(expected, color=INK2, linestyle=(0, (5, 4)), linewidth=1.3, zorder=4)
ax.text(-0.45, expected + 40, f"Hạng trung bình kỳ vọng nếu H₀ đúng = (N+1)/2 = {vn(expected)}",
        color=INK2, fontsize=10, ha="left", va="bottom")
ax.set_xticks(x, xt)
ax.set_ylim(0, mr.max() * 1.12)
ax.set_ylabel("Hạng trung bình R̄ᵢ")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: vn(v)))
style_axes(ax)
ax.set_title(f"Hạng trung bình băng thông tăng dần qua các chuẩn bộ nhớ\n"
             f"Kruskal-Wallis: H({k - 1}) = {vn(H, 2)}, p < 0,001, ε² = {vn(eps2, 2)}",
             loc="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/hinh1_hang_trung_binh.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------- Hinh 2 (boxplot)
fig, ax = plt.subplots(figsize=(9, 5.6))
data = [df.loc[df[GROUP] == g, VALUE].values for g in GROUPS]
bp = ax.boxplot(data, positions=x, widths=0.5, patch_artist=True, showfliers=True,
                medianprops=dict(color=INK, linewidth=2),
                whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                boxprops=dict(edgecolor=BLUE, linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                                markeredgecolor=INK2, alpha=0.5))
for patch in bp["boxes"]:
    patch.set_facecolor(BLUE_LIGHT)
ax.set_yscale("log")
for xi, g in zip(x, GROUPS):
    med = np.median(df.loc[df[GROUP] == g, VALUE])
    ax.text(xi + 0.32, med, vn(med, 1), va="center", ha="left", fontsize=10, color=INK)
ax.set_xticks(x, xt)
ax.set_ylabel("Băng thông bộ nhớ (GB/s, thang log)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: vn(v)))
style_axes(ax)
ax.set_title("Phân phối băng thông theo chuẩn bộ nhớ (vạch đen = trung vị)",
             loc="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/hinh2_boxplot_bang_thong.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------- Hinh 3 (luoi Dunn)
fig, ax = plt.subplots(figsize=(9, 5.8))
CAP = 10.0   # -log10(p) vuot 10 thi cung dam nhat
rgb_hi = np.array(matplotlib.colors.to_rgb(BLUE))
for i in range(1, k):            # hang = nhom sau
    for j in range(i):           # cot  = nhom truoc
        diff, z, padj = dunn[(GROUPS[j], GROUPS[i])]
        sig = padj < ALPHA
        if sig:
            s = min(-np.log10(max(padj, 1e-300)), CAP) / CAP
            s = 0.18 + 0.82 * s
            face = (1 - s) * np.ones(3) + s * rgb_hi
            txt = "white" if s > 0.55 else INK
        else:
            face, txt = np.array([0.93, 0.93, 0.92]), INK2
        ax.add_patch(Rectangle((j, i - 1), 1, 1, facecolor=face, edgecolor="white", linewidth=3))
        ptxt = "p < 0,001" if padj < 0.001 else f"p = {vn(padj, 3)}"
        ax.text(j + 0.5, i - 1 + 0.38, ptxt, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=txt)
        ax.text(j + 0.5, i - 1 + 0.66, f"Δ hạng = {vn(diff)}", ha="center", va="center",
                fontsize=9.5, color=txt)
        if not sig:
            ax.text(j + 0.5, i - 1 + 0.88, "không khác biệt", ha="center", va="center",
                    fontsize=8.5, color=txt)
ax.set_xlim(0, k - 1)
ax.set_ylim(k - 1, 0)
ax.set_xticks(np.arange(k - 1) + 0.5, GROUPS[:-1])
ax.set_yticks(np.arange(k - 1) + 0.5, GROUPS[1:])
ax.xaxis.tick_top()
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0, labelsize=11)
fig.suptitle("Dunn post-hoc, hiệu chỉnh Bonferroni (10 cặp): 9/10 cặp khác biệt",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.01,
         f"Màu đậm = p điều chỉnh càng nhỏ. Δ hạng = hiệu hạng trung bình (hàng − cột). "
         f"Ngưỡng ý nghĩa: p điều chỉnh < {vn(ALPHA, 2)}.",
         fontsize=9.5, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(f"{OUT}/hinh3_dunn_bonferroni.png", dpi=300)
plt.close(fig)

print("Da luu 4 hinh vao thu muc:", os.path.abspath(OUT))
