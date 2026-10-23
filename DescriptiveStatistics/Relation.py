import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("GPU_10_Features_Processed_Final.csv")

num_cols = [
    "Release_Year",
    "Process_nm",
    "Memory_MB",
    "Memory_Bus_Bit",
    "Memory_Bandwidth_GBs",
    "Memory_Speed_MHz",
]

corr = df[num_cols].corr()

plt.figure(figsize=(7, 6.5))
ax = plt.gca()
ax.grid(False)

cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(cax, fraction=0.046, pad=0.04)

ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))

ax.set_xticklabels(
    num_cols, rotation=35, ha="left", rotation_mode="anchor", fontsize=9
)
ax.set_yticklabels(num_cols, fontsize=9)

plt.title(
    "Ma trận tương quan giữa các đặc trưng",
    fontsize=12,
    weight="bold",
    x=0.45,
    ha="center",
    pad=25
)

# Đặt chữ động theo độ sáng
for i in range(len(num_cols)):
  for j in range(len(num_cols)):
    val = corr.iloc[i, j]
    text_color = "white" if abs(val) > 0.55 else "black"
    ax.text(
        j,
        i,
        f"{val:.2f}",
        ha="center",
        va="center",
        color=text_color,
        fontsize=9,
        weight="bold" if abs(val) > 0.7 else "normal",
    )

plt.tight_layout()
plt.show()
# BIỂU ĐỒ 2: Scatter plot (Release Year vs Process nm)
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Release_Year"],
    df["Process_nm"],
    color="#2ca02c",
    alpha=0.6,
    edgecolors="none",
    s=40,
)
plt.title(
    "2. Tiến trình (Process nm) theo Năm phát hành",
    fontsize=12,
    weight="bold",
    pad=15,
)
plt.xlabel("Release Year", fontsize=10)
plt.ylabel("Process (nm)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# BIỂU ĐỒ 3: Scatter plot (Memory MB vs Bandwidth)
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Memory_MB"],
    df["Memory_Bandwidth_GBs"],
    color="#1f77b4",
    alpha=0.6,
    edgecolors="none",
    s=40,
)
plt.title(
    "3. Mối quan hệ giữa Dung lượng VRAM và Băng thông",
    fontsize=12,
    weight="bold",
    pad=15,
)
plt.xlabel("Memory Capacity (MB)", fontsize=10)
plt.ylabel("Bandwidth (GB/s)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# BIỂU ĐỒ 4: Boxplot so sánh Băng thông theo Hãng

plt.figure(figsize=(8, 5))
mfg_names = df["Manufacturer"].dropna().unique()
mfg_data = [
    df[df["Manufacturer"] == mfg]["Memory_Bandwidth_GBs"].dropna().values
    for mfg in mfg_names
]

bp = plt.boxplot(mfg_data, tick_labels=mfg_names, patch_artist=True)

# Tô màu hộp
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
for patch, color in zip(bp["boxes"], colors[: len(mfg_names)]):
  patch.set_facecolor(color)
  patch.set_alpha(0.7)

plt.title(
    "4. Băng thông bộ nhớ theo Hãng sản xuất",
    fontsize=12,
    weight="bold",
    pad=15,
)
plt.xlabel("Manufacturer", fontsize=10)
plt.ylabel("Bandwidth (GB/s)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
