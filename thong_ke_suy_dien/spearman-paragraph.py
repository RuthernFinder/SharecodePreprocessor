import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("default")
plt.rcParams.update(plt.rcParamsDefault)

df = pd.read_csv("GPU_10_Features_Processed_Final (2).csv")
clean_df = df.dropna(subset=["Process_nm", "Memory_Bandwidth_GBs"]).copy()

desktop_df = clean_df[clean_df["Is_Notebook"] == 0]
laptop_df = clean_df[clean_df["Is_Notebook"] == 1]

# BIỂU ĐỒ 1: Mục 2.1 - Tương quan Phi tuyến Tổng thể
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

ax.scatter(
    clean_df["Process_nm"],
    clean_df["Memory_Bandwidth_GBs"],
    color="#2b5c8f",
    alpha=0.4,
    s=25,
    edgecolor="none",
    label="Mẫu dữ liệu GPU (N = 2904)",
)

poly_ov = np.polyfit(
    clean_df["Process_nm"], clean_df["Memory_Bandwidth_GBs"], 2
)
line_ov = np.poly1d(poly_ov)
x_ov = np.linspace(
    clean_df["Process_nm"].min(), clean_df["Process_nm"].max(), 100
)
ax.plot(
    x_ov,
    line_ov(x_ov),
    color="#2ca02c",
    linewidth=2.5,
    linestyle="-",
    label=r"Xu hướng Phi tuyến ($\rho_s = -0.4005$)",
)

ax.set_title(
    "Mục 2.1: Tương quan Phi tuyến giữa Process và Memory Bandwidth",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel("Tiến trình chế tạo chip - Process (nm)", fontsize=9.5)
ax.set_ylabel("Băng thông bộ nhớ - Memory Bandwidth (GB/s)", fontsize=9.5)
ax.legend(loc="upper right", frameon=True, fontsize=8.5)
plt.tight_layout()
plt.show()

# BIỂU ĐỒ 2: Mục 2.2.1a - Tương quan Phi tuyến Desktop
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

ax.scatter(
    desktop_df["Process_nm"],
    desktop_df["Memory_Bandwidth_GBs"],
    color="#1f77b4",
    alpha=0.4,
    s=25,
    edgecolor="none",
    label=r"Dữ liệu Desktop ($n_1 = 2411$)",
)

# Đường xu hướng PHI TUYẾN Desktop (Bậc 2)
poly_dt = np.polyfit(
    desktop_df["Process_nm"], desktop_df["Memory_Bandwidth_GBs"], 2
)
line_dt = np.poly1d(poly_dt)
x_dt = np.linspace(
    desktop_df["Process_nm"].min(), desktop_df["Process_nm"].max(), 100
)
ax.plot(
    x_dt,
    line_dt(x_dt),
    color="#0d3b66",
    linewidth=2.5,
    linestyle="-",
    label=r"Xu hướng Phi tuyến Desktop ($\rho_1 = -0.4307$)",
)

ax.set_title(
    "Mục 2.2.1a: Tương quan Phi tuyến ở hệ Desktop (Is_Notebook = 0)",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel("Tiến trình chế tạo chip - Process (nm)", fontsize=9.5)
ax.set_ylabel("Băng thông bộ nhớ - Memory Bandwidth (GB/s)", fontsize=9.5)
ax.legend(loc="upper right", frameon=True, fontsize=8.5)
plt.tight_layout()
plt.show()

# BIỂU ĐỒ 3: Mục 2.2.1b - Tương quan Phi tuyến Laptop
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

ax.scatter(
    laptop_df["Process_nm"],
    laptop_df["Memory_Bandwidth_GBs"],
    color="#ff7f0e",
    alpha=0.5,
    s=25,
    edgecolor="none",
    label=r"Dữ liệu Laptop ($n_2 = 493$)",
)

# Đường xu hướng PHI TUYẾN Laptop (Bậc 2)
poly_lp = np.polyfit(
    laptop_df["Process_nm"], laptop_df["Memory_Bandwidth_GBs"], 2
)
line_lp = np.poly1d(poly_lp)
x_lp = np.linspace(
    laptop_df["Process_nm"].min(), laptop_df["Process_nm"].max(), 100
)
ax.plot(
    x_lp,
    line_lp(x_lp),
    color="#d95f02",
    linewidth=2.5,
    linestyle="-",
    label=r"Xu hướng Phi tuyến Laptop ($\rho_2 = -0.1916$)",
)

ax.set_title(
    "Mục 2.2.1b: Tương quan Phi tuyến ở hệ Laptop (Is_Notebook = 1)",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel("Tiến trình chế tạo chip - Process (nm)", fontsize=9.5)
ax.set_ylabel("Băng thông bộ nhớ - Memory Bandwidth (GB/s)", fontsize=9.5)
ax.legend(loc="upper right", frameon=True, fontsize=8.5)
plt.tight_layout()
plt.show()

# BIỂU ĐỒ 4: Mục 2.2.2 - So sánh 2 Đường cong Phi tuyến
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

ax.scatter(
    desktop_df["Process_nm"],
    desktop_df["Memory_Bandwidth_GBs"],
    color="#1f77b4",
    alpha=0.35,
    s=20,
    label="Desktop Points",
)
ax.scatter(
    laptop_df["Process_nm"],
    laptop_df["Memory_Bandwidth_GBs"],
    color="#ff7f0e",
    alpha=0.45,
    s=20,
    label="Laptop Points",
)

# Vẽ 2 đường cong so sánh
ax.plot(
    x_dt,
    line_dt(x_dt),
    color="#0d3b66",
    linewidth=2.5,
    label=r"Đường cong Desktop ($\rho_1 = -0.4307$)",
)
ax.plot(
    x_lp,
    line_lp(x_lp),
    color="#d95f02",
    linewidth=2.5,
    label=r"Đường cong Laptop ($\rho_2 = -0.1916$)",
)

ax.set_title(
    "Mục 2.2.2: So sánh đường cong xu hướng giữa Desktop và Laptop",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel("Tiến trình chế tạo chip - Process (nm)", fontsize=9.5)
ax.set_ylabel("Băng thông bộ nhớ - Memory Bandwidth (GB/s)", fontsize=9.5)
ax.legend(loc="upper right", frameon=True, fontsize=8.5)
plt.tight_layout()
plt.show()