import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
df = pd.read_csv("GPU_10_Features.csv")


# Hàm hỗ trợ trích xuất số từ chuỗi
def extract_numeric(series):
  return series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)


# Tiền xử lý dữ liệu cho 6 biến định lượng
df_clean = pd.DataFrame({
    "Memory_Speed_Value": extract_numeric(df["Memory_Speed"]),
    "Memory_Bandwidth_Value": extract_numeric(df["Memory_Bandwidth"]),
    "Memory_Value": extract_numeric(df["Memory"]),
    "Memory_Bus_Value": extract_numeric(df["Memory_Bus"]),
    "Process_Value": extract_numeric(df["Process"]),
    "Release_Date": (
        df["Release_Date"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    ),
})


# ==========================================
# 2. HÀM VẼ BIỂU ĐỒ TẦN SỐ CHO CÁC BIẾN ĐỊNH LƯỢNG
# ==========================================
def plot_hist_graph(data_series, name, num_bins=20):
  sample = data_series.dropna().values
  if len(sample) == 0:
    return

  # Thuật toán tính khoảng x_axis tương tự R (plotHist)
  min_val = math.floor(np.min(sample))
  max_val = np.max(sample) * 1.1
  step = math.floor((max_val - min_val) / num_bins)
  if step <= 0:
    step = 1

  breaks = np.arange(min_val, max_val + step, step)

  # Tạo cửa sổ biểu đồ
  plt.figure(figsize=(9, 5))
  counts, bins, _ = plt.hist(
      sample, bins=breaks, color="lightgreen", edgecolor="black"
  )

  # Hiển thị số tần số trên đỉnh mỗi cột
  for count, bin_left in zip(counts, bins):
    if count > 0:
      plt.text(
          bin_left + step / 2,
          count + (max(counts) * 0.015),
          str(int(count)),
          ha="center",
          va="bottom",
          fontsize=8,
      )

  plt.xlabel(name, fontsize=11)
  plt.ylabel("Tần số", fontsize=11)
  plt.xticks(breaks, rotation=45, fontsize=8)
  plt.title(f"Biểu đồ phân phối tần số của {name}", fontsize=12)
  plt.grid(axis="y", linestyle="--", alpha=0.5)
  plt.tight_layout()
  plt.show()


# ==========================================
# 3. HÀM VẼ BIỂU ĐỒ CHO CÁC BIẾN PHÂN LOẠI (NVIDIA/AMD, NOTEBOOK, DRAM)
# ==========================================
def plot_categorical_graphs(df):

  # A. Biểu đồ 1: Hãng sản xuất (Nvidia, AMD, Intel, ATI)
  fig, ax = plt.subplots(figsize=(8, 5))  # Chỉ tạo 1 đồ thị trên khung
  mfg_counts = df["Manufacturer"].value_counts()
  bars1 = ax.bar(
      mfg_counts.index,
      mfg_counts.values,
      color=["#76b900", "#ed1c24", "#0071c5", "#e06666"],
      edgecolor="black",
  )
  ax.set_title("Phân bố Hãng sản xuất (Manufacturer)", fontsize=12)
  ax.set_ylabel("Tần số (Số lượng GPU)", fontsize=11)
  for bar in bars1:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 15,
        str(int(yval)),
        ha="center",
        va="bottom",
        fontsize=9,
    )
  plt.tight_layout()
  plt.show()

  # B. Biểu đồ Notebook GPU (Desktop vs Laptop)
  fig, ax = plt.subplots(figsize=(7, 5))
  notebook_counts = df["Notebook_GPU"].value_counts()
  bars2 = ax.bar(
      notebook_counts.index,
      notebook_counts.values,
      color=["skyblue", "orange"],
      edgecolor="black",
  )
  ax.set_title("Phân bố GPU Desktop (No) vs Laptop (Yes)", fontsize=12)
  ax.set_ylabel("Tần số (Số lượng GPU)", fontsize=11)
  for bar in bars2:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 20,
        str(int(yval)),
        ha="center",
        va="bottom",
        fontsize=9,
    )
  plt.tight_layout()
  plt.show()

  # C. Biểu đồ 3: Top Loại bộ nhớ (DRAM / Memory Type)
  fig, ax = plt.subplots(figsize=(9, 5))
  dram_counts = df["Memory_Type"].value_counts().head(7)
  bars3 = ax.bar(
      dram_counts.index.astype(str),
      dram_counts.values,
      color="lightcoral",
      edgecolor="black",
  )
  ax.set_title("Phân bố Loại bộ nhớ (Top DRAM Types)", fontsize=12)
  ax.set_ylabel("Tần số (Số lượng GPU)", fontsize=11)
  ax.tick_params(axis="x", rotation=30)
  for bar in bars3:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 15,
        str(int(yval)),
        ha="center",
        va="bottom",
        fontsize=8,
    )
  plt.tight_layout()
  plt.show()


# ==========================================
# 4. CHẠY VẼ TẤT CẢ BIỂU ĐỒ
# ==========================================

# A. Vẽ các biểu đồ phân phối tần số định lượng (6 biểu đồ)
features = [
    ("Memory_Speed_Value", "Memory Speed [MHz]"),
    ("Memory_Bandwidth_Value", "Memory Bandwidth [GB/sec]"),
    ("Memory_Value", "Memory Value [MB]"),
    ("Memory_Bus_Value", "Memory Bus [Bit]"),
    ("Process_Value", "Process [nm]"),
    ("Release_Date", "Release Date [Year]"),
]

for col, label in features:
  plot_hist_graph(df_clean[col], name=label, num_bins=20)

# B. Vẽ cửa sổ tổng hợp cho các đặc trưng phân loại (Nvidia/AMD, Notebook, DRAM)
plot_categorical_graphs(df)
