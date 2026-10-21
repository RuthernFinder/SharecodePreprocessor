import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("GPU_10_Features_Processed_Final.csv")

# A. Biểu đồ memory speed
def plot_memory_speed_graph(df):
    """Vẽ biểu đồ cột phân bố Memory Speed theo các dải xung nhịp chuẩn."""
    speed_series = df["Memory_Speed_MHz"].dropna()

    def categorize_speed(speed):
        if speed <= 500:
            return "≤ 500 MHz"
        elif speed <= 800:
            return "501 - 800 MHz"
        elif speed <= 1200:
            return "801 - 1200 MHz"
        elif speed <= 1600:
            return "1201 - 1600 MHz"
        elif speed <= 2000:
            return "1601 - 2000 MHz"
        else:
            return "> 2000 MHz"

    grouped_speed = speed_series.apply(categorize_speed)
    order = [
        "≤ 500 MHz",
        "501 - 800 MHz",
        "801 - 1200 MHz",
        "1201 - 1600 MHz",
        "1601 - 2000 MHz",
        "> 2000 MHz",
    ]
    counts = grouped_speed.value_counts().reindex(order).fillna(0)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        counts.index, counts.values, color="mediumseagreen", edgecolor="black"
    )

    plt.xlabel("Memory Speed [MHz]", fontsize=11)
    plt.ylabel("Tần số (Số lượng GPU)", fontsize=11)
    plt.title("Biểu đồ phân phối tần số của Memory Speed [MHz]", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    max_val = max(counts.values)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + (max_val * 0.015),
                f"{int(yval):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()
    
# B. Biểu đồ memory bandwidth
def plot_bandwidth_graph(df):
    bw_series = df["Memory_Bandwidth_GBs"].dropna()

    def categorize_bw(bw):
        if bw <= 32:
            return "≤ 32 GB/s"
        elif bw <= 64:
            return "32 - 64"
        elif bw <= 128:
            return "64 - 128"
        elif bw <= 256:
            return "128 - 256"
        elif bw <= 512:
            return "256 - 512"
        else:
            return "> 512 GB/s"

    grouped_bw = bw_series.apply(categorize_bw)
    order = [
        "≤ 32 GB/s",
        "32 - 64",
        "64 - 128",
        "128 - 256",
        "256 - 512",
        "> 512 GB/s",
    ]
    counts = grouped_bw.value_counts().reindex(order).fillna(0)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(counts.index, counts.values, color="skyblue", edgecolor="black")

    plt.xlabel("Memory Bandwidth [GB/s]", fontsize=11)
    plt.ylabel("Tần số (Số lượng GPU)", fontsize=11)
    plt.title("Biểu đồ phân phối tần số của Memory Bandwidth [GB/s]", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    max_val = max(counts.values)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + (max_val * 0.015),
                f"{int(yval):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()

# C. Biểu đồ memory mb
def plot_memory_mb_graph(df):
  """Vẽ biểu đồ cột phân bố Dung lượng bộ nhớ Memory MB theo chuẩn dung lượng RAM."""
  mem_series = df["Memory_MB"].dropna().astype(int)

  def categorize_mem(mem):
    if mem <= 512:
      return "≤ 512 MB"
    elif mem == 1024:
      return "1 GB (1024MB)"
    elif mem == 2048:
      return "2 GB (2048MB)"
    elif mem == 3072 or mem == 4096:
      return "3-4 GB"
    elif 4096 < mem <= 8192:
      return "5-8 GB"
    else:
      return "> 8 GB"

  grouped_mem = mem_series.apply(categorize_mem)
  order = [
      "≤ 512 MB",
      "1 GB (1024MB)",
      "2 GB (2048MB)",
      "3-4 GB",
      "5-8 GB",
      "> 8 GB",
  ]
  counts = grouped_mem.value_counts().reindex(order).fillna(0)

  plt.figure(figsize=(10, 5))
  bars = plt.bar(
      counts.index, counts.values, color="lightcoral", edgecolor="black"
  )

  plt.xlabel("Memory Capacity [MB / GB]", fontsize=11)
  plt.ylabel("Tần số (Số lượng GPU)", fontsize=11)
  plt.title("Biểu đồ phân phối tần số của Memory Value [MB]", fontsize=12)
  plt.grid(axis="y", linestyle="--", alpha=0.5)

  max_val = max(counts.values)
  for bar in bars:
    yval = bar.get_height()
    if yval > 0:
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          yval + (max_val * 0.015),
          f"{int(yval):,}",
          ha="center",
          va="bottom",
          fontsize=8,
      )

  plt.tight_layout()
  plt.show()

# D. Biểu đồ process
def plot_process_nm_graph(df):
  """Vẽ biểu đồ cột phân bố Tiến trình sản xuất Process (nm) theo từng mốc công nghệ."""
  proc_series = df["Process_nm"].dropna().astype(int)

  def categorize_proc(proc):
    if proc <= 28:
      return "≤ 28 nm"
    elif proc <= 40:
      return "30 - 40 nm"
    elif proc <= 65:
      return "55 - 65 nm"
    elif proc <= 90:
      return "80 - 90 nm"
    else:
      return "> 90 nm"

  grouped_proc = proc_series.apply(categorize_proc)
  order = ["≤ 28 nm", "30 - 40 nm", "55 - 65 nm", "80 - 90 nm", "> 90 nm"]
  counts = grouped_proc.value_counts().reindex(order).fillna(0)

  plt.figure(figsize=(10, 5))
  bars = plt.bar(
      counts.index, counts.values, color="khaki", edgecolor="black"
  )

  plt.xlabel("Process Technology [nm]", fontsize=11)
  plt.ylabel("Tần số (Số lượng GPU)", fontsize=11)
  plt.title("Biểu đồ phân phối tần số của Process [nm]", fontsize=12)
  plt.grid(axis="y", linestyle="--", alpha=0.5)

  max_val = max(counts.values)
  for bar in bars:
    yval = bar.get_height()
    if yval > 0:
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          yval + (max_val * 0.015),
          f"{int(yval):,}",
          ha="center",
          va="bottom",
          fontsize=8,
      )

  plt.tight_layout()
  plt.show()

# E. Biểu đồ memory bus
def plot_memory_bus_graph(df):
  bus_series = df["Memory_Bus_Bit"].dropna().astype(int)

  def categorize_bus(bus):
    if bus <= 64:
      return "≤ 64 Bit"
    elif bus == 128:
      return "128 Bit"
    elif bus == 192:
      return "192 Bit"
    elif bus == 256:
      return "256 Bit"
    elif bus == 384:
      return "384 Bit"
    elif bus == 512:
      return "512 Bit"
    elif 64 < bus < 512:
      return "Khác (64-512)"
    else:
      return "> 512 Bit"

  grouped_bus = bus_series.apply(categorize_bus)
  order = [
      "≤ 64 Bit",
      "128 Bit",
      "192 Bit",
      "256 Bit",
      "384 Bit",
      "512 Bit",
      "Khác (64-512)",
      "> 512 Bit",
  ]
  bus_counts = grouped_bus.value_counts().reindex(order).fillna(0)
  bus_counts = bus_counts[bus_counts > 0]

  plt.figure(figsize=(10, 5))
  bars = plt.bar(
      bus_counts.index, bus_counts.values, color="lightgreen", edgecolor="black"
  )

  plt.xlabel("Memory Bus [Bit]", fontsize=11)
  plt.ylabel("Tần số (Số lượng GPU)", fontsize=11)
  plt.title("Biểu đồ phân phối tần số của Memory Bus [Bit]", fontsize=12)
  plt.xticks(rotation=25, fontsize=9)
  plt.grid(axis="y", linestyle="--", alpha=0.5)

  max_val = max(bus_counts.values)
  for bar in bars:
    yval = bar.get_height()
    if yval > 0:
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          yval + (max_val * 0.015),
          f"{int(yval):,}",
          ha="center",
          va="bottom",
          fontsize=8,
      )

  plt.tight_layout()
  plt.show()
    

# F. Biểu đồ các biến phân loại
def plot_categorical_graphs(df):
  # a. Hãng sản xuất (Manufacturer)
  fig, ax = plt.subplots(figsize=(8, 5))
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

  # b. GPU Desktop vs Laptop (Is_Notebook)
  fig, ax = plt.subplots(figsize=(7, 5))
  notebook_map = {0: "No (Desktop)", 1: "Yes (Laptop)"}
  notebook_counts = df["Is_Notebook"].map(notebook_map).value_counts()
  bars2 = ax.bar(
      notebook_counts.index,
      notebook_counts.values,
      color=["skyblue", "orange"],
      edgecolor="black",
  )
  ax.set_title("Phân bố GPU Desktop vs Laptop", fontsize=12)
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

  # c. Top Loại bộ nhớ (Memory Type)
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

  # d. Phân bố GPU ra mắt theo Tháng (Release Month)
  fig, ax = plt.subplots(figsize=(10, 5))
  month_counts = (
      df["Release_Month"].dropna().astype(int).value_counts().sort_index()
  )
  month_labels = [f"Tháng {m}" for m in month_counts.index]

  bars4 = ax.bar(
      month_labels, month_counts.values, color="#4e79a7", edgecolor="black"
  )
  ax.set_title("Phân bố số lượng GPU ra mắt theo Tháng (1 - 12)", fontsize=12)
  ax.set_ylabel("Tần số (Số lượng GPU)", fontsize=11)
  ax.tick_params(axis="x", rotation=30)
  ax.grid(axis="y", linestyle="--", alpha=0.5)

  for bar in bars4:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 8,
        str(int(yval)),
        ha="center",
        va="bottom",
        fontsize=8,
    )
  plt.tight_layout()
  plt.show()

# e. Biểu đồ Phân bố GPU theo Năm phát hành (tách biệt do có một số năm có số lượng GPU ra mắt rất ít, nên cần tách riêng để biểu diễn rõ ràng hơn)
def plot_release_year_graph(df):
  year_counts = (
      df["Release_Year"].dropna().astype(int).value_counts().sort_index()
  )

  plt.figure(figsize=(10, 5))
  bars = plt.bar(
      year_counts.index.astype(str),
      year_counts.values,
      color="lightgreen",
      edgecolor="black",
  )

  plt.xlabel("Release Date [Year]", fontsize=11)
  plt.ylabel("Tần số (Số lượng GPU)", fontsize=11)
  plt.title("Biểu đồ phân phối tần số của Release Date [Year]", fontsize=12)
  plt.xticks(rotation=45, fontsize=8)
  plt.grid(axis="y", linestyle="--", alpha=0.5)

  for bar in bars:
    yval = bar.get_height()
    if yval > 0:
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          yval + 10,
          str(int(yval)),
          ha="center",
          va="bottom",
          fontsize=8,
      )

  plt.tight_layout()
  plt.show()

# F. Vẽ Các Biểu Đồ
continuous_features = [
    ("Memory_Speed_MHz", "Memory Speed [MHz]"),
]
plot_categorical_graphs(df) #Cho các giá trị phân loại

if "Release_Year" in df.columns:
  plot_release_year_graph(df)

if "Process_nm" in df.columns:
  plot_process_nm_graph(df)

if "Memory_MB" in df.columns:
  plot_memory_mb_graph(df)

if "Memory_Bus_Bit" in df.columns:
  plot_memory_bus_graph(df)

if "Memory_Bandwidth_GBs" in df.columns:
  plot_bandwidth_graph(df)

if "Memory_Speed_MHz" in df.columns:
  plot_memory_speed_graph(df)








  
