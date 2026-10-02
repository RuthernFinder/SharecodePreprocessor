import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("GPU_10_Features.csv")


# Hàm trích xuất giá trị số từ chuỗi (VD: "1024 MB" -> 1024.0, "2009" -> 2009.0)
def extract_numeric(series):
  return series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)


# 2. Tạo hàm vẽ Q-Q Plot kiểm tra Phân phối chuẩn (Chuẩn giao diện ggqqplot)
def plot_ggqqplot(data_series, ylabel_name, title_name=""):
  sample = data_series.dropna().values
  n = len(sample)
  if n == 0:
    return

  # Sắp xếp mẫu dữ liệu
  sorted_data = np.sort(sample)

  # Tính phân vị lý thuyết của Phân phối chuẩn chuẩn hóa N(0, 1)
  probs = (np.arange(1, n + 1) - 0.5) / n
  theoretical_quantiles = stats.norm.ppf(probs)

  # Đường tham chiếu đi qua Q1 (25%) và Q3 (75%) theo thuật toán qqline
  q25_x, q75_x = stats.norm.ppf([0.25, 0.75])
  q25_y, q75_y = np.percentile(sorted_data, [25, 75])

  slope = (q75_y - q25_y) / (q75_x - q25_x)
  intercept = q25_y - slope * q25_x
  fit_y = intercept + slope * theoretical_quantiles

  # Tính sai số chuẩn (Standard Error) & Giới hạn dải tin cậy 95%
  pdf = stats.norm.pdf(theoretical_quantiles)
  pdf = np.where(pdf < 1e-6, 1e-6, pdf)  # Tránh lỗi chia cho 0 ở hai đầu biên

  se = (slope / pdf) * np.sqrt(probs * (1 - probs) / n)
  z_crit = stats.norm.ppf(0.975)  # Mức tin cậy 95%

  lower_bound = fit_y - z_crit * se
  upper_bound = fit_y + z_crit * se

  # Khởi tạo khung biểu đồ
  plt.figure(figsize=(7, 6))

  # 1. Dải tin cậy 95% (Confidence Ribbon màu xám)
  plt.fill_between(
      theoretical_quantiles,
      lower_bound,
      upper_bound,
      color="lightgray",
      alpha=0.7,
      label="95% CI",
  )

  # 2. Đường tham chiếu phân phối chuẩn
  plt.plot(
      theoretical_quantiles, fit_y, color="gray", linestyle="-", linewidth=1.5
  )

  # 3. Các điểm dữ liệu mẫu thực tế
  plt.scatter(
      theoretical_quantiles, sorted_data, color="black", s=10, zorder=3
  )

  # Nhãn và giao diện phẳng tối giản (Minimal Theme)
  plt.xlabel("Theoretical", fontsize=11)
  plt.ylabel(ylabel_name, fontsize=11)
  if title_name:
    plt.title(title_name, fontsize=12)

  # Định dạng đường viền và lưới đồ thị
  ax = plt.gca()
  ax.set_facecolor("white")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("black")
  ax.spines["bottom"].set_color("black")
  plt.grid(True, linestyle=":", alpha=0.3)

  plt.tight_layout()
  plt.show()


# 3. Trích xuất đủ 6 biến định lượng từ file CSV
df_clean = pd.DataFrame({
    "Memory_Speed": extract_numeric(df["Memory_Speed"]),
    "Memory_Bandwidth": extract_numeric(df["Memory_Bandwidth"]),
    "Memory": extract_numeric(df["Memory"]),
    "Memory_Bus": extract_numeric(df["Memory_Bus"]),
    "Process": extract_numeric(df["Process"]),
    "Release_Date": (
        df["Release_Date"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    ),
})

# 4. Danh sách cấu hình vẽ cho ĐỦ 6 ĐẶC TRƯNG ĐỊNH LƯỢNG
qq_configs = [
    (
        df_clean["Memory_Speed"],
        "Memory Speed [MHz]",
        "Biểu đồ QQ-plot của Memory_speed",
    ),
    (
        df_clean["Memory_Bandwidth"],
        "Memory Bandwidth [GB/sec]",
        "Biểu đồ QQ-plot của Memory_bandwidth",
    ),
    (
        df_clean["Memory"],
        "Memory Value [MB]",
        "Biểu đồ QQ-plot của Memory_value",
    ),
    (
        df_clean["Memory_Bus"],
        "Memory Bus [Bit]",
        "Biểu đồ QQ-plot của Memory_bus",
    ),
    (
        df_clean["Process"],
        "Process [nm]",
        "Biểu đồ QQ-plot của Process",
    ),
    (
        df_clean["Release_Date"],
        "Release Date [Year]",
        "Biểu đồ QQ-plot của Release_date",
    ),
]

# Chạy vòng lặp vẽ lần lượt cả 6 biểu đồ
for data, ylabel, title in qq_configs:
  plot_ggqqplot(data, ylabel_name=ylabel, title_name=title)