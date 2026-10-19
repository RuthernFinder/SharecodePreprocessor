import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("GPU_10_Features_Processed_Final.csv")


# Xóa khoảng trắng thừa trong tên cột
"""df.columns = df.columns.str.strip()"""

# Hàm tự động tìm cột dựa trên danh sách từ khóa linh hoạt
def find_column(df, keywords):
  for col in df.columns:
    col_clean = col.lower().replace(" ", "").replace("_", "")
    for kw in keywords:
      if kw in col_clean:
        return df[col], col
  return None, None


# Hàm trích xuất giá trị số (loại bỏ đơn vị chữ nếu có)
def extract_numeric(series):
  if series is None:
    return None
  if pd.api.types.is_numeric_dtype(series):
    return series.astype(float)
  extracted = series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
  return pd.to_numeric(extracted, errors="coerce")


# Hàm trích xuất riêng NĂM (4 chữ số)
def extract_year(series):
  if series is None:
    return None
  date_parsed = pd.to_datetime(series, errors="coerce")
  years = date_parsed.dt.year
  if years.isna().all():
    years = pd.to_numeric(
        series.astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )
  return years


# Hàm trích xuất riêng THÁNG (1-12)
def extract_month(series):
  if series is None:
    return None
  date_parsed = pd.to_datetime(series, errors="coerce")
  months = date_parsed.dt.month
  if months.isna().all():
    extracted = series.astype(str).str.extract(r"(\b[1-9]\b|\b1[0-2]\b)")[0]
    months = pd.to_numeric(extracted, errors="coerce")
  return months


# 2. Định nghĩa từ khóa nhận diện cho từng đặc trưng (Hỗ trợ cả Anh lẫn Việt)
keyword_mapping = {
    "Memory_Speed": ["speed", "tocdo", "mhz", "clock"],
    "Memory_Bandwidth": ["bandwidth", "bangthong", "gb/s", "gb/sec"],
    "Memory": ["memory", "ram", "bonho", "value", "vram"],
    "Memory_Bus": ["bus", "bit"],
    "Process": ["process", "tientrinh", "nm"],
    "Release_Date": ["release", "date", "ngay", "nhan", "phat-hanh", "year"],
}

# 3. Trích xuất dữ liệu cho từng đặc trưng
df_clean = pd.DataFrame()

for feature, keywords in keyword_mapping.items():
  col_data, col_name = find_column(df, keywords)
  if col_data is not None:
    if feature == "Release_Date":
      df_clean["Release_Year"] = extract_year(col_data)
      df_clean["Release_Month"] = extract_month(col_data)
    else:
      df_clean[feature] = extract_numeric(col_data)

# 4. Hàm vẽ Q-Q Plot kiểm tra Phân phối chuẩn
def plot_ggqqplot(data_series, ylabel_name, title_name=""):
  if data_series is None:
    return

  sample = data_series.dropna().values
  n = len(sample)
  if n == 0:
    return

  # Sắp xếp mẫu dữ liệu
  sorted_data = np.sort(sample)

  # Tính phân vị lý thuyết N(0, 1)
  probs = (np.arange(1, n + 1) - 0.5) / n
  theoretical_quantiles = stats.norm.ppf(probs)

  # Đường tham chiếu qua Q1 (25%) và Q3 (75%)
  q25_x, q75_x = stats.norm.ppf([0.25, 0.75])
  q25_y, q75_y = np.percentile(sorted_data, [25, 75])

  slope = (q75_y - q25_y) / (q75_x - q25_x) if (q75_x - q25_x) != 0 else 1
  intercept = q25_y - slope * q25_x
  fit_y = intercept + slope * theoretical_quantiles

  # Dải tin cậy 95%
  pdf = stats.norm.pdf(theoretical_quantiles)
  pdf = np.where(pdf < 1e-6, 1e-6, pdf)
  se = (slope / pdf) * np.sqrt(probs * (1 - probs) / n)
  z_crit = stats.norm.ppf(0.975)

  lower_bound = fit_y - z_crit * se
  upper_bound = fit_y + z_crit * se

  # Khởi tạo khung đồ thị
  plt.figure(figsize=(7, 6))

  # Dải tin cậy 95% (Màu xám)
  plt.fill_between(
      theoretical_quantiles,
      lower_bound,
      upper_bound,
      color="lightgray",
      alpha=0.7,
      label="95% CI",
  )

  # Đường tham chiếu & điểm dữ liệu
  plt.plot(
      theoretical_quantiles, fit_y, color="gray", linestyle="-", linewidth=1.5
  )
  plt.scatter(
      theoretical_quantiles, sorted_data, color="black", s=10, zorder=3
  )

  plt.xlabel("Theoretical", fontsize=11)
  plt.ylabel(ylabel_name, fontsize=11)
  if title_name:
    plt.title(title_name, fontsize=12)

  ax = plt.gca()
  ax.set_facecolor("white")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("black")
  ax.spines["bottom"].set_color("black")
  plt.grid(True, linestyle=":", alpha=0.3)

  plt.tight_layout()
  plt.show()


# 5. Cấu hình và thực hiện vẽ 7 biểu đồ
qq_configs = [
    (
        df_clean.get("Memory_Speed"),
        "Memory Speed [MHz]",
        "Biểu đồ QQ-plot của Memory_speed",
    ),
    (
        df_clean.get("Memory_Bandwidth"),
        "Memory Bandwidth [GB/sec]",
        "Biểu đồ QQ-plot của Memory_bandwidth",
    ),
    (
        df_clean.get("Memory"),
        "Memory Value [MB]",
        "Biểu đồ QQ-plot của Memory_value",
    ),
    (
        df_clean.get("Memory_Bus"),
        "Memory Bus [Bit]",
        "Biểu đồ QQ-plot của Memory_bus",
    ),
    (
        df_clean.get("Process"),
        "Process [nm]",
        "Biểu đồ QQ-plot của Process",
    ),
    (
        df_clean.get("Release_Year"),
        "Release Year [Year]",
        "Biểu đồ QQ-plot của Release_year",
    ),
    (
        df_clean.get("Release_Month"),
        "Release Month [1-12]",
        "Biểu đồ QQ-plot của Release_month",
    ),
]

for data, ylabel, title in qq_configs:
  if data is not None:
    plot_ggqqplot(data, ylabel_name=ylabel, title_name=title)
