import pandas as pd
import numpy as np
import re

# Đọc dữ liệu từ file data.
df = pd.read_csv("GPU_10_Features.csv")

# Viết hàm làm sạch dữ liệu.
def parse_num(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(val))
    return float(match.group(1)) if match else np.nan


def parse_year(dt):
    if pd.isna(dt):
        return np.nan
    match = re.search(r"(\d{4})", str(dt))
    return float(match.group(1)) if match else np.nan

# Tiếp nhận số liệu đầu vào - lược bỏ đơn vị đo lường
df["Memory_Value"] = df["Memory"].apply(parse_num) if "Memory" in df else np.nan
df["Memory_Bandwidth_Value"] = (
    df["Memory_Bandwidth"].apply(parse_num) if "Memory_Bandwidth" in df else np.nan
)
df["Memory_Speed_Value"] = (
    df["Memory_Speed"].apply(parse_num) if "Memory_Speed" in df else np.nan
)
df["Process_Value"] = (
    df["Process"].apply(parse_num) if "Process" in df else np.nan
)
df["Memory_Bus_Value"] = (
    df["Memory_Bus"].apply(parse_num) if "Memory_Bus" in df else np.nan
)
df["Is_Notebook"] = df["Notebook_GPU"].apply(
    lambda x: 1 if str(x).strip().lower() == "yes" else 0
)
df["Release_Year_Value"] = (
    df["Release_Date"].apply(parse_year) if "Release_Date" in df else np.nan
)


# Danh sách các cột định lượng cần thống kê mô tả
cols = [
    "Memory_Value",
    "Memory_Bandwidth_Value",
    "Memory_Speed_Value",
    "Process_Value",
    "Memory_Bus_Value",
    "Is_Notebook",
    "Release_Year_Value",
]


# Tính toán các chỉ số thống kê mô tả
summary_df = pd.DataFrame(
    {
        "mean": df[cols].mean(),
        "s2": df[cols].var(
            ddof=1
        ),  # Phương sai mẫu (ddof=1 tương đương hàm var())
        "q1": df[cols].quantile(0.25),  # Tứ phân vị thứ nhất (Q1)
        "q2": df[cols].quantile(0.50),  # Tứ phân vị thứ hai (Q2 / Median)
        "q3": df[cols].quantile(0.75),  # Tứ phân vị thứ ba (Q3)
        "min": df[cols].min(),  # Giá trị nhỏ nhất
        "max": df[cols].max(),  # Giá trị lớn nhất
    }
)

# Làm tròn 2 chữ số thập phân
summary_df = summary_df.round(2)

# Hiển thị kết quả
print(summary_df)