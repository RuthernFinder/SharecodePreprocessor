import pandas as pd
import numpy as np
import re

# Đọc dữ liệu từ file data.
df = pd.read_csv("GPU_10_Features_Processed_Final.csv")


# Danh sách các cột định lượng cần thống kê mô tả
cols = [
    "Memory_MB",
    "Memory_Bandwidth_GBs",
    "Memory_Speed_MHz",
    "Process_nm",
    "Memory_Bus_Bit",
    "Is_Notebook",
    "Release_Year",
    "Release_Month",  # Thêm tính toán cho cột tháng
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
