import pandas as pd
import numpy as np
# 1. ĐỌC DỮ LIỆU
file_path = "GPU_10_Features.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")
text_cols = ["Name", "Manufacturer", "Notebook_GPU", "Memory_Type", "Release_Date"]
for col in text_cols:
    df[col] = (
        df[col]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
df["Manufacturer"] = df["Manufacturer"].str.title().replace({"Nvidia": "NVIDIA", "Amd": "AMD", "Ati": "ATI"})
df["Memory_Type"] = df["Memory_Type"].str.upper().replace({"EDRAM": "eDRAM"})

# Chỉnh Notebook_GPU: Yes = 1, No = 0, Khuyết = NaN (Kiểu Int64 hỗ trợ NaN)
df["Is_Notebook"] = df["Notebook_GPU"].str.lower().map({"yes": 1, "no": 0}).astype("Int64")
# 3. TRÍCH XUẤT DỮ LIỆU SỐ (Dữ liệu khuyết: NaN)
def extract_number(series):
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce",
    )
df["Max_Power_Watts"] = extract_number(df["Max_Power"])
df["Memory_MB"]= extract_number(df["Memory"])
df["Memory_Bus_Bit"] = extract_number(df["Memory_Bus"])
df["Memory_Speed_MHz"] = extract_number(df["Memory_Speed"])
# Xử lý Memory_Bandwidth (Quy về GB/sec, giữ NaN nếu thiếu)
bw_extracted = extract_number(df["Memory_Bandwidth"])
is_mb = df["Memory_Bandwidth"].astype("string").str.contains("MB/sec", case=False, na=False)
df["Memory_Bandwidth_GBs"] = np.where(is_mb, bw_extracted / 1000.0, bw_extracted)

# Quy đổi Memory sang GB
df["Memory_GB"] = df["Memory_MB"] / 1024.0
# 4. CHUẨN HÓA NÀY GIỜ (Khuyết -> NaT / NaN)
date_parsed = pd.to_datetime(df["Release_Date"], format="%d-%b-%Y", errors="coerce")
df["Release_Year"]  = date_parsed.dt.year.astype("Int64")   # Giữ NaN chuẩn dạng số nguyên
df["Release_Month"] = date_parsed.dt.month.astype("Int64")  # Giữ NaN chuẩn dạng số nguyên
df["Release_Date_Parsed"] = date_parsed.dt.strftime("%Y-%m-%d")
# 5. LỌC TRÙNG & XUẤT FILE SẠCH CÓ NATIVE NaN
df = df.drop_duplicates(subset=["Name", "Release_Date"]).copy()

final_cols = [
    "Name", "Manufacturer", "Is_Notebook", "Memory_Type",
    "Release_Date_Parsed", "Release_Year", "Release_Month",
    "Max_Power_Watts", "Memory_MB", "Memory_GB",
    "Memory_Bus_Bit", "Memory_Bandwidth_GBs", "Memory_Speed_MHz"
]
df_final = df[final_cols].rename(columns={"Release_Date_Parsed": "Release_Date"})
# Xuất file CSV (Pandas sẽ ghi các ô NaN thành ô trống rỗng)
output_file = "GPU_10_Features_Clean_NaN.csv"
df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
