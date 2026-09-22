import pandas as pd
import numpy as np

# 1. Đọc file CSV chứa đúng 10 thuộc tính
file_path = "GPU_10_Features.csv"
df = pd.read_csv(file_path)

print(f"=== KÍCH THƯỚC BAN ĐẦU: {df.shape[0]} dòng, {df.shape[1]} cột ===")

# xóa các dòng khuyết dữ liệu
df_clean = df.dropna().copy()

print(f"=== KÍCH THƯỚC SAU KHI VỨT BỎ DỮ LIỆU KHUYẾT: {df_clean.shape[0]} dòng ===")

# 3. CHUẨN HÓA DỮ LIỆU CHUỖI VĂN BẢN (STRING CLEANING)
# Xóa khoảng trắng thừa và ký tự xuống dòng (\n) ở Release_Date và Name
df_clean['Name'] = df_clean['Name'].astype(str).str.strip()
df_clean['Manufacturer'] = df_clean['Manufacturer'].astype(str).str.strip()
df_clean['SLI_Crossfire'] = df_clean['SLI_Crossfire'].astype(str).str.strip()
df_clean['Release_Date_Clean'] = df_clean['Release_Date'].astype(str).str.strip()

# 4. TRÍCH XUẤT VÀ CHUẨN HÓA KIỂU SỐ (NUMERIC EXTRACT)
# Max_Power ('141 Watts' -> 141.0)
df_clean['Max_Power_Watts'] = df_clean['Max_Power'].astype(str).str.extract(r'([\d\.]+)').astype(float)

# Memory ('1024 MB ' -> 1024.0)
df_clean['Memory_MB'] = df_clean['Memory'].astype(str).str.extract(r'([\d\.]+)').astype(float)

# Memory_Bus ('256 Bit ' -> 256.0)
df_clean['Memory_Bus_Bit'] = df_clean['Memory_Bus'].astype(str).str.extract(r'([\d\.]+)').astype(float)

# Memory_Bandwidth ('64GB/sec' -> 64.0)
df_clean['Memory_Bandwidth_GBs'] = df_clean['Memory_Bandwidth'].astype(str).str.extract(r'([\d\.]+)').astype(float)

# Memory_Speed ('1000 MHz' -> 1000.0)
df_clean['Memory_Speed_MHz'] = df_clean['Memory_Speed'].astype(str).str.extract(r'([\d\.]+)').astype(float)

# Resolution_WxH ('2560x1600' -> Res_Width: 2560.0, Res_Height: 1600.0)
df_clean[['Res_Width', 'Res_Height']] = df_clean['Resolution_WxH'].astype(str).str.extract(r'(\d+)x(\d+)').astype(float)

# 5. CHUẨN HÓA KIỂU THỜI GIAN (DATETIME)
df_clean['Release_Date_Parsed'] = pd.to_datetime(df_clean['Release_Date_Clean'], errors='coerce')
df_clean['Release_Year'] = df_clean['Release_Date_Parsed'].dt.year

# 6. LỌC LẠI CÁC CỘT ĐÃ ĐƯỢC CHUẨN HÓA HOÀN CHỈNH
final_normalized_cols = [
    'Manufacturer',
    'Name',
    'SLI_Crossfire',
    'Release_Date_Clean',
    'Release_Year',
    'Max_Power_Watts',
    'Memory_MB',
    'Memory_Bus_Bit',
    'Memory_Bandwidth_GBs',
    'Memory_Speed_MHz',
    'Res_Width',
    'Res_Height'
]

df_final = df_clean[final_normalized_cols].rename(columns={'Release_Date_Clean': 'Release_Date'})

# 7. XUẤT RA FILE CSV MỚI HOÀN TOÀN CHUẨN HÓA
output_file = "GPU_10_Features_Normalized.csv"
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n[X] Đã chuẩn hóa xong! File lưu tại: {output_file}")
print("\n=== BẢNG TỔNG HỢP KIỂU DỮ LIỆU SAU CHUẨN HÓA ===")
print(df_final.dtypes)

print("\n=== 5 DÒNG ĐẦU DỮ LIỆU CHUẨN HÓA ===")
print(df_final.head())
