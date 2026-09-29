import pandas as pd
import numpy as np
from typing import Dict, Any


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    1. KIỂM TRA TRÙNG LẶP (DUPLICATE CHECKING)
    - Trùng lặp hoàn toàn (Exact Duplicate Rows)
    - Trùng lặp theo tên linh kiện (Duplicate GPU Names)
    """
    exact_duplicates_count = int(df.duplicated().sum())

    # Kiểm tra trùng tên GPU
    duplicate_names_mask = df.duplicated(subset=['Name'], keep=False)
    duplicate_names_count = int(df.duplicated(subset=['Name']).sum())
    duplicate_names_detail = df[duplicate_names_mask][['Manufacturer', 'Name', 'Release_Date', 'Release_Year']]

    return {
        "exact_duplicates_count": exact_duplicates_count,
        "duplicate_names_count": duplicate_names_count,
        "duplicate_names_detail": duplicate_names_detail
    }


def check_logic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    2. KIỂM TRA LOGIC VÀ QUY TẮC MIỀN DỮ LIỆU
    Thực hiện kiểm tra các điều kiện ràng buộc kỹ thuật của GPU:
    - Logic Độ phân giải hiển thị(resolution):
      Res_Width phải lớn hơn Res_Height và cả hai phải > 0.
    - Logic Thông số vật lý / Phần cứng:
      Các chỉ số điện năng, bộ nhớ, bus width, bandwidth, clock speed phải > 0.
    - Logic Danh mục hợp lệ:
      Manufacturer phải thuộc các hãng hợp lệ và SLI_Crossfire chỉ nhận 'Yes'/'No'.
    """
    logic_results = {}

    #Logic Độ phân giải
    invalid_res = df[
        (df['Res_Width'] <= df['Res_Height']) |
        (df['Res_Width'] <= 0) |
        (df['Res_Height'] <= 0)
    ]
    logic_results["invalid_resolution_count"] = len(invalid_res)
    logic_results["invalid_resolution_detail"] = invalid_res[['Name', 'Res_Width', 'Res_Height']]

    # Thông số vật lý
    hw_columns = [
        'Max_Power_Watts', 'Memory_MB', 'Memory_Bus_Bit',
        'Memory_Bandwidth_GBs', 'Memory_Speed_MHz'
    ]
    invalid_hw = df[(df[hw_columns] <= 0).any(axis=1)]
    logic_results["invalid_hardware_values_count"] = len(invalid_hw)
    logic_results["invalid_hardware_detail"] = invalid_hw[['Name'] + hw_columns]

    # Logic Giá trị danh mục hợp lệ
    valid_manufacturers = {'Nvidia', 'AMD', 'Intel', 'ATI'}
    valid_sli_values = {'Yes', 'No'}

    invalid_mfg = df[~df['Manufacturer'].isin(valid_manufacturers)]
    invalid_sli = df[~df['SLI_Crossfire'].isin(valid_sli_values)]

    logic_results["invalid_manufacturer_count"] = len(invalid_mfg)
    logic_results["invalid_sli_count"] = len(invalid_sli)

    return logic_results

def run_full_data_audit(file_path: str):
    """
    HÀM ĐIỀU HÀNH CHÍNH (MASTER AUDIT FUNCTION)
    Tập trung vào xử lý và kiểm tra dữ liệu sau khi đã định dạng.
    """
    print(f"=== BẮT ĐẦU XỬ LÝ & KIỂM TRA DỮ LIỆU: {file_path} ===")
    df = pd.read_csv(file_path)
    print(f"Kích thước tập dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột\n")

    # Kiểm tra trùng lặp
    dup_res = check_duplicates(df)
    print("1. KẾT QUẢ KIỂM TRA TRÙNG LẶP:")
    print(f"   - Số dòng trùng lặp hoàn toàn: {dup_res['exact_duplicates_count']}")
    print(f"   - Số tên GPU bị trùng lặp: {dup_res['duplicate_names_count']}")
    if dup_res['duplicate_names_count'] > 0:
        print("   -> Chi tiết các GPU trùng tên:")
        print(dup_res['duplicate_names_detail'].to_string(index=False))
    print("-" * 60)

    # Kiểm tra Logic
    logic_res = check_logic(df)
    print("2. KẾT QUẢ KIỂM TRA LOGIC và ĐIỀU KIỆN")
    print(f"   - Lỗi logic Độ phân giải : {logic_res['invalid_resolution_count']} trường hợp")
    print(f"   - Giá trị phần cứng (<= 0): {logic_res['invalid_hardware_values_count']} trường hợp")
    print(f"   - Nhà sản xuất không hợp lệ: {logic_res['invalid_manufacturer_count']} trường hợp")
    print(f"   - Giá trị SLI/Crossfire không hợp lệ: {logic_res['invalid_sli_count']} trường hợp")
    print("-" * 60)
# Chạy trực tiếp
if __name__ == "__main__":
    run_full_data_audit('GPU_10_Features_Normalized.csv')
