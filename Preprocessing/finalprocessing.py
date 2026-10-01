import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def audit_and_clean_gpu_dataset(
    input_path: str,
    output_path: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    1. Đọc file nguyên trạng và kiểm kê tỷ lệ giá trị khuyết thiếu (NaN).
    2. Kiểm tra các điều kiện logic cứng (Ràng buộc miền giá trị phần cứng,
       công thức quy đổi MB/GB, danh mục hợp lệ).
    3. Xử lý khoảng trắng thừa (string strip) và loại bỏ các dòng lỗi logic (nếu có).
    4. Xử lý trùng lặp:
       - Bỏ dòng trùng 100%.
       - Đối với GPU trùng tên (Name duplicate), giữ lại dòng có số lượng thông tin
         khả dụng (non-null) cao nhất.
    5. Định dạng kiểu dữ liệu tối ưu cho mô hình thống kê trong Python/R.
    6. Xuất ra file CSV sạch cuối cùng.
    """
    df = pd.read_csv(input_path)
    original_shape = df.shape
    # Log xử lý
    audit_log = {
        "original_shape": original_shape,
        "missing_summary": None,
        "removed_exact_duplicates": 0,
        "removed_invalid_logic": 0,
        "duplicate_names_handled": 0,
        "final_shape": None
    }
    # BƯỚC 1: XỬ LÝ TRÙNG LẶP HOÀN TOÀN
    exact_dup_count = df.duplicated().sum()
    if exact_dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    audit_log["removed_exact_duplicates"] = exact_dup_count

    # BƯỚC 4: KIỂM TRA & BỎ CÁC DÒNG VI PHẠM LOGIC CỨNG
    # Chỉ số phần cứng không được <= 0 (bỏ qua giá trị NaN)
    hw_cols = [
        'Max_Power_Watts', 'Memory_MB', 'Memory_GB',
        'Memory_Bus_Bit', 'Memory_Bandwidth_GBs', 'Memory_Speed_MHz'
    ]
    invalid_hw_mask = (df[hw_cols] <= 0).any(axis=1)

    # Quy đổi dung lượng bộ nhớ bị lệch (Memory_GB != Memory_MB / 1024)
    valid_mem_mask = df['Memory_GB'].notna() & df['Memory_MB'].notna()
    mem_mismatch_mask = valid_mem_mask & ((df['Memory_GB'] - (df['Memory_MB'] / 1024.0)).abs() > 1e-4)

  # Giá trị danh mục không thuộc tập hợp cho phép
      # 1. Khai báo các tập hợp danh mục chuẩn theo ngành phần cứng GPU)
    DOMAIN_VALID_MANUFACTURERS = {
        'NVIDIA', 'AMD', 'INTEL', 'ATI', '3DFX', 'MATROX',
        'VIA', 'SIS', 'S3', 'QUALCOMM', 'APPLE', 'ARM'
    }
    DOMAIN_VALID_MEMORY_TYPES = {
        'DDR', 'DDR2', 'DDR3', 'DDR4', 'DDR5',
        'GDDR', 'GDDR2', 'GDDR3', 'GDDR4', 'GDDR5', 'GDDR5X', 'GDDR6', 'GDDR6X',
        'HBM-1', 'HBM-2', 'HBM-3', 'EDRAM', 'LPDDR', 'LPDDR2', 'LPDDR3', 'LPDDR4', 'LPDDR5'
    }
    DOMAIN_VALID_IS_NOTEBOOK = {0, 1}

    # 2. Chuẩn hóa dạng chuỗiđể tránh bắt nhầm lỗi định dạng
    mfg_clean = df['Manufacturer'].astype(str).str.strip().str.upper()
    mem_type_clean = df['Memory_Type'].astype(str).str.strip().str.upper()

    # 3. Tạo boolean mask lọc các bản ghi vi phạm để không bắt lỗi dòng NaN
    invalid_mfg_mask = df['Manufacturer'].notna() & (~mfg_clean.isin(DOMAIN_VALID_MANUFACTURERS))
    invalid_notebook_mask = df['Is_Notebook'].notna() & (~df['Is_Notebook'].isin(DOMAIN_VALID_IS_NOTEBOOK))
    invalid_mem_type_mask = df['Memory_Type'].notna() & (~mem_type_clean.isin(DOMAIN_VALID_MEMORY_TYPES))
    # Tổng hợp các dòng vi phạm điều kiện kỹ thuật
    total_invalid_mask = (
        invalid_hw_mask | mem_mismatch_mask |
        invalid_mfg_mask | invalid_notebook_mask | invalid_mem_type_mask
    )
    invalid_rows_count = total_invalid_mask.sum()

    if invalid_rows_count > 0:
        df = df[~total_invalid_mask].reset_index(drop=True)
    audit_log["removed_invalid_logic"] = invalid_rows_count

    # BƯỚC 5: XỬ LÝ TRÙNG LẶP KHÓA TÊN GPU
    # giữ lại dòng chứa nhiều thông tin nhất của GPU đó.
    initial_name_dup = df['Name'].duplicated().sum()
    if initial_name_dup > 0:
        # Tính mật độ dữ liệu khả dụng cho từng dòng
        df['__data_density'] = df.notna().sum(axis=1)
        # Sắp xếp tên tăng dần, mật độ dữ liệu giảm dần
        df = df.sort_values(by=['Name', '__data_density'], ascending=[True, False])
        # Lọc bỏ trùng tên, giữ lại bản ghi đầy đủ thông tin nhất
        df = df.drop_duplicates(subset=['Name'], keep='first')
        df = df.drop(columns=['__data_density']).reset_index(drop=True)
    audit_log["duplicate_names_handled"] = initial_name_dup

    # BƯỚC 6: ĐỊNH DẠNG KIỂU DỮ LIỆU TỐI ƯU CHO PHÂN TÍCH
    if 'Release_Year' in df.columns:
        df['Release_Year'] = df['Release_Year'].astype('Int64')
    if 'Release_Month' in df.columns:
        df['Release_Month'] = df['Release_Month'].astype('Int64')
    if 'Is_Notebook' in df.columns:
        df['Is_Notebook'] = df['Is_Notebook'].astype('int64')

    # BƯỚC 7: XUẤT FILE CSV CUỐI CÙNG
    df.to_csv(output_path, index=False)
    audit_log["final_shape"] = df.shape

    print("\n1. KẾT QUẢ XỬ LÝ LOGIC:")
    print(f"   - Dòng trùng lặp 100% bị xóa: {audit_log['removed_exact_duplicates']}")
    print(f"   - Dòng vi phạm logic kỹ thuật/phần cứng bị xóa: {audit_log['removed_invalid_logic']}")
    print(f"   - GPU trùng tên đã được xử lý (Giữ dòng tối ưu nhất): {audit_log['duplicate_names_handled']}")

    print("\n 2. TỔNG KẾT ĐẦU RA:")
    print(f"- Kích thước tập dữ liệu cuối cùng: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"- Xuất file thành công tại: {output_path}")

    return df, audit_log

# Thực thi trực tiếp
if __name__ == "__main__":
    file_dau_vao = 'GPU_10_Features_Clean_NaN.csv'
    file_dau_ra = 'GPU_10_Features_Processed_Final.csv'
    df_clean, report = audit_and_clean_gpu_dataset(file_dau_vao, file_dau_ra)
