import pandas as pd
import sys

def check_csv_dtypes(file_path: str):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[X] Lỗi không đọc được file: {e}")
        return
    total_rows, total_cols = df.shape

    print("=" * 80)
    print(f" FILE: {file_path}")
    print(f" KÍCH THƯỚC: {total_rows:,} dòng x {total_cols} cột (Full thuộc tính)")
    print("=" * 80)

    summary_list = []
    for col in df.columns:
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        dtype_str = str(df[col].dtype)

        # Lấy mẫu giá trị thực tế đầu tiên (không bị null)
        sample = df[col].dropna().iloc[0] if non_null > 0 else "All Null"

        summary_list.append({
            'Cột (Column)': col,
            'Kiểu Dữ Liệu (Dtype)': dtype_str,
            'Non-Null': f"{non_null:,}/{total_rows:,}",
            'Khuyết (Null)': null_count,
            'Ví Dụ Giá Trị Mẫu': repr(sample)
        })

    # Tạo DataFrame tổng hợp để in đẹp trên Terminal
    report_df = pd.DataFrame(summary_list)

    # Cấu hình Pandas hiển thị full cột, full dòng trên Terminal không bị cắt dòng
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)

    print(report_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    target_file = "GPU_10_Features.csv"

    check_csv_dtypes(target_file)
