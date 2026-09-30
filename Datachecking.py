import pandas as pd
import sys

def check_csv_dtypes(file_path: str):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[X] Lỗi không đọc được file: {e}")
        return

    summary_list = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)

        summary_list.append({
            'Cột (Column)': col,
            'Kiểu Dữ Liệu (Dtype)': dtype_str,
        })

    # Tạo DataFrame tổng hợp để in đẹp trên Terminal
    report_df = pd.DataFrame(summary_list)

    # Cấu hình Pandas hiển thị full cột, full dòng trên Terminal không bị cắt dòng
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    target_file = "GPU_10_Features.csv"

    check_csv_dtypes(target_file)
