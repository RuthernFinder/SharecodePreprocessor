import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_data(file_path: str, export_csv: bool = True):
    """
    Hàm đọc dữ liệu GPU, tính toán độ khuyết dữ liệu trên từng tiêu chí,
    sắp xếp thứ hạng và trực quan hóa kết quả.
        file_path(str): Đường dẫn tới tập tin CSV.
        export_csv(bool): Lưu báo cáo xếp hạng ra file CSV nếu là True.
    """
    print(f"Đang đọc tập tin: {file_path}")
    df = pd.read_csv(file_path)
    total_samples = len(df)
    total_features = len(df.columns)
    print(f"Tổng số mẫu (Rows): {total_samples}")
    print(f"Tổng số tiêu chí (Columns): {total_features}\n")

    # Số dữ liệu khuyết từng thuộc tính
    missing_count = df.isnull().sum() #Hàm cho biết đâu là phần tử null và tổng tất cả
    # Tỷ lệ khuyết dữ liệu
    missing_percent = (missing_count / total_samples) * 100

    # Tạo DataFrame
    missing_summary = pd.DataFrame({
        'Criteria': df.columns,
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_percent
    })

    # Sắp xếp giảm dần theo số lượng mẫu bị thiếu
    missing_summary = missing_summary.sort_values(
        by='Missing_Count', ascending=False
    ).reset_index(drop=True)

    # Thêm cột Thứ hạng (Rank)
    missing_summary.index = missing_summary.index + 1
    missing_summary.index.name = 'Rank'

    #kiểm tra độ khuyết dữ liệu từng thuộc tính
    def classify_severity(pct):
        if pct == 0:
            return "Đầy đủ (0%)"
        elif pct < 5:
            return "Rất thấp (<5%)"
        elif pct < 20:
            return "Thấp (5-20%)"
        elif pct < 50:
            return "Trung bình (20-50%)"
        else:
            return "Nghiêm trọng (>50%)"

    missing_summary['Severity_Level'] = missing_summary['Missing_Percentage'].apply(classify_severity)
    # in ra bảng xếp hạng
    print("BẢNG XẾP HẠNG ĐỘ KHUYẾT DỮ LIỆU")
    print(missing_summary.to_string())

    # Xuất báo cáo CSV nếu được yêu cầu
    if export_csv:
        output_filename = "GPU_Missing_Data_Report.csv"
        missing_summary.to_csv(output_filename, encoding='utf-8-sig')
        print(f"\n[X] Đã xuất báo cáo chi tiết ra file: {output_filename}")

    # 3. Trực quan hóa dữ liệu khuyết
    plot_missing_data(missing_summary, df)

    return missing_summary

def plot_missing_data(missing_summary: pd.DataFrame, df: pd.DataFrame):# hàm vẽ biểu đồ
    """
    Hàm vẽ biểu đồ cột tỷ lệ khuyết dữ liệu và biểu đồ Heatmap tương quan khuyết.
    """
    sns.set_theme(style="whitegrid")

    # --- Biểu đồ 1: Tỷ lệ khuyết dữ liệu theo tiêu chí ---
    plt.figure(figsize=(14, 8))

    # Chỉ lấy các cột có dữ liệu bị khuyết để vẽ biểu đồ
    missing_only = missing_summary[missing_summary['Missing_Count'] > 0]

    barplot = sns.barplot(
        x='Missing_Percentage',
        y='Criteria',
        data=missing_only,
        palette='magma'
    )

    plt.title('Tỷ lệ % khuyết dữ liệu của từng tiêu chí GPU', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Tỷ lệ khuyết (%)', fontsize=12)
    plt.ylabel('Tiêu chí (Criteria)', fontsize=12)
    plt.xlim(0, 100)

    # Hiển thị con số % chính xác trên từng thanh biểu đồ
    for p in barplot.patches:
        width = p.get_width()
        if width > 0:
            barplot.annotate(
                f'{width:.1f}%',
                (width + 1, p.get_y() + p.get_height() / 2.),
                ha='left', va='center',
                fontsize=9, color='black'
            )

    plt.tight_layout()
    # plt.savefig('missing_data_barplot.png', dpi=300)

    # --- Biểu đồ 2: Heatmap tương quan khuyết dữ liệu giữa các thuộc tính quan trọng ---
    # Kiểm tra sự xuất hiện đồng thời của các giá trị Null giữa các thuộc tính
    plt.figure(figsize=(12, 10))
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    if cols_with_missing:
        # Ma trận True/False về sự khuyết dữ liệu
        null_matrix = df[cols_with_missing].isnull()

        # Ma trận tương quan giữa sự khuyết dữ liệu của các cột
        corr_matrix = null_matrix.corr()

        sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            annot=False,
            linewidths=0.5,
            cbar_kws={'label': 'Hệ số tương quan khuyết (1 = Khuyết cùng nhau)'}
        )
        plt.title('Ma trận tương quan khuyết dữ liệu (Missingness Correlation)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        # plt.savefig('missing_data_correlation.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    file_path = "All_GPUs.csv"
    missing_report = analyze_missing_data(file_path,False)
