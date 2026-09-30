import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_missing_data_plots(file_path: str, show_plot: bool = True, save_plot: bool = True):
    """
    Đọc dữ liệu GPU và trực quan hóa độ khuyết dữ liệu bằng biểu đồ.
    Hoàn toàn không in bất kỳ thông tin nào ra Terminal.

    Parameters:
        file_path (str): Đường dẫn tới tập tin CSV (ví dụ: 'All_GPUs.csv')
        show_plot (bool): Hiển thị biểu đồ ra màn hình nếu là True.
        save_plot (bool): Lưu biểu đồ thành file hình ảnh nếu là True.
    """
    # 1. Đọc dữ liệu và tính toán độ khuyết
    df = pd.read_csv(file_path)
    total_samples = len(df)
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / total_samples) * 100

    # Bảng tổng hợp dữ liệu khuyết
    missing_summary = pd.DataFrame({
        'Criteria': df.columns,
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_percent
    }).sort_values(by='Missing_Count', ascending=False).reset_index(drop=True)

    # Cấu hình giao diện Seaborn
    sns.set_theme(style="whitegrid")

    # --- Biểu đồ 1: Biểu đồ cột tỷ lệ khuyết dữ liệu theo tiêu chí ---
    missing_only = missing_summary[missing_summary['Missing_Count'] > 0]

    if not missing_only.empty:
        plt.figure(figsize=(14, 8))

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

        # Hiển thị con số % chính xác lên trên từng thanh biểu đồ
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
        if save_plot:
            plt.savefig('missing_data_barplot.png', dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()

    # --- Biểu đồ 2: Heatmap tương quan khuyết dữ liệu giữa các thuộc tính ---
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    if cols_with_missing:
        plt.figure(figsize=(12, 10))

        # Ma trận Boolean (True khi ô bị khuyết/NaN)
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
        if save_plot:
            plt.savefig('missing_data_correlation.png', dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()

# --- Thực thi chương trình ---
if __name__ == "__main__":
    file_path = "All_GPUs.csv"
    generate_missing_data_plots(file_path, show_plot=True, save_plot=True)
