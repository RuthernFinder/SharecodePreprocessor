import pandas as pd

# 1. Đọc file CSV gốc chứa toàn bộ 34 thuộc tính
input_file = "All_GPUs.csv"
df = pd.read_csv(input_file)

# 2. Danh sách 10 thuộc tính cần giữ lại
selected_columns = [
    'Manufacturer',
    'Max_Power',
    'Memory',
    'Memory_Bus',
    'Memory_Bandwidth',
    'Memory_Speed',
    'Name',
    'Release_Date',
    'Resolution_WxH',
    'SLI_Crossfire'
]

# 3. Lọc dữ liệu chỉ lấy 10 cột đã chọn
df_selected = df[selected_columns]

# 4. Xuất dữ liệu ra file CSV mới (không lưu chỉ số dòng index)
output_file = "GPU_10_Features.csv"
df_selected.to_csv(output_file, index=False, encoding='utf-8-sig')
