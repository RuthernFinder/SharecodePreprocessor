# import các thư viện cần thiết
library(stringr)
library(tidyr)
library(dplyr)
library(zoo)
library(Metrics)
library(caret)
library(MASS)
library(ggplot2)
library(reshape2)
library(mltools)
library(DescTools)
library(plotly)
library(mice)
# 1. Khai báo thư viện và đọc dữ liệu từ gốc
# Sử dụng na.strings để tự động nhận diện các giá trị rỗng, khoảng trắng, hoặc ký tự nhiễu (như "\n- ") thành NA (Not Available)
gpu_data <- read.csv("/kaggle/input/datasets/iliassekkaf/computerparts/All_GPUs.csv",
                     stringsAsFactors = FALSE,
                     na.strings = c("NA", "", "-", "\n- ", "nan", "None"))

# Kích thước tập dữ liệu
total_rows <- nrow(gpu_data) # Kỳ vọng: 3406
total_cols <- ncol(gpu_data) # Kỳ vọng: 34

# 2. Xây dựng hàm tính toán tỷ lệ thiếu hụt từ con số 0
# is.na() tạo ra ma trận logic (TRUE/FALSE) cho toàn bộ bảng.
# colSums() sẽ cộng tổng các giá trị TRUE (tương đương 1) theo từng cột.
missing_counts <- colSums(is.na(gpu_data))

# Tính phần trăm (Số lượng thiếu / Tổng số dòng * 100)
missing_percentages <- (missing_counts / total_rows) * 100

# 3. Gom nhóm và định dạng thành một Data Frame hoàn chỉnh để dễ phân tích
missing_analysis <- data.frame(
  Thuoc_Tinh = names(missing_percentages),
  So_Luong_Thieu = missing_counts,
  Phan_Tram_Thieu_Hut = round(missing_percentages, 2) # Làm tròn 2 chữ số thập phân
)

# Sắp xếp lại bảng theo tỷ lệ thiếu hụt giảm dần (từ cao nhất đến thấp nhất)
missing_analysis <- missing_analysis[order(-missing_analysis$Phan_Tram_Thieu_Hut), ]

# Xóa index dòng mặc định để bảng in ra gọn gàng hơn
rownames(missing_analysis) <- NULL

# Hiển thị kết quả ra console
print("--- BẢNG XẾP HẠNG TỶ LỆ THIẾU HỤT CỦA 34 THUỘC TÍNH ---")
print(missing_analysis)


# =====================================================================
# BỔ SUNG: TIỀN XỬ LÝ VÀ PHÂN TÍCH THỐNG KÊ CƠ BẢN
# =====================================================================

# 4. Viết hàm regex (biểu thức chính quy) để bóc tách số liệu khỏi chuỗi văn bản
# Dữ liệu gốc chứa chữ (vd: "738 MHz", "141 Watts"). Ta cần trích xuất số để tính Mean, Median.
clean_to_numeric <- function(column_data) {
  # gsub() tìm và thay thế tất cả ký tự KHÔNG phải là số (0-9) hoặc dấu chấm (.) bằng rỗng
  cleaned_text <- gsub("[^0-9.]", "", column_data)
  return(as.numeric(cleaned_text))
}

# Áp dụng hàm làm sạch cho các cột hiệu năng vật lý quan trọng
gpu_data$Core_Speed_Clean <- clean_to_numeric(gpu_data$Core_Speed)
gpu_data$Memory_Bandwidth_Clean <- clean_to_numeric(gpu_data$Memory_Bandwidth)
gpu_data$Max_Power_Clean <- clean_to_numeric(gpu_data$Max_Power)
gpu_data$Memory_Clean <- clean_to_numeric(gpu_data$Memory)

# 5. In thống kê mô tả (Min, 1st Qu, Median, Mean, 3rd Qu, Max) của các biến cốt lõi
print("--- THỐNG KÊ MÔ TẢ CÁC BIẾN SỐ CHÍNH ---")
summary(gpu_data[, c("Core_Speed_Clean", "Memory_Bandwidth_Clean", "Max_Power_Clean", "Memory_Clean")])

# 6. Trực quan hóa tỷ lệ thiếu hụt bằng ggplot2
# Cần cài đặt gói nếu chưa có: install.packages("ggplot2")
library(ggplot2)

# Tạo biểu đồ thanh ngang (Bar chart)
ggplot(missing_analysis, aes(x = reorder(Thuoc_Tinh, Phan_Tram_Thieu_Hut), y = Phan_Tram_Thieu_Hut)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black", alpha = 0.8) +
  coord_flip() + # Xoay ngang trục để 34 tên biến không bị đè lên nhau
  geom_text(aes(label = paste0(Phan_Tram_Thieu_Hut, "%")),
            hjust = -0.1, size = 3, color = "darkred") + # Gắn số % trực tiếp lên biểu đồ
  labs(
    title = "Đánh giá mức độ toàn vẹn dữ liệu: Tỷ lệ khuyết của 34 thuộc tính GPU",
    subtitle = paste("Phân tích trên tổng số mẫu N =", total_rows),
    x = "Thuộc tính phần cứng",
    y = "Tỷ lệ thiếu hụt (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.y = element_text(size = 9)
  )

  # 1. Đọc dữ liệu gốc
  # Đảm bảo set working directory (setwd) đến thư mục chứa file All_GPUs.csv trước khi chạy
  gpu_data <- read.csv("/kaggle/input/datasets/iliassekkaf/computerparts/All_GPUs.csv", stringsAsFactors = FALSE)

  # 2. Định nghĩa danh sách các cột cần giữ lại (khớp chính xác chữ hoa/chữ thường trong CSV)
  columns_to_keep <- c(
    "L2_Cache",
    "Manufacturer",
    "Max_Power",
    "Memory",
    "Memory_Bus",
    "Memory_Bandwidth",
    "Memory_Speed",
    "Name",
    "Release_Date",
    "Resolution_WxH"
  )

  # ==========================================
  # CÁCH 1: Lọc bằng Base R (R gốc - Không cần thư viện)
  # ==========================================
  # Sử dụng cú pháp [hàng, cột] để trích xuất dữ liệu
  gpu_cleaned_base <- gpu_data[, columns_to_keep]

  # Kiểm tra nhanh kích thước bảng mới (Kỳ vọng: 3406 dòng x 10 cột)
  print(paste("Kích thước dữ liệu mới (Base R):", nrow(gpu_cleaned_base), "dòng,", ncol(gpu_cleaned_base), "cột"))


  # ==========================================
  # CÁCH 2: Lọc bằng thư viện dplyr (Khuyên dùng)
  # ==========================================
  # Cài đặt thư viện nếu chưa có: install.packages("dplyr")
  library(dplyr)

  # Sử dụng toán tử pipe (%>%) và hàm select() để code trực quan hơn
  gpu_cleaned_dplyr <- gpu_data %>%
    select(all_of(columns_to_keep))

  # In cấu trúc bảng mới để kiểm tra các kiểu dữ liệu
  str(gpu_cleaned_dplyr)


  # 3. Xuất kết quả ra file CSV mới để tiếp tục phân tích sau này
  # Tham số row.names = FALSE giúp tránh việc R tự động thêm một cột STT không cần thiết
  write.csv(gpu_cleaned_base, "Filtered_GPUs.csv", row.names = FALSE)
  print("Đã lưu tập dữ liệu mới vào file 'Filtered_GPUs.csv'")# 1. Đọc dữ liệu gốc
  # Đảm bảo set working directory (setwd) đến thư mục chứa file All_GPUs.csv trước khi chạy
  gpu_data <- read.csv("/kaggle/input/datasets/iliassekkaf/computerparts/All_GPUs.csv", stringsAsFactors = FALSE)

  # 2. Định nghĩa danh sách các cột cần giữ lại (khớp chính xác chữ hoa/chữ thường trong CSV)
  columns_to_keep <- c(
    "L2_Cache",
    "Manufacturer",
    "Max_Power",
    "Memory",
    "Memory_Bus",
    "Memory_Bandwidth",
    "Memory_Speed",
    "Name",
    "Release_Date",
    "Resolution_WxH"
  )

  # ==========================================
  # CÁCH 1: Lọc bằng Base R (R gốc - Không cần thư viện)
  # ==========================================
  # Sử dụng cú pháp [hàng, cột] để trích xuất dữ liệu
  gpu_cleaned_base <- gpu_data[, columns_to_keep]

  # Kiểm tra nhanh kích thước bảng mới (Kỳ vọng: 3406 dòng x 10 cột)
  print(paste("Kích thước dữ liệu mới (Base R):", nrow(gpu_cleaned_base), "dòng,", ncol(gpu_cleaned_base), "cột"))


  # ==========================================
  # CÁCH 2: Lọc bằng thư viện dplyr (Khuyên dùng)
  # ==========================================
  # Cài đặt thư viện nếu chưa có: install.packages("dplyr")
  library(dplyr)

  # Sử dụng toán tử pipe (%>%) và hàm select() để code trực quan hơn
  gpu_cleaned_dplyr <- gpu_data %>%
    select(all_of(columns_to_keep))

  # In cấu trúc bảng mới để kiểm tra các kiểu dữ liệu
  str(gpu_cleaned_dplyr)


  # 3. Xuất kết quả ra file CSV mới để tiếp tục phân tích sau này
  # Tham số row.names = FALSE giúp tránh việc R tự động thêm một cột STT không cần thiết
  write.csv(gpu_cleaned_base, "Filtered_GPUs.csv", row.names = FALSE)
  print("Đã lưu tập dữ liệu mới vào file 'Filtered_GPUs.csv'")
