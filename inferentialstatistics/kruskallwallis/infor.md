# Các bước xử lý trong krusall 
| Tầng | Phép kiểm định | H0 (mặc định) | H1 |
| :---: | :--- | :--- | :--- |
| 1 | Kruskal-Wallis | Phân phối băng thông của 5 chuẩn giống nhau | Ít nhất một chuẩn có băng thông có xu hướng cao hơn hoặc thấp hơn |
| 2 | Dunn (10 cặp, hai phía) | Hai chuẩn trong cặp không khác nhau | Hai chuẩn trong cặp khác nhau |
1. Lọc ra các thành phần không hợp lệ hay là bỏ NaN ứng với các băng thông không có dữ liệu. Sau đó đếm các phần tử còn dùng được cho mỗi mmr type
2. Sắp xếp thứ hạng và rồi tổng hạng của tụi nó lại và đề ra được hạng trung bình
3. từ công thức chuẩn của krusal ta có được H khi chưa lọc qua dữ liệu trùn
4. sau khi tính được trùng C và hiệu chỉnh H thì được H gốc 
5. Số bậc tự do của mẫu này là 5-1. tính được P valude theo chi bình phương rồi kết luận p value là có sự liên quan của ít nhất một phần tử so với các phần tử khác

# Duntest và bonferni
lấy bonferni( alpha) mặc định là 5% 
xét 10 cặp theo thứu tự và tính các hạng trung bình, P của nó (với Ho của từng cặp là như nhau và được đánh dấu trên bảng)

---------------------------
# # Lưu ý về một số file
imageforsta.py là file dùng để tạo ảnh trực quan về số liệu trước xử lý
ppnc.py dùng để phân 5 nhóm và các tính toán sơ bộ các thành phần băng thông để củng cố cho kiểm định buộc phải là phi tham số
kruskall.py là file chính để tính toán toàn bộ và exposure in report
