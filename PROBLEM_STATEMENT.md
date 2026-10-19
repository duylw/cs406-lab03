Dựa trên nội dung tài liệu **Lab-03: Image Enhancing**, mục tiêu là xây dựng một ứng dụng nâng cao chất lượng ảnh với các yêu cầu cụ thể sau:

### 1. Đầu vào (Input)

* Xây dựng tính năng cho phép tải lên (upload) một ảnh màu từ máy tính cá nhân.



### 2. Các chức năng xử lý ảnh chính

* **Yêu cầu 1: Khử nhiễu / Làm mịn (Denoising / Smoothing)**

* Thực hiện làm mịn và loại bỏ nhiễu trên ảnh.


* Dựa theo gợi ý hiển thị, ứng dụng cần áp dụng các bộ lọc như **Mean filter** (lọc trung bình) và **Median filter** (lọc trung vị) trên các dạng nhiễu thường gặp (ví dụ: Gaussian noise, Salt & Pepper noise) để so sánh kết quả.




* **Yêu cầu 2: Làm sắc nét ảnh (Sharpening)**

* Tăng cường độ rõ nét và tương phản chi tiết của ảnh (giảm độ mờ).




* **Yêu cầu 3: Phát hiện biên cạnh (Edge Detection)**

* Triển khai đầy đủ cả 3 bộ lọc biên cạnh:


1. **Sobel**

2. **Prewitt**

3. **Canny Edge**






### 3. Đầu ra & Giao diện hiển thị (Output)

* Hiển thị kết quả biên cạnh với cả 3 bộ lọc đã xử lý.


* Trình bày trực quan so sánh giữa ảnh gốc, ảnh sau xử lý (hoặc ảnh có thêm nhiễu và kết quả khử nhiễu) theo dạng lưới/hàng ngang như các mẫu gợi ý trong slide.