Triển khai gọn nhẹ **thuật toán Bầy Ong Nhân Tạo (ABC)** cho bài toán **tối thiểu hóa liên tục**

## Tính năng
- Ba pha kinh điển: **Employed → Onlooker → Scout** với chọn **tham lam (greedy)**.
- **Cập nhật 1 chiều** ngẫu nhiên mỗi lần (đúng ABC cơ sở).
- **Roulette selection** cho onlooker dựa trên **fitness** an toàn cho tối thiểu hóa.
- **Xử lý biên** bằng cắt ngưỡng (clipping).
- Có thể **tái lập** kết quả bằng `seed`.
