# ABC (Artificial Bee Colony) 

## Cấu trúc thư mục
abc-base/
├─ abc.py
├─ run_demo.py 
├─ README.md 
└─ PSEUDOCODE.md 

## Cách chạy nhanh
```bash
pip install numpy
python run_demo.py
Thay hàm mục tiêu trong run_demo.py bằng hàm của bạn (nhận np.ndarray 1D, trả về float).

Điều chỉnh bounds, sn, limit, max_cycles theo bài toán.

API rút gọn — mục đích: “cheat-sheet” cách gọi class và đọc kết quả
Dùng để nhìn nhanh chữ ký hàm, đối số cần truyền và dữ liệu trả về — không cần mở mã nguồn.

python
Copy code
ArtificialBeeColony(
  obj_func,               # Callable[[np.ndarray], float] — hàm mục tiêu (tối thiểu hóa)
  bounds=(lower, upper),  # (np.ndarray, np.ndarray) — mỗi cái shape (D,)
  sn,                     # số nguồn thức ăn (int)
  limit,                  # ngưỡng scout (int)
  max_cycles,             # số vòng lặp (int)
  seed=None               # seed ngẫu nhiên (int, tùy chọn)
).run()  # -> ABCResult(x_best, f_best, history["f_best_per_cycle"])
Gợi ý tham số
sn: 10–50 cho D vừa; D lớn → tăng sn.

limit: 20–200; nhỏ quá dễ “đốt” nguồn, lớn quá lãng phí vòng lặp.

max_cycles: đặt theo thời gian/độ chính xác mong muốn (vd. 100–1000+).
```

##Ghi chú thực hành (rất quan trọng)
Greedy tăng tốc hội tụ nhưng có thể giảm đa dạng; Scout và ngẫu nhiên trong φ giúp tránh kẹt sớm.
Mỗi lần cập nhật một chiều (đúng bản gốc). Biến thể có thể cập nhật nhiều chiều để tăng tốc (đánh đổi ổn định).
Luôn xử lý biên sau khi tạo lân cận; chuẩn hóa xác suất roulette bằng cách cộng ε nhỏ để tránh chia 0.

