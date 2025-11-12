# ABC-Base (Artificial Bee Colony) — Tài liệu dự án

Triển khai gọn nhẹ **thuật toán Bầy Ong Nhân Tạo (ABC)** cho bài toán **tối thiểu hóa liên tục** với ràng buộc hộp.

## API
API rút gọn — mục đích: “cheat-sheet” cách gọi class và đọc kết quả

Dùng để nhìn nhanh chữ ký hàm, đối số cần truyền và dữ liệu trả về — không cần mở mã nguồn.

ArtificialBeeColony(
  obj_func,               # Callable[[np.ndarray], float] — hàm mục tiêu (tối thiểu hóa)
  bounds=(lower, upper),  # (np.ndarray, np.ndarray), mỗi cái shape (D,)
  sn,                     # số nguồn thức ăn (int)
  limit,                  # ngưỡng scout (int)
  max_cycles,             # số vòng lặp (int)
  seed=None               # seed ngẫu nhiên (int, tùy chọn)
).run()  # -> ABCResult(x_best, f_best, history["f_best_per_cycle"])

