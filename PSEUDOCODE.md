# Thuật toán ABC (Bầy Ong Nhân Tạo) — Mã giả 

## Đầu vào / Đầu ra / Ký hiệu
- **Đầu vào**
  - `f(x)`: hàm mục tiêu (mặc định **tối thiểu hóa**). *[Điền tên hàm của bạn nếu muốn]*  
  - `bounds = [x_min, x_max]`: miền ràng buộc theo từng chiều. *[Điền miền cụ thể nếu có]*  
  - `D`: số chiều; `SN`: số nguồn thức ăn.  
  - `limit`: số lần cải thiện thất bại liên tiếp trước khi chuyển thành scout.  
  - `maxCycle`: số vòng lặp.
- **Biến**
  - `x_i ∈ ℝ^D`: nghiệm (nguồn) thứ i; `trial_i`: bộ đếm thất bại của i.  
  - `φ_ij ~ U[-1,1]`: hệ số nhiễu khi tạo lân cận tại chiều j của nguồn i.
- **Đầu ra**
  - `x*`: nghiệm tốt nhất, `f(x*)`: giá trị mục tiêu tốt nhất.

---

## Thủ tục phụ (hàm trợ giúp)
- **Fitness (cho tối thiểu hóa)**  
  Nếu `f(x) ≥ 0`: `fit = 1/(1 + f(x))`; ngược lại: `fit = 1 + |f(x)|`.  
- **Xử lý biên (boundary)**  
  Cắt từng chiều về `[x_min[j], x_max[j]]` (clipping). *[Có thể đổi sang reflection nếu bạn áp dụng]*  
- **Roulette selection**  
  Chọn chỉ số i theo xác suất `p_i = fit_i / Σ fit`.

---

## Mã giả
FUNCTION ABC(f, bounds=[x_min, x_max], D, SN, limit, maxCycle):

    # 1) Khởi tạo
    for i = 1..SN:
        x[i]    ← Uniform(x_min, x_max) in ℝ^D
        fit[i]  ← fitness_from_objective( f(x[i]) )
        trial[i]← 0
    x_best ← argmin_i f(x[i])

    for cycle = 1..maxCycle:

        # 2) Pha Employed
        for i = 1..SN:
            k ← random in {1..SN} \ {i}
            j ← random in {1..D}
            φ ← Uniform(-1, 1)
            v ← x[i]
            v[j] ← x[i][j] + φ * (x[i][j] - x[k][j])
            v ← boundary_clip(v, bounds)
            if f(v) ≤ f(x[i]):         # Greedy selection
                x[i]   ← v
                fit[i] ← fitness_from_objective( f(x[i]) )
                trial[i] ← 0
            else:
                trial[i] ← trial[i] + 1

        # 3) Pha Onlooker
        p ← normalize( fit )            # p_i = fit[i] / Σ fit
        count ← 0
        while count < SN:
            i ← roulette_select(p)
            k ← random in {1..SN} \ {i}
            j ← random in {1..D}
            φ ← Uniform(-1, 1)
            v ← x[i]
            v[j] ← x[i][j] + φ * (x[i][j] - x[k][j])
            v ← boundary_clip(v, bounds)
            if f(v) ≤ f(x[i]):
                x[i]   ← v
                fit[i] ← fitness_from_objective( f(x[i]) )
                trial[i] ← 0
            else:
                trial[i] ← trial[i] + 1
            count ← count + 1

        # 4) Pha Scout
        for i = 1..SN:
            if trial[i] ≥ limit:
                x[i]    ← Uniform(x_min, x_max) in ℝ^D
                fit[i]  ← fitness_from_objective( f(x[i]) )
                trial[i]← 0

        # 5) Cập nhật tốt nhất
        if min_i f(x[i]) < f(x_best):
            x_best ← argmin_i f(x[i])

    RETURN x_best, f(x_best)
