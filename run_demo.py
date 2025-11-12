import numpy as np
import matplotlib.pyplot as plt
from abc_base import ArtificialBeeColony  

def steering_vector(theta_rad: float, N: int, d_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(N)
    phase = 2.0 * np.pi * d_lambda * n * np.sin(theta_rad)
    return np.exp(1j * phase)  

def make_obj_func(a: np.ndarray, Rn: np.ndarray):
    N = a.size
    def obj_func(x: np.ndarray) -> float:
        w = x[:N] + 1j * x[N:]
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        num = np.abs(np.vdot(a, w)) ** 2
        den = np.real(np.vdot(w, Rn @ w)) + 1e-12
        return -float(num / den)  
    return obj_func

def array_factor_db(w: np.ndarray, angles_deg: np.ndarray, d_lambda: float = 0.5) -> np.ndarray:
    N = w.size
    n = np.arange(N)
    th = np.deg2rad(angles_deg)
    af = np.exp(1j * (2*np.pi*d_lambda*np.outer(n, np.sin(th))))  
    gain = np.abs(np.conj(af).T @ w)                              
    gain = gain / (gain.max() + 1e-12)
    return 20*np.log10(gain + 1e-12)

def beam_metrics(g_db: np.ndarray, angles_deg: np.ndarray, theta0_deg: float):
    idx0 = int(np.argmin(np.abs(angles_deg - theta0_deg)))
    peak_db = float(g_db[idx0])

    left = idx0
    while left > 0 and g_db[left-1] >= peak_db - 3: 
        left -= 1
    right = idx0
    while right < len(g_db)-1 and g_db[right+1] >= peak_db - 3:
        right += 1
    bw_3db = float(angles_deg[right] - angles_deg[left])

    mask = np.ones_like(g_db, dtype=bool)
    mask[left:right+1] = False 
    candidates = np.where(
        (g_db[1:-1] > g_db[:-2]) & (g_db[1:-1] > g_db[2:]) & mask[1:-1]
    )[0] + 1
    sll_db = float(g_db[candidates].max()) if candidates.size else float("nan")

    return dict(peak_db=peak_db, bw_3db=bw_3db, sll_db=sll_db)


if __name__ == "__main__":
    N = 8
    theta0_deg = 20.0
    theta0 = np.deg2rad(theta0_deg)
    a = steering_vector(theta0, N)
    Rn = np.eye(N, dtype=np.complex128)

    D = 2 * N
    lower = -1.0 * np.ones(D)
    upper =  1.0 * np.ones(D)
    obj = make_obj_func(a, Rn)

    sn = 30
    limit = 100
    max_cycles = 300
    seed = 42

    abc = ArtificialBeeColony(
        obj_func=obj,
        bounds=(lower, upper),
        sn=sn,
        limit=limit,
        max_cycles=max_cycles,
        seed=seed,
    )
    res = abc.run()

    print("=== ABC Beamforming Demo ===")
    print(f"Best objective (−SNR): {res.f_best:.6f}")
    print(f"Approx SNR (maximized): {-res.f_best:.6f}")
    w_best = res.x_best[:N] + 1j * res.x_best[N:]
    w_best = w_best / (np.linalg.norm(w_best) + 1e-12)
    print("w_best (first 4 elems):", w_best[:4])
    print("History length:", len(res.history["f_best_per_cycle"]), "cycles")

    snr_hist = -res.history["f_best_per_cycle"]
    snr_hist_db = 10*np.log10(np.maximum(snr_hist, 1e-12))

    plt.figure()
    plt.plot(snr_hist)
    plt.xlabel("Chu kỳ"); plt.ylabel("SNR (tỉ lệ)")
    plt.title("Hội tụ ABC (beamforming)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("abc_convergence.png", dpi=300); plt.close()

    plt.figure()
    plt.plot(snr_hist_db)
    plt.xlabel("Chu kỳ"); plt.ylabel("SNR (dB)")
    plt.title("Hội tụ ABC (beamforming) — thang dB")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("abc_convergence_db.png", dpi=300); plt.close()

    angles_deg = np.linspace(-90, 90, 1081)
    g_best_db = array_factor_db(w_best, angles_deg)
    w_mrt = a / (np.linalg.norm(a) + 1e-12)  
    g_mrt_db = array_factor_db(w_mrt, angles_deg)

    plt.figure()
    plt.plot(angles_deg, g_best_db, label="ABC tối ưu")
    plt.plot(angles_deg, g_mrt_db, "--", label="Chuẩn lý thuyết (MRT)")
    plt.axvline(theta0_deg, color="k", linestyle=":", label=f"Hướng mục tiêu {theta0_deg:.1f}°")
    plt.xlim(-90, 90); plt.ylim(-60, 0)
    plt.xlabel("Góc (độ)"); plt.ylabel("Độ lợi (dB, chuẩn hóa)")
    plt.title("Dạng bức xạ mảng ULA"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("beam_pattern.png", dpi=300); plt.close()

    metrics = beam_metrics(g_best_db, angles_deg, theta0_deg)
    snr_best = -res.f_best
    with open("beam_metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== ABC Beamforming Metrics ===\n")
        f.write(f"SNR_best (linear): {snr_best:.6f}\n")
        f.write(f"SNR_best (dB):     {10*np.log10(max(snr_best,1e-12)):.3f} dB\n")
        f.write(f"Mainlobe peak:      {metrics['peak_db']:.2f} dB\n")
        f.write(f"Beamwidth (-3 dB):  {metrics['bw_3db']:.2f} deg\n")
        f.write(f"Sidelobe level (SLL): {metrics['sll_db']:.2f} dB\n")

    print("\nĐã lưu hình: abc_convergence.png, abc_convergence_db.png, beam_pattern.png")
    print("Đã lưu số liệu: beam_metrics.txt")
