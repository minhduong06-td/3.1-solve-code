import time
import numpy as np
import matplotlib.pyplot as plt
from abc_base import ArtificialBeeColony 
from variants.abc_gbest import GbestABC

def steering_vector(theta_rad: float, N: int, d_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(N)
    phase = 2.0 * np.pi * d_lambda * n * np.sin(theta_rad)
    return np.exp(1j * phase)

def make_obj_func(a: np.ndarray, Rn: np.ndarray):
    N = a.size
    def obj_func(x: np.ndarray) -> float:
        w = x[:N] + 1j * x[N:]
        norm = np.linalg.norm(w)
        if norm > 0: w = w / norm
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

def beam_metrics(g_db, angles_deg, theta0_deg):
    idx0 = int(np.argmin(np.abs(angles_deg - theta0_deg)))
    peak_db = float(g_db[idx0])
    left = idx0
    while left > 0 and g_db[left-1] >= peak_db - 3: left -= 1
    right = idx0
    while right < len(g_db)-1 and g_db[right+1] >= peak_db - 3: right += 1
    bw_3db = float(angles_deg[right] - angles_deg[left])
    mask = np.ones_like(g_db, dtype=bool)
    mask[left:right+1] = False
    idx = np.where((g_db[1:-1] > g_db[:-2]) & (g_db[1:-1] > g_db[2:]) & mask[1:-1])[0] + 1
    sll_db = float(g_db[idx].max()) if idx.size else float("nan")
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

    sn, limit, max_cycles, seed = 30, 100, 300, 42
    beta = 0.9

    t0 = time.time()
    abc = GbestABC(
        obj_func=obj,
        bounds=(lower, upper),
        sn=sn,
        limit=limit,
        max_cycles=max_cycles,
        seed=seed,
        beta=beta,
    )
    res = abc.run()
    t1 = time.time()

    w_best = res.x_best[:N] + 1j * res.x_best[N:]
    w_best = w_best / (np.linalg.norm(w_best) + 1e-12)
    snr_best = -res.f_best
    print("=== GABC Beamforming Demo ===")
    print(f"Best objective (−SNR): {res.f_best:.6f}")
    print(f"Approx SNR (maximized): {snr_best:.6f}")
    print(f"Time: {t1-t0:.3f} s")

    a_norm = a / (np.linalg.norm(a) + 1e-12)
    alignment = np.abs(np.vdot(a_norm, w_best))
    print(f"Alignment MRT: {alignment:.4f}")

    snr_hist = -res.history["f_best_per_cycle"]
    snr_hist_db = 10*np.log10(np.maximum(snr_hist, 1e-12))

    target = 0.95 * 8.0
    ge = np.where(snr_hist >= target)[0]
    cycles_to_95 = int(ge[0]) if ge.size else -1

    plt.figure(); plt.plot(snr_hist)
    plt.xlabel("Chu kỳ"); plt.ylabel("SNR (tỉ lệ)")
    plt.title("GABC hội tụ (beamforming)"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("gabc_convergence.png", dpi=300); plt.close()

    plt.figure(); plt.plot(snr_hist_db)
    plt.xlabel("Chu kỳ"); plt.ylabel("SNR (dB)")
    plt.title("GABC hội tụ — thang dB"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("gabc_convergence_db.png", dpi=300); plt.close()

    angles_deg = np.linspace(-90, 90, 1081)
    g_best_db = array_factor_db(w_best, angles_deg)
    w_mrt = a_norm
    g_mrt_db = array_factor_db(w_mrt, angles_deg)

    plt.figure()
    plt.plot(angles_deg, g_best_db, label="GABC tối ưu")
    plt.plot(angles_deg, g_mrt_db, "--", label="MRT (chuẩn)")
    plt.axvline(theta0_deg, color="k", linestyle=":", label=f"Hướng {theta0_deg:.1f}°")
    plt.xlim(-90, 90); plt.ylim(-60, 0)
    plt.xlabel("Góc (độ)"); plt.ylabel("Độ lợi (dB, chuẩn hóa)")
    plt.title("Dạng bức xạ ULA — GABC vs MRT")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("gabc_beam_pattern.png", dpi=300); plt.close()

    metrics = beam_metrics(g_best_db, angles_deg, theta0_deg)
    with open("gabc_metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== GABC Beamforming Metrics ===\n")
        f.write(f"SNR_best (linear): {snr_best:.6f}\n")
        f.write(f"SNR_best (dB):     {10*np.log10(max(snr_best,1e-12)):.3f} dB\n")
        f.write(f"cycles_to_95:      {cycles_to_95}\n")
        f.write(f"Alignment MRT:     {alignment:.4f}\n")
        f.write(f"Beamwidth (-3 dB): {metrics['bw_3db']:.2f} deg\n")
        f.write(f"SLL:               {metrics['sll_db']:.2f} dB\n")
    print("Saved: gabc_convergence*.png, gabc_beam_pattern.png, gabc_metrics.txt")
