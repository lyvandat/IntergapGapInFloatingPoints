"""
=============================================================================
DEMO 3: INTEGER GAPS IN FLOATING-POINT NUMBERS - NAVIGATION & ROBOTICS
=============================================================================
Demonstrates accumulated errors from integer gaps in real-time systems:
- Patriot Missile bug (1991): time variable loses precision after many hours
- GPS drift: coordinate error due to floating-point precision
- Kalman Filter: large time variable freezes the system

How to run:
    cd navigation_robotics
    python demo_navigation_robotics.py

Output: 3 PNG files created in the output/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)


# ===========================================================================
# PART 1: Patriot Missile Bug (1991) - reproduction
# ===========================================================================
def demo_patriot_missile():
    """
    Reproduces the Patriot Missile bug (Feb 25, 1991, Dhahran, Saudi Arabia).

    Root cause: The Patriot ran on an Intel 8086. Time was tracked as a
    24-BIT SIGNED FIXED-POINT register counting 0.1-second ticks.

    A 24-bit signed (two's complement) register has:
        1 sign bit + 23 fraction bits  →  denominator = 2^23

    0.1 in binary = 0.000110011001100110011... (infinite repeating).
    Truncated to 23 binary fraction bits (signed 24-bit register):
        floor(0.1 * 2^23) / 2^23 = 838860 / 8388608
                                  = 0.09999990463256836...

    Error per tick = 0.1 - 0.09999990463... ≈ 9.54 × 10^-8 seconds.
    After 3,600,000 ticks (100 hours): error ≈ 0.343 seconds ← matches ESA report.

    NOTE: float32 is shown only for comparison; the real bug used
    a 24-bit SIGNED FIXED-POINT integer register, NOT a 32-bit float.
    float32 stores 0.1 slightly ABOVE 0.1, while the 24-bit truncation
    stores it slightly BELOW — opposite sign, different magnitude.
    """
    print("=" * 70)
    print("PHẦN 1: TÁI HIỆN LỖI PATRIOT MISSILE (1991)")
    print("=" * 70)

    dt_true = 0.1   # exact 0.1 seconds

    # === CORRECT: 24-bit SIGNED fixed-point (actual Patriot hardware) ===
    # A 24-bit two's complement register = 1 sign bit + 23 fraction bits.
    # Denominator is therefore 2^23, not 2^24.
    #   floor(0.1 * 2^23) = floor(838860.8) = 838860 counts
    #   838860 / 2^23 = 0.09999990463...
    FRACTION_BITS = 23                                     # signed 24-bit -> 23 fraction bits
    dt_24bit_counts = int(dt_true * (2 ** FRACTION_BITS))  # 838860
    dt_24bit = dt_24bit_counts / (2 ** FRACTION_BITS)      # 0.09999990463...
    error_per_tick_24bit = dt_true - dt_24bit              # ~9.537e-8 s (matches ESA/Wikipedia)

    # === FOR COMPARISON ONLY: how float32 stores 0.1 ===
    # float32 rounds 0.1 UP (opposite direction from 24-bit truncation)
    dt_fp32 = float(np.float32(0.1))                      # 0.10000000149...
    error_per_tick_fp32 = dt_fp32 - dt_true               # ~+1.49e-9 s (positive!)

    # === FOR COMPARISON ONLY: float64 ===
    dt_fp64 = float(np.float64(0.1))                      # 0.10000000000000000555...
    error_per_tick_fp64 = dt_fp64 - dt_true               # ~+5.55e-18 s

    print(f"\n--- Biểu diễn nhị phân của 0.1 với các độ chính xác khác nhau ---")
    print(f"  Giá trị đúng (thập phân):       0.10000000000000000000...")
    print(f"  24-bit SIGNED fixed (Patriot):  {dt_24bit:.20f}  "
          f"[sai {error_per_tick_24bit:+.4e} s/tick, ÂM]"
          f"\n    (= {dt_24bit_counts}/2^{FRACTION_BITS}  — 1 sign bit + 23 fraction bits)")
    print(f"  float32     (so sánh):       {dt_fp32:.20f}  "
          f"[sai {error_per_tick_fp32:+.4e} s/tick, DƯƠNG]")
    print(f"  float64     (so sánh):       {dt_fp64:.20f}  "
          f"[sai {error_per_tick_fp64:+.4e} s/tick]")
    print(f"\n  ⚠️  24-bit TRUNCATION (floor) → THIẾU mỗi tick")
    print(f"  ⚠️  float32  ROUNDING (round) → THỪA mỗi tick (dấu ngược!)")
    print(f"  → Patriot dùng 24-bit fixed, KHÔNG phải float32!")

    hours_to_test = [0.1, 1, 8, 20, 50, 72, 100]
    scud_speed = 1676.0  # m/s

    print(f"\n--- Sai số tích lũy theo thời gian (dùng 24-bit fixed-point) ---")
    print(f"  Sai số mỗi tick (24-bit) = {error_per_tick_24bit:.4e} giây")
    print(f"  Công thức: error(n_ticks) = n_ticks × {error_per_tick_24bit:.4e}\n")
    print(f"  {'Giờ':>6s} | {'Số tick':>12s} | {'Sai số (s)':>12s} | "
          f"{'Lệch vị trí (m)':>16s} | {'float32 sai số (s)':>18s}")
    print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*16} | {'-'*18}")

    results_24bit = []
    results_fp32 = []
    for hours in hours_to_test:
        n_ticks = int(hours * 3600 / dt_true)

        # 24-bit fixed-point: error grows linearly (each tick loses same amount)
        # Patriot clock reads: n_ticks * dt_24bit  (lags behind real time)
        # True time:           n_ticks * dt_true
        error_24bit = n_ticks * error_per_tick_24bit   # positive = Patriot runs slow
        error_meters_24bit = abs(error_24bit) * scud_speed

        # float32 comparison: error grows differently (and in opposite direction)
        # float32 overestimates 0.1, so clock runs FAST
        error_fp32 = n_ticks * error_per_tick_fp32
        results_24bit.append((hours, n_ticks, error_24bit, error_meters_24bit))
        results_fp32.append(error_fp32)

        marker = "  ← THẢM HỌA!" if hours >= 100 else ("  ← NGUY HIỂM!" if error_meters_24bit > 100 else "")
        print(f"  {hours:>6.1f} | {n_ticks:>12,d} | {error_24bit:>+12.6f} | "
              f"{error_meters_24bit:>16.1f} | {error_fp32:>+18.4e}{marker}")

    print(f"\n  Thực tế (ESA report):    sai lệch = 0.3433 s → 573 m tại 100h")
    print(f"  Mô phỏng 24-bit:         sai lệch = "
          f"{results_24bit[-1][2]:.4f} s → {results_24bit[-1][3]:.0f} m ✓")
    print(f"\n  ⚠️  Hệ thống không đánh chặn được tên lửa Scud → 28 binh sĩ Mỹ tử trận")

    # --- Plot charts ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    fig.suptitle("Patriot Missile bug (1991)  —  24-bit fixed-point truncation of 0.1\n"
                 "float32 shown for comparison only: it runs FAST (opposite sign!)",
                 fontsize=12, fontweight='bold')

    hours_arr       = [r[0] for r in results_24bit]
    errors_s_24bit  = [r[2] for r in results_24bit]   # positive (clock lags)
    errors_m_24bit  = [r[3] for r in results_24bit]
    errors_s_fp32   = results_fp32

    # Subplot 1: time error comparison — 24-bit vs float32
    ax = axes[0]
    ax.plot(hours_arr, errors_s_24bit, 'ro-', linewidth=2, markersize=7,
            label='24-bit fixed (Patriot, actual)')
    ax.plot(hours_arr, errors_s_fp32,  'b^--', linewidth=1.5, markersize=6,
            label='float32 (comparison, opposite sign)')
    ax.axhline(y=0.3433, color='red', ls='--', alpha=0.6,
               label='ESA report: +0.34 s (clock too slow)')
    ax.axhline(y=0, color='gray', ls=':', lw=1)
    ax.set_xlabel("Operating time (hours)")
    ax.set_ylabel("Clock error (seconds)\n+ = runs slow,  − = runs fast")
    ax.set_title("Accumulated clock error\n"
                 "24-bit truncation (−) vs float32 rounding (+): opposite sides!", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 2: position error (24-bit, historical)
    ax = axes[1]
    colors = ['green' if m < 100 else 'orange' if m < 500 else 'red'
              for m in errors_m_24bit]
    ax.bar(range(len(hours_arr)), errors_m_24bit, color=colors)
    ax.set_xticks(range(len(hours_arr)))
    ax.set_xticklabels([f'{h}h' for h in hours_arr])
    ax.set_ylabel("Position error (meters)")
    ax.set_title(f"Scud position error  [{scud_speed} m/s]\n"
                 f"24-bit clock error × {scud_speed} m/s", fontsize=10)
    ax.axhline(y=573, color='red', ls='--', label='ESA actual (~573 m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (_, m) in enumerate(zip(hours_arr, errors_m_24bit)):
        ax.text(i, m + 5, f'{m:.0f}m', ha='center', fontsize=7)

    # Subplot 4: summary table
    ax = axes[2]
    ax.axis('off')
    summary = f"""
  PATRIOT MISSILE — CORRECT ANALYSIS

  Hardware:  Intel 8086
  Register:  24-bit fixed-point
  Tick:      every 0.1 seconds

  ┌─────────────────────────────────────────────────┐
  │  0.1 in binary = 0.000110011001100110011...     │
  │  Repeats forever — must be cut off at 24 bits   │
  │                                                  │
  │  24-bit value = 1677721 / 2^24                  │
  │               = 0.099999904632...               │
  │               < 0.1  (floor / truncation)       │
  │                                                  │
  │  Error / tick = 0.1 − 0.09999990... = 9.54e-8 s│
  │  × 3,600,000 ticks (100 h) = 0.343 s ✓ (ESA)  │
  └─────────────────────────────────────────────────┘

  WHY NOT float32?
    float32 rounds 0.1 UP to 0.1000000015...
    → clock runs FAST (opposite error direction)
    → different magnitude: only 0.005 s after 100 h
    → float32 was NOT the actual Patriot bug

  LESSON: the 24-bit FIXED-POINT TRUNCATION of 0.1
  is the specific representation error that matches
  the historical 0.34 s / 573 m failure.
"""
    ax.text(0.03, 0.97, summary, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fff8dc', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "patriot_missile.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Biểu đồ đã lưu: {path}")





# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  DEMO: LỖ HỔNG SỐ NGUYÊN TRONG SỐ THỰC - ĐỊNH VỊ & ĐIỀU KHIỂN")
    print("█" * 70)

    demo_patriot_missile()

    print("\n" + "=" * 70)
    print("KẾT LUẬN:")
    print("=" * 70)
    print("""
  1. Lỗi cốt lõi KHÔNG phải float32: hệ thống Patriot dùng thanh ghi
     24-bit SIGNED fixed-point (1 bit dấu + 23 bit phần lẻ).
     Phân số 0.1 bị TRUNCATE (floor) tại bit 23:
       838860 / 2^23 = 0.09999990... < 0.1
     → đồng hồ chạy CHẬM hơn thực tế.

  2. float32 KHÔNG tái hiện đúng bug: float32 ROUND UP 0.1 thành
     0.10000000149... → đồng hồ chạy NHANH (dấu ngược!), sai số
     chỉ 0.005s sau 100h, hoàn toàn khác với 0.34s thực tế.

  3. Sai số tích lũy tuyến tính:
     9.54 × 10⁻⁸ s/tick × 3,600,000 tick = 0.343 s → 575 m
     khớp chính xác báo cáo ESA (0.343 s / 573 m).

  4. Bài học: khi chuyển đổi số thực sang biểu diễn hữu hạn (dù là
     fixed-point hay float), luôn phải kiểm tra chiều và độ lớn của
     sai số tích lũy — đặc biệt trong hệ thống chạy liên tục dài hạn.
    """)
    print(f"  Tất cả biểu đồ đã lưu trong: {output_dir}/")
    print("=" * 70)
