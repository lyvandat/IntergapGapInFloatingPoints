"""
=============================================================================
DEMO 2: INTEGER GAPS IN FLOATING-POINT NUMBERS - MACHINE LEARNING
=============================================================================
Demonstrates how integer gaps in IEEE 754 floating-point representation
affect neural network training: gradient vanishing/exploding, FP16 vs FP32.

Principle:
- FP16: 10-bit mantissa -> exact up to 2^11 = 2,048 (beyond this -> lost integers)
- FP32: 23-bit mantissa -> exact up to 2^24 = 16,777,216
- Very small gradients  -> underflow to 0 (vanishing)
- Very large gradients  -> overflow to Inf/NaN (exploding)

How to run:
    cd machine_learning
    python demo_machine_learning.py

Output: 4 PNG files created in the output/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

# ===========================================================================
# PART 4: Training loop simulation - FP16 vs FP32
# ===========================================================================
def demo_training_simulation():
    """
    Simple training simulation: optimize f(x) = (x - target)^2.
    Compares results when using FP16 vs FP32.
    """
    print("\n" + "=" * 70)
    print("PHẦN 4: MÔ PHỎNG TRAINING - FP16 vs FP32")
    print("=" * 70)

    target = 3.14159
    lr = 0.01
    n_steps = 500

    # --- FP32 Training ---
    x_fp32 = np.float32(0.0)
    history_fp32 = [float(x_fp32)]
    loss_fp32 = []
    for step in range(n_steps):
        loss = np.float32((x_fp32 - np.float32(target)) ** 2)
        grad = np.float32(2.0) * (x_fp32 - np.float32(target))
        x_fp32 = np.float32(x_fp32 - np.float32(lr) * grad)
        history_fp32.append(float(x_fp32))
        loss_fp32.append(float(loss))

    # --- FP16 Training ---
    x_fp16 = np.float16(0.0)
    history_fp16 = [float(x_fp16)]
    loss_fp16 = []
    for step in range(n_steps):
        loss = np.float16((x_fp16 - np.float16(target)) ** 2)
        grad = np.float16(2.0) * (x_fp16 - np.float16(target))
        x_fp16 = np.float16(x_fp16 - np.float16(lr) * grad)
        history_fp16.append(float(x_fp16))
        loss_fp16.append(float(loss))

    # --- FP16 with large target (inside the gap zone) ---
    target_large = 2100.0  # FP16 gap = 2 here
    x_fp16_large = np.float16(2000.0)
    history_fp16_large = [float(x_fp16_large)]
    loss_fp16_large = []
    for step in range(n_steps):
        loss = np.float16((x_fp16_large - np.float16(target_large)) ** 2)
        grad = np.float16(2.0) * (x_fp16_large - np.float16(target_large))
        x_fp16_large = np.float16(x_fp16_large - np.float16(lr) * grad)
        history_fp16_large.append(float(x_fp16_large))
        loss_fp16_large.append(float(loss))

    x_fp32_large = np.float32(2000.0)
    history_fp32_large = [float(x_fp32_large)]
    loss_fp32_large = []
    for step in range(n_steps):
        loss = np.float32((x_fp32_large - np.float32(target_large)) ** 2)
        grad = np.float32(2.0) * (x_fp32_large - np.float32(target_large))
        x_fp32_large = np.float32(x_fp32_large - np.float32(lr) * grad)
        history_fp32_large.append(float(x_fp32_large))
        loss_fp32_large.append(float(loss))

    # Print results
    print(f"\n  Bài toán: Tối ưu f(x) = (x - target)² bằng Gradient Descent")
    print(f"  Learning rate = {lr}, Steps = {n_steps}\n")

    print(f"  --- Target = {target} (vùng an toàn cho FP16) ---")
    print(f"  FP32 kết quả: x = {history_fp32[-1]:.6f} (sai {abs(history_fp32[-1]-target):.6e})")
    print(f"  FP16 kết quả: x = {history_fp16[-1]:.6f} (sai {abs(history_fp16[-1]-target):.6e})")

    print(f"\n  --- Target = {target_large} (VÙNG LỖ HỔNG FP16, gap=2) ---")
    print(f"  FP32 kết quả: x = {history_fp32_large[-1]:.6f} "
          f"(sai {abs(history_fp32_large[-1]-target_large):.6e})")
    print(f"  FP16 kết quả: x = {history_fp16_large[-1]:.6f} "
          f"(sai {abs(history_fp16_large[-1]-target_large):.6e})")

    fp16_final = history_fp16_large[-1]
    if abs(fp16_final - target_large) > 0.5:
        print(f"  ⚠️  FP16 KHÔNG THỂ hội tụ chính xác! Gần nhất FP16 biểu diễn được = "
              f"{float(np.float16(target_large))}")

    # --- Plot charts ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training simulation: FP16 vs FP32\n"
                 "FP16 cannot converge inside the integer-gap zone",
                 fontsize=13, fontweight='bold')

    steps = np.arange(n_steps)

    # Loss - small target
    ax = axes[0, 0]
    ax.semilogy(steps, loss_fp32, 'b-', linewidth=1.5, label='FP32', alpha=0.7)
    ax.semilogy(steps, [max(l, 1e-10) for l in loss_fp16], 'r-', linewidth=1.5,
                label='FP16', alpha=0.7)
    ax.set_title(f"Loss curve (target = {target})\nBoth converge well", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trajectory - small target
    ax = axes[0, 1]
    ax.plot(history_fp32[:100], 'b-', linewidth=1.5, label='FP32', alpha=0.7)
    ax.plot(history_fp16[:100], 'r--', linewidth=1.5, label='FP16', alpha=0.7)
    ax.axhline(y=target, color='green', linestyle=':', label=f'Target = {target}')
    ax.set_title(f"x trajectory (target = {target})\nfirst 100 steps", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss - large target (gap zone)
    ax = axes[1, 0]
    ax.semilogy(steps, loss_fp32_large, 'b-', linewidth=1.5, label='FP32', alpha=0.7)
    ax.semilogy(steps, [max(l, 1e-10) for l in loss_fp16_large], 'r-',
                linewidth=1.5, label='FP16', alpha=0.7)
    ax.set_title(f"Loss curve (target = {target_large} - GAP ZONE)\n"
                 f"FP16 loss cannot fall to 0!", fontsize=10, color='red')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trajectory - large target
    ax = axes[1, 1]
    ax.plot(history_fp32_large[:200], 'b-', linewidth=1.5, label='FP32', alpha=0.7)
    ax.plot(history_fp16_large[:200], 'r--', linewidth=1.5, label='FP16', alpha=0.7)
    ax.axhline(y=target_large, color='green', linestyle=':',
               label=f'Target = {target_large}')
    nearest_fp16 = float(np.float16(target_large))
    ax.axhline(y=nearest_fp16, color='red', linestyle=':',
               label=f'Nearest FP16 = {nearest_fp16}')
    ax.set_title(f"x trajectory (target = {target_large})\n"
                 f"FP16 stuck at {nearest_fp16} (gap!)", fontsize=10, color='red')
    ax.set_xlabel("Step")
    ax.set_ylabel("x")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Biểu đồ đã lưu: {path}")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  DEMO: LỖ HỔNG SỐ NGUYÊN TRONG SỐ THỰC - ỨNG DỤNG HỌC MÁY")
    print("█" * 70)

    demo_training_simulation()

    print("\n" + "=" * 70)
    print("KẾT LUẬN:")
    print("=" * 70)
    print("""
  1. FP16 lỗ hổng nguyên bắt đầu tại 2^11 = 2,048: khi target nằm trong
     vùng này (vd: 2,100), FP16 bị kẹt tại giá trị gần nhất biểu diễn
     được và KHÔNG BAO GIỜ hội tụ đúng dù chạy thêm bao nhiêu bước.

  2. FP32 không bị ảnh hưởng ở vùng này (ngưỡng 2^24 = 16,777,216),
     vẫn hội tụ chính xác đến target ngay cả khi target = 2,100.

  3. Kết luận thực tiễn: khi huấn luyện mạng neural với FP16, các trọng
     số và gradient nằm trong vùng lỗ hổng bị lượng tử hóa thô, gây ra
     sai số không thể loại bỏ chỉ bằng cách chạy thêm epoch.
    """)
    print(f"  Tất cả biểu đồ đã lưu trong: {output_dir}/")
    print("=" * 70)
