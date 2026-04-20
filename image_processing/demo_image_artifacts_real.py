"""
Floating-point integer gaps — real image artifact demo.

Two effects, one output image (output/float_gap_artifacts.png):

  ROW 1  HDR BANDING (raccoon face)
    float16 gap = 16 in range [16384, 32768)
    256 tonal levels → 16 levels: smooth gradients become obvious colour bands.

  ROW 2  COORDINATE SHIFT (staircase photo)
    float32 gap = 2 at offset 2^24 = 16,777,216
    50 % of pixel columns mapped to the wrong source → vertical stripe artifacts.

How to run:
    python demo_image_artifacts_real.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.datasets import face, ascent
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# EFFECT 1 — HDR BANDING
# =============================================================================
def apply_hdr_banding(img_rgb: np.ndarray, hdr_offset: float = 16384.0):
    """
    Simulate one round-trip through a float16 HDR pipeline.

    Pixel values (0–255) are shifted into the float16 integer-gap zone
    [16384, 32768), where consecutive representable values are 16 apart.
    Every 16 original intensity levels collapse to the same float16 value,
    turning smooth gradients into coarse steps.
    """
    def quantise(channel):
        scaled   = channel.astype(np.float64) + hdr_offset   # shift into gap zone
        f16      = scaled.astype(np.float16)                  # gap = 16 here
        restored = np.clip(f16.astype(np.float64) - hdr_offset, 0, 255)
        return restored.astype(np.uint8)

    banded = np.stack([quantise(img_rgb[:, :, c]) for c in range(3)], axis=2)

    # Count distinct levels per channel
    levels_in  = [len(np.unique(img_rgb[:, :, c])) for c in range(3)]
    levels_out = [len(np.unique(banded[:, :, c]))   for c in range(3)]
    return banded, levels_in, levels_out


# =============================================================================
# EFFECT 2 — COORDINATE SHIFT
# =============================================================================
def apply_coord_artifact(img_gray: np.ndarray, offset_exp: int = 24):
    """
    Simulate satellite coordinate mapping through float32.

    Each pixel's x-coordinate is: local_x = float32(global_x) - float32(OFFSET)
    Beyond 2^offset_exp, float32 gap = 2, so consecutive integers collide:
        float32(OFFSET + 1) rounds to OFFSET  → pixel 1 reads from pixel 0
        float32(OFFSET + 3) rounds to OFFSET+4 → pixel 3 reads from pixel 4
    50 % of columns land on the wrong source, creating periodic stripes .
    """
    OFFSET = np.float32(2 ** offset_exp)
    h, w   = img_gray.shape

    mapping = np.array([
        int(round(float(np.float32(x) + OFFSET) - float(OFFSET)))
        for x in range(w)
    ], dtype=np.int32)
    mapping = np.clip(mapping, 0, w - 1)

    corrupted  = img_gray[:, mapping]
    error_cols = mapping != np.arange(w)
    n_wrong    = int(error_cols.sum())
    return corrupted, error_cols, n_wrong


# =============================================================================
# COMBINED FIGURE
# =============================================================================
def make_figure():
    img_face   = face()    # (768, 1024, 3) uint8
    img_stairs = ascent()  # (512, 512)     uint8

    # gap=32 → 256 input levels → ~8 output levels (highly visible posterization)
    banded, lvl_in, lvl_out = apply_hdr_banding(img_face, hdr_offset=32768.0)
    corrupted, err_cols, n_wrong = apply_coord_artifact(img_stairs)

    # -- zoom into smooth background for banding: raccoon upper-right (leaves) --
    CROP_R = slice(40, 310)
    CROP_C = slice(680, 1024)
    crop_orig   = img_face[CROP_R, CROP_C]
    crop_banded = banded[CROP_R, CROP_C]

    # -- highlight wrong columns on the staircase --
    stairs_rgb = np.stack([img_stairs]*3, axis=2).copy()
    stairs_err = stairs_rgb.copy()
    stairs_err[:, err_cols] = ( 
        stairs_err[:, err_cols] * 0.3
        + np.array([220, 30, 30])
    ).clip(0, 255).astype(np.uint8)

    # ── layout: 2 rows × 3 cols ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    fig.patch.set_facecolor("#0d1117")

    TITLE_KW   = dict(color="white",  fontsize=11, fontweight="bold", pad=5)
    LABEL_KW   = dict(color="#aaaaaa", fontsize=9,  pad=3)
    EFFECT1_C  = "#ff6a00"   # orange for banding
    EFFECT2_C  = "#ff1744"   # red for coordinate error

    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # ── ROW 0: HDR Banding ────────────────────────────────────────────────────
    axes[0, 0].imshow(img_face)
    axes[0, 0].set_title(
        f"EFFECT 1 — HDR Banding  [float16 gap = 32]\n"
        f"Original  ({lvl_in[1]} tonal levels)",
        **TITLE_KW)
    clean(axes[0, 0])

    axes[0, 1].imshow(banded)
    axes[0, 1].set_title(
        f"After float16 HDR processing  ({lvl_out[1]} levels)",
        color=EFFECT1_C, fontsize=11, fontweight="bold", pad=5)
    clean(axes[0, 1])

    # Zoom: place original and banded crops side-by-side in one panel
    gap = np.full((crop_orig.shape[0], 4, 3), 50, dtype=np.uint8)
    zoom_pair = np.hstack([crop_orig, gap, crop_banded])
    axes[0, 2].imshow(zoom_pair)
    axes[0, 2].set_title("Background zoom: smooth → banded",
                           color=EFFECT1_C, fontsize=11, fontweight="bold", pad=5)
    # label left/right halves
    mid = zoom_pair.shape[1] // 2
    axes[0, 2].text(mid * 0.5, zoom_pair.shape[0] - 12, "original",
                     color="white", fontsize=8.5, ha="center",
                     fontweight="bold")
    axes[0, 2].text(mid * 1.5, zoom_pair.shape[0] - 12, "float16",
                     color=EFFECT1_C, fontsize=8.5, ha="center",
                     fontweight="bold")
    axes[0, 2].axvline(x=crop_orig.shape[1] + 2, color="#888", lw=1.5)
    clean(axes[0, 2])

    # ── ROW 1: Coordinate Artifact ────────────────────────────────────────────
    axes[1, 0].imshow(img_stairs, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title(
        f"EFFECT 2 — Coordinate Shift  [float32 gap = 2 at 2²⁴]\n"
        f"Original  (all {img_stairs.shape[1]} columns correct)",
        **TITLE_KW)
    clean(axes[1, 0])

    axes[1, 1].imshow(corrupted, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title(
        f"After float32 coordinate mapping  ({n_wrong} / {img_stairs.shape[1]} wrong)",
        color=EFFECT2_C, fontsize=11, fontweight="bold", pad=5)
    clean(axes[1, 1])

    # Zoom: show a 200×200 crop with error columns highlighted
    cz = slice(100, 380)
    zoom_clean = np.stack([img_stairs[cz, cz]]*3, axis=2)
    zoom_bad   = stairs_err[cz, cz]
    g2 = np.full((zoom_clean.shape[0], 4, 3), 50, dtype=np.uint8)
    coord_pair = np.hstack([zoom_clean, g2, zoom_bad])
    axes[1, 2].imshow(coord_pair)
    axes[1, 2].set_title("Zoom: original vs corrupted (red = wrong col)",
                           color=EFFECT2_C, fontsize=11, fontweight="bold", pad=5)
    mid2 = coord_pair.shape[1] // 2
    axes[1, 2].text(mid2 * 0.5, coord_pair.shape[0] - 12, "original",
                     color="white", fontsize=8.5, ha="center",
                     fontweight="bold")
    axes[1, 2].text(mid2 * 1.5, coord_pair.shape[0] - 12, "float32",
                     color=EFFECT2_C, fontsize=8.5, ha="center",
                     fontweight="bold")
    axes[1, 2].axvline(x=zoom_clean.shape[1] + 2, color="#888", lw=1.5)
    clean(axes[1, 2])

    # ── Main title ─────────────────────────────────────────────────────────────
    fig.suptitle("Floating-Point Integer Gaps — Visible Image Artifacts\n"
                 "Real photographs from scipy.datasets  (raccoon face + staircase)",
                  color="white", fontsize=13, fontweight="bold", y=1.01)

    # ── Save ───────────────────────────────────────────────────────────────────
    out = os.path.join(output_dir, "float_gap_artifacts.png")
    fig.savefig(out, dpi=140, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    print(f"  Banding:  {lvl_in[1]} levels → {lvl_out[1]}  "
          f"({100-lvl_out[1]/lvl_in[1]*100:.0f}% tonal info lost)")
    print(f"  Coord:    {n_wrong}/{img_stairs.shape[1]} columns wrong "
          f"({n_wrong/img_stairs.shape[1]*100:.0f}%)")


if __name__ == "__main__":
    make_figure()
