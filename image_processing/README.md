# Image Processing — Integer Gaps in Floating-Point Numbers

Demonstrates two concrete image artifacts caused by floating-point integer
gaps, using real photographs from `scipy.datasets`.

| Photo | Source | Size |
|-------|--------|------|
| Raccoon face | `scipy.datasets.face()` | 1024×768 RGB |
| Staircase | `scipy.datasets.ascent()` | 512×512 grayscale |

## Output

One file: `output/float_gap_artifacts.png`

![Float gap artifacts](output/float_gap_artifacts.png)

## What it shows

**Row 1 — HDR Banding (raccoon face)**

Pixel values are shifted into the float16 integer-gap zone
`[32768, 65504)` where the gap between consecutive representable values is **32**.
Every 32 original intensity levels collapse to one float16 value.

```
pixel = 0   →  32768  →  float16(32768) = 32768  →  restored = 0
pixel = 1   →  32769  →  float16(32769) = 32768  →  restored = 0   ← same!
...
pixel = 31  →  32799  →  float16(32799) = 32768  →  restored = 0   ← same!
pixel = 32  →  32800  →  float16(32800) = 32800  →  restored = 32
```

**Result:** 253 tonal levels → 9 levels (96% of tonal information lost).
Smooth gradients in fur and background become obvious colour bands.

---

**Row 2 — Coordinate Shift (staircase photo)**

In satellite/GIS imagery, pixel coordinates carry a large global atlas offset.
Beyond `2^24 = 16,777,216`, float32 gap = 2, so consecutive integers collide:

```
float32(16_777_216 + 1)  →  16_777_216   (rounds down → column duplicated)
float32(16_777_216 + 3)  →  16_777_220   (skips column 3)
```

**Result:** 255 / 512 columns (50%) mapped to the wrong source.
Visible as periodic vertical stripe artifacts.

## Requirements

- Python 3.8+
- numpy, matplotlib, scipy, pooch, pillow

## Setup & Run

### 1. Create virtual environment

```bash
# Windows
python -m venv .venv

# Linux / macOS
python3 -m venv .venv
```

### 2. Activate

```bash
# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, scipy downloads the test images (~600 KB) from GitHub
> into a local cache (`~/.local/share/scipy-data/`).

### 4. Run

```bash
PYTHONIOENCODING=utf-8 python demo_image_artifacts_real.py
```

### 5. Deactivate

```bash
deactivate
```

## Folder Structure

```
image_processing/
├── output/
│   └── float_gap_artifacts.png   # Combined output (HDR banding + coordinate shift)
├── demo_image_artifacts_real.py  # Main demo script
├── requirements.txt
├── README.md
└── .gitignore
```
