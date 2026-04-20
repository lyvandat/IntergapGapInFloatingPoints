# Ariane 5 Flight 501 Disaster Simulation (June 4, 1996)

> **"The best documented and most expensive software bug in history."**

Animated simulation of the float → int16 cast overflow that destroyed a
**$370 million rocket** 37 seconds after launch.

![Ariane 5 disaster](output/ariane5_disaster.gif)

## What it shows

The demo accumulates horizontal velocity `BH` in **three precisions simultaneously**
and displays them live in the telemetry panel and graph:

| Precision | Mantissa bits | Exact integers up to | Behaviour in simulation |
|-----------|---------------|----------------------|-------------------------|
| `float64` | 52 | 2⁵³ | smooth curve — ground truth |
| `float32` | 23 | 2²⁴ | tiny drift (~0.01) — barely visible |
| `float16` | 10 | 2¹¹ = 2,048 | **staircases and freezes** at 16,384 |

### Timeline

| Time | Event |
|------|-------|
| 0 s | Ignition / liftoff |
| 5 s | Gravity turn — BH starts growing |
| ~10 s | float16 starts diverging (gap = 2) |
| ~25 s | **float16 stuck at 16,384** — increment absorbed by gap = 16 |
| 36.7 s | **BH > 32,767 → int16 Operand Error** — SRI halts |
| 36.7–39 s | Rocket tumbles, garbage steering commands |
| ~39 s | **Explosion** — 160-particle animation + shockwave |
| 45 s | Simulation ends |

### The actual bug (3 lines of Ada)

```ada
BH_Float : Long_Float;        -- 64-bit float, horizontal velocity
BH_Int16 : Integer_16;        -- 16-bit signed integer (-32768..+32767)
BH_Int16 := Integer_16(BH_Float);   -- unchecked cast → Operand Error
```

Code copied from Ariane 4, where the smaller rocket never reached `2^15 = 32,768`.
Ariane 5 crossed that threshold at t = 36.7 s.

## Requirements

- Python 3.8+
- numpy, matplotlib, pillow

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

### 4. Run

```bash
# Interactive animated window
python simulate_ariane5.py

# Save animation as GIF  →  output/ariane5_disaster.gif
python simulate_ariane5.py --save-gif

# Save 4 static snapshots  →  output/snapshot_t*.png
python simulate_ariane5.py --snapshots

# Adjust playback speed
python simulate_ariane5.py --speed 2
python simulate_ariane5.py --speed 0.5
```

### 5. Deactivate

```bash
deactivate
```

## Output files

| File | Description |
|------|-------------|
| `output/ariane5_disaster.gif` | Full 45-second animation |
| `output/ariane5_poster.png` | Final frame (explosion) |
| `output/snapshot_t15_climbing.png` | Rocket climbing, float16 already drifting |
| `output/snapshot_t30_gap_visible.png` | float16 stuck at 16,384 vs float64 = 22,977 |
| `output/snapshot_t37_overflow.png` | int16 OVERFLOW — SRI Operand Error |
| `output/snapshot_t40_explosion.png` | Vehicle destroyed |

## float16 gap math

For float16 (10-bit mantissa), at value `2^n` the gap is `2^(n−10)`:

| BH value | float16 gap | effect on accumulator |
|----------|-------------|----------------------|
| 1,024 | 1 | no issue |
| 2,048 | 2 | small drift |
| 8,192 | 8 | moderate drift |
| 16,384 | **16** | increment ≤ 16 → absorbed, BH **frozen** |
| 32,768 | 32 | (float64 already overflows int16 here) |

## References

- ESA/CNES Inquiry Board Report, July 19, 1996:
  https://www.ima.umn.edu/~arnold/disasters/ariane5rep.html

## Folder Structure

```
ariane5_disaster/
├── output/
│   ├── ariane5_disaster.gif
│   ├── ariane5_poster.png
│   └── snapshot_t*.png
├── simulate_ariane5.py   # Main simulation script
├── requirements.txt
├── README.md
└── .gitignore
```
