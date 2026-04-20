# Machine Learning — Integer Gaps in Floating-Point Numbers

Demonstrates how float16 integer gaps prevent a neural network from converging
when the target value falls inside the gap zone.

## What it shows

A simple gradient descent loop minimises `f(x) = (x − target)²`.
The same optimisation is run in **FP32** and **FP16** for two targets:

| Target | FP32 result | FP16 result |
|--------|-------------|-------------|
| `3.14159` (safe zone) | converges correctly | converges (slightly less accurate) |
| `2100.0` (gap zone, gap = 2) | converges correctly | **stuck at 2052 — never reaches 2100** |

**Why FP16 gets stuck at target = 2100:**

float16 has a 10-bit mantissa → integers are exact only up to `2^11 = 2,048`.
In the range `[2048, 4096)`, the gap between consecutive float16 values is **2**.
The nearest float16 to 2100 is 2100.0, but intermediate gradient steps land on
representable values that are far from the target, causing the optimiser to
oscillate and never converge.

## Output

One file: `output/training_simulation.png`

Four subplots:
- Loss curve (target = 3.14) — both converge
- x trajectory (target = 3.14) — both reach the target
- Loss curve (target = 2100) — FP16 loss flatlines, cannot reach zero
- x trajectory (target = 2100) — FP16 stuck, FP32 converges correctly

## Requirements

- Python 3.8+
- numpy
- matplotlib

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
PYTHONIOENCODING=utf-8 python demo_machine_learning.py
```

### 5. Deactivate

```bash
deactivate
```

## Key Takeaway

FP16 integer gaps don't just reduce precision — they can make optimisation
**completely unable to converge** when the target lies inside a gap zone.
Running more epochs does not help; the gap is a structural limitation of the
number format.

## Folder Structure

```
machine_learning/
├── output/
│   └── training_simulation.png   # FP16 vs FP32 convergence comparison
├── demo_machine_learning.py      # Main demo script
├── requirements.txt
├── README.md
└── .gitignore
```
