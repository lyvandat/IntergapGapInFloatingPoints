# Navigation & Robotics — Integer Gaps in Floating-Point Numbers

Reproduces the **Patriot Missile bug (Feb 25, 1991)** — a real disaster caused
by the truncation of `0.1` in a 24-bit signed fixed-point register.

## The bug

The Patriot system tracked time using a **24-bit signed fixed-point register**
(Intel 8086).  A 24-bit signed register has 1 sign bit + 23 fraction bits,
so the denominator is `2^23 = 8,388,608`.

`0.1` in binary is `0.000110011001100110011…` (infinite repeating).
Truncated to 23 fraction bits:

```
floor(0.1 × 2²³) / 2²³  =  838860 / 8388608  =  0.09999990463…
```

**Error per tick** = `0.1 − 0.09999990463 ≈ 9.54 × 10⁻⁸ seconds`

After 100 hours (`3,600,000 ticks`):
```
3,600,000 × 9.54 × 10⁻⁸  =  0.343 seconds  →  575 metres  ✓ (ESA report: 0.343 s / 573 m)
```

> **float32 is NOT the bug.**  float32 rounds `0.1` *up* to `0.100000001490…`
> (opposite sign, only 0.005 s error after 100 h).  The actual register used
> 24-bit fixed-point *truncation*, which rounds *down*.

## Output

One file: `output/1_patriot_missile.png`

Four subplots:
- Accumulated clock error: 24-bit (lags) vs float32 (leads) vs ESA reference
- Scud position error in metres at each test hour
- Binary expansion of `0.1` — bits beyond position 23 are truncated
- Summary analysis box

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
PYTHONIOENCODING=utf-8 python demo_navigation_robotics.py
```

### 5. Deactivate

```bash
deactivate
```

## Key Takeaway

The error is not about floating-point *overflow* or *underflow* — it is about
the fundamental impossibility of representing `1/10` exactly in binary.
Regardless of the precision used (float16, float32, float64, or fixed-point),
any finite binary representation of `0.1` introduces a small systematic error
that grows linearly with the number of accumulations.

The Patriot's specific 24-bit signed truncation produced exactly the
`9.54 × 10⁻⁸ s/tick` error that accumulated to the fatal `0.343 s` after 100 h.

## Folder Structure

```
navigation_robotics/
├── output/
│   └── 1_patriot_missile.png     # Accumulated error + binary expansion charts
├── demo_navigation_robotics.py   # Main demo script
├── requirements.txt
├── README.md
└── .gitignore
```
