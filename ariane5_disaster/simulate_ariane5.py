"""
=============================================================================
ARIANE 5 FLIGHT 501 DISASTER SIMULATION (June 4, 1996)
=============================================================================
Animated visualization of the floating-point / integer-gap / int16-cast
bug that destroyed a $370 million rocket 37 seconds after launch.

The demo visualizes THREE manifestations of "integer gaps in floats":

1. INTEGER GAP IN FLOAT16
   - float16 mantissa is only 10 bits -> can represent every integer up
     to 2^11 = 2,048. Beyond that, gap = 2, 4, 8, 16, 32...
   - The animation accumulates BH (horizontal velocity) in float16
     alongside float64. You can SEE the float16 curve staircase as the
     per-tick increment gets SMALLER than the float16 integer gap, so
     increments get absorbed entirely.

2. ACCUMULATED FLOAT PRECISION DRIFT
   - The same accumulation in float32 drifts slightly from float64,
     because 1.5 * sqrt(t-5) * dt is not exactly representable each tick.
   - This drift is small but real and shown as "drift" in telemetry.

3. INTEGER BOUNDARY (int16 overflow) -- the actual Ariane 5 bug
   - When BH > 32,767, casting to int16 raises Ada Operand Error.
   - This is the "integer side" of the same coin: any float > 32,767
     has no representable int16 -- a gap that spans to infinity.

Historical facts modelled:
- BH overflow at t ~= 36.7s  (real event: t = 36.7s)
- Vehicle breakup at t ~= 39s (real event: t = 39s)

How to run:
    python simulate_ariane5.py                 # interactive animation
    python simulate_ariane5.py --save-gif      # save as GIF
    python simulate_ariane5.py --speed 2       # 2x playback speed
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================
INT16_MAX = 32767
INT16_MIN = -32768

T_PITCH_START = 5.0
T_BREAKUP_DELAY = 2.3
SIM_END = 45.0

BH_COEF = 183.8        # tuned so BH hits 32,767 at t ~= 36.7s

FPS = 30
FRAME_DT = 1.0 / FPS
TOTAL_FRAMES = int(SIM_END * FPS)

# Rocket / flame / explosion sizes (bumped up for visibility)
ROCKET_SCALE = 320
EXPLOSION_PARTICLES = 160
EXPLOSION_SIZE_MULT = 2.5

# Colors
SKY_COLOR = "#0a1a3a"
SKY_TOP = "#050817"
GROUND_COLOR = "#2d1b0e"
PAD_COLOR = "#555555"
ROCKET_BODY = "#f2f2f2"
ROCKET_DARK = "#222222"
ROCKET_STRIPE = "#c62828"
FLAME_OUTER = "#ff6a00"
FLAME_INNER = "#ffd54f"
TRAIL_COLOR = "#ffa726"
NOMINAL_COLOR = "#4caf50"
TEXT_OK = "#00e676"
TEXT_WARN = "#ffd54f"
TEXT_BAD = "#ff1744"
TEXT_INFO = "#64b5f6"
GRID_COLOR = "#334466"
F64_COLOR = "#00e676"
F32_COLOR = "#ffd54f"
F16_COLOR = "#ff6a00"

EXPLOSION_COLORS = ["#ff1100", "#ff6a00", "#ffb400", "#ffea00", "#ffffff"]


# =============================================================================
# PHYSICS + PRECISION ACCUMULATION
# =============================================================================
def true_bh_instantaneous(t: float) -> float:
    """Closed-form BH(t) in float64 (ground truth)."""
    if t < T_PITCH_START:
        return 0.0
    return BH_COEF * (t - T_PITCH_START) ** 1.5


def precompute_bh_traces():
    """
    Precompute BH integrated in three precisions.

    The SRI computes BH by integrating horizontal acceleration over
    many tiny ticks.  We emulate that integration in float64, float32,
    and float16, using the analytical derivative of BH(t):

        d/dt [K * (t - 5)^1.5] = 1.5 * K * sqrt(t - 5)

    The dramatic effect is in float16: once BH grows past 2048, the
    float16 integer gap becomes larger than the per-tick increment,
    so increments are absorbed entirely -> curve flattens into steps.
    """
    dt = 0.004  # 4 ms tick (similar to real SRI sample rate ~72 Hz)
    n = int(SIM_END / dt) + 1
    ts = np.linspace(0, SIM_END, n)
    bh_f64 = np.zeros(n, dtype=np.float64)
    bh_f32 = np.zeros(n, dtype=np.float32)
    bh_f16 = np.zeros(n, dtype=np.float16)

    dt32 = np.float32(dt)
    dt16 = np.float16(dt)
    coef32 = np.float32(1.5 * BH_COEF)
    coef16 = np.float16(1.5 * BH_COEF)

    for i in range(1, n):
        t = ts[i]
        if t < T_PITCH_START:
            continue
        tau = t - T_PITCH_START
        dbh64 = 1.5 * BH_COEF * np.sqrt(tau) * dt
        bh_f64[i] = bh_f64[i - 1] + dbh64

        tau32 = np.float32(tau)
        dbh32 = coef32 * np.sqrt(tau32) * dt32
        bh_f32[i] = np.float32(bh_f32[i - 1]) + dbh32

        tau16 = np.float16(tau)
        dbh16 = coef16 * np.sqrt(tau16) * dt16
        candidate = np.float16(bh_f16[i - 1]) + dbh16
        if np.isfinite(candidate):
            bh_f16[i] = candidate
        else:
            bh_f16[i] = np.float16(65504)

    return ts, bh_f64, bh_f32, bh_f16, dt


_BH_TS, _BH_F64, _BH_F32, _BH_F16, _BH_DT = precompute_bh_traces()


def bh_values_at(t: float):
    """Return (true, f64_acc, f32_acc, f16_acc) BH values at time t."""
    idx = min(max(int(round(t / _BH_DT)), 0), len(_BH_TS) - 1)
    return (true_bh_instantaneous(t),
            float(_BH_F64[idx]),
            float(_BH_F32[idx]),
            float(_BH_F16[idx]))


def float16_gap_at(val: float) -> float:
    """Distance between `val` and the next representable float16."""
    if val >= 65504:
        return float("inf")
    v = np.float16(val)
    nxt = np.nextafter(v, np.float16(np.inf))
    return float(nxt) - float(v)


def float32_gap_at(val: float) -> float:
    v = np.float32(val)
    nxt = np.nextafter(v, np.float32(np.inf))
    return float(nxt) - float(v)


def cast_to_int16_unsafe(val: float):
    """The Ariane 5 bug: unchecked float -> int16 cast.

    In Ada this raised Operand Error when the value was out of range.
    """
    if val is None or not np.isfinite(val):
        return None, True
    if val > INT16_MAX or val < INT16_MIN:
        return None, True
    return int(val), False


# Trajectory (for rendering)
def nominal_altitude(t: float) -> float:
    if t < T_PITCH_START:
        return 0.5 * 10.0 * t ** 2
    t_p = t - T_PITCH_START
    y0 = 0.5 * 10.0 * T_PITCH_START ** 2
    vy0 = 10.0 * T_PITCH_START
    return y0 + vy0 * t_p + 0.5 * 8.0 * t_p ** 2


def nominal_horizontal(t: float) -> float:
    if t < T_PITCH_START:
        return 0.0
    t_p = t - T_PITCH_START
    return 0.5 * 3.0 * t_p ** 2


def nominal_vx(t: float) -> float:
    if t < T_PITCH_START:
        return 0.0
    return 3.0 * (t - T_PITCH_START)


def nominal_vy(t: float) -> float:
    if t < T_PITCH_START:
        return 10.0 * t
    return 10.0 * T_PITCH_START + 8.0 * (t - T_PITCH_START)


def nominal_pitch(t: float) -> float:
    if t < T_PITCH_START:
        return 0.0
    return min(np.radians(45), np.radians(1.5) * (t - T_PITCH_START))


# =============================================================================
# SIMULATION STATE
# =============================================================================
class Ariane5Simulation:
    def __init__(self):
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.pitch = 0.0
        self.sri_working = True
        self.overflow_time = None
        self.breakup_time = None
        self.explosion = None

        self.traj_x = [0.0]
        self.traj_y = [0.0]

        self._failure_x = 0.0
        self._failure_y = 0.0
        self._failure_vx = 0.0
        self._failure_vy = 0.0

    def step(self, t: float):
        self.t = t
        bh_true, bh64, bh32, bh16 = bh_values_at(t)

        # Overflow detection uses the true BH (as SRI computed with float64)
        _, overflowed = cast_to_int16_unsafe(bh64)
        if overflowed and self.sri_working:
            self.sri_working = False
            self.overflow_time = t
            self._failure_x = nominal_horizontal(t)
            self._failure_y = nominal_altitude(t)
            self._failure_vx = nominal_vx(t)
            self._failure_vy = nominal_vy(t)

        if self.sri_working:
            self.x = nominal_horizontal(t)
            self.y = nominal_altitude(t)
            self.vx = nominal_vx(t)
            self.vy = nominal_vy(t)
            self.pitch = nominal_pitch(t)
        else:
            dt_f = t - self.overflow_time
            g = 9.81
            self.x = self._failure_x + self._failure_vx * dt_f
            self.y = (self._failure_y
                      + self._failure_vy * dt_f
                      - 0.5 * g * dt_f ** 2)
            self.vx = self._failure_vx
            self.vy = self._failure_vy - g * dt_f

            base_pitch = nominal_pitch(self.overflow_time)
            chaos = np.radians(40) * np.sin(dt_f * 8.0) * min(dt_f * 2.0, 2.0)
            self.pitch = base_pitch + chaos

            if dt_f >= T_BREAKUP_DELAY and self.breakup_time is None:
                self.breakup_time = t
                self.explosion = Explosion(self.x, self.y, EXPLOSION_PARTICLES)

        if self.breakup_time is None:
            self.traj_x.append(self.x)
            self.traj_y.append(self.y)

        return bh_true, bh64, bh32, bh16


# =============================================================================
# EXPLOSION
# =============================================================================
class Explosion:
    def __init__(self, x0, y0, n_particles=EXPLOSION_PARTICLES):
        self.x0 = x0
        self.y0 = y0
        self.start_t = None
        rng = np.random.default_rng(42)

        angles = rng.uniform(0, 2 * np.pi, n_particles)
        speeds = rng.uniform(150, 700, n_particles)
        self.vx = speeds * np.cos(angles)
        self.vy = speeds * np.sin(angles) + 100
        self.sizes = rng.uniform(60, 340, n_particles) * EXPLOSION_SIZE_MULT
        color_idx = rng.integers(0, len(EXPLOSION_COLORS), n_particles)
        self.colors = [EXPLOSION_COLORS[i] for i in color_idx]
        self.lifetimes = rng.uniform(1.0, 3.5, n_particles)
        self.ring_radius = 0.0

    def get_particles(self, t):
        if self.start_t is None:
            self.start_t = t
        dt = t - self.start_t
        if dt < 0:
            return None
        px = self.x0 + self.vx * dt
        py = self.y0 + self.vy * dt - 0.5 * 300.0 * dt ** 2
        alpha = np.clip(1.0 - dt / self.lifetimes, 0.0, 1.0)
        size = self.sizes * (1.0 + dt * 0.5)
        self.ring_radius = dt * 900.0
        return px, py, size, alpha, self.colors


# =============================================================================
# DRAWING HELPERS
# =============================================================================
def _rotate(points, angle):
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def rocket_body_polygon(x, y, pitch, scale=ROCKET_SCALE):
    w = 0.28 * scale
    h = 1.9 * scale
    nh = 0.55 * scale
    fw = 0.25 * scale
    fh = 0.4 * scale
    pts = np.array([
        [-w / 2, 0],
        [-w / 2 - fw, 0],
        [-w / 2, fh],
        [-w / 2, h],
        [0, h + nh],
        [w / 2, h],
        [w / 2, fh],
        [w / 2 + fw, 0],
        [w / 2, 0],
    ])
    pts = _rotate(pts, pitch)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def rocket_stripe_polygon(x, y, pitch, scale=ROCKET_SCALE):
    w = 0.28 * scale
    stripe_h = 0.16 * scale
    stripe_y = 1.15 * scale
    pts = np.array([
        [-w / 2, stripe_y],
        [-w / 2, stripe_y + stripe_h],
        [w / 2, stripe_y + stripe_h],
        [w / 2, stripe_y],
    ])
    pts = _rotate(pts, pitch)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def rocket_window_polygon(x, y, pitch, scale=ROCKET_SCALE):
    """Nose-cone window circle approximated as a polygon."""
    cx = 0
    cy = 1.65 * scale
    r = 0.07 * scale
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
    pts = _rotate(pts, pitch)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def flame_polygon(x, y, pitch, t, scale=ROCKET_SCALE, intensity=1.0):
    w = 0.28 * scale
    flicker = 0.15 * np.sin(t * 40) + 0.08 * np.sin(t * 27 + 1.3)
    h = (1.0 + flicker) * scale * intensity
    pts = np.array([
        [-w / 2 * 0.7, 0],
        [-w / 3, -h * 0.3],
        [-w / 6, -h * 0.6],
        [0, -h],
        [w / 6, -h * 0.6],
        [w / 3, -h * 0.3],
        [w / 2 * 0.7, 0],
    ])
    pts = _rotate(pts, pitch)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def flame_inner_polygon(x, y, pitch, t, scale=ROCKET_SCALE, intensity=1.0):
    w = 0.28 * scale
    flicker = 0.1 * np.sin(t * 40 + 0.5)
    h = (0.62 + flicker) * scale * intensity
    pts = np.array([
        [-w / 4, 0],
        [0, -h],
        [w / 4, 0],
    ])
    pts = _rotate(pts, pitch)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


# =============================================================================
# MAIN ANIMATION
# =============================================================================
def create_animation(save_gif=False, speed=1.0):
    sim = Ariane5Simulation()

    fig = plt.figure(figsize=(17, 10), facecolor=SKY_TOP)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.22, wspace=0.1,
                           left=0.05, right=0.98, top=0.94, bottom=0.07)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_telem = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[1, 1])

    # ---------------- Main view ----------------
    ax_main.set_facecolor(SKY_COLOR)
    ax_main.set_xlim(-1500, 7500)
    ax_main.set_ylim(-500, 9500)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("Downrange distance (m)", color="white", fontsize=10)
    ax_main.set_ylabel("Altitude (m)", color="white", fontsize=10)
    ax_main.tick_params(colors="white", labelsize=9)
    for spine in ax_main.spines.values():
        spine.set_edgecolor("#555")
    ax_main.grid(True, alpha=0.15, color=GRID_COLOR)
    ax_main.set_title("Ariane 5 Flight 501  -  June 4, 1996",
                       color="white", fontsize=15, fontweight="bold", pad=10)

    rng = np.random.default_rng(1)
    star_x = rng.uniform(-1500, 7500, 100)
    star_y = rng.uniform(4000, 9500, 100)
    star_s = rng.uniform(1, 14, 100)
    ax_main.scatter(star_x, star_y, s=star_s, color="white", alpha=0.7, zorder=-8)

    ax_main.axhspan(-500, 0, color=GROUND_COLOR, zorder=-5)
    ax_main.plot([-1500, 7500], [0, 0], color="#5a3a1e", lw=2.5, zorder=-4)
    ax_main.add_patch(Rectangle((-150, 0), 300, 160, color=PAD_COLOR, zorder=-3))
    ax_main.add_patch(Rectangle((-110, 160), 24, 240, color="#444", zorder=-3))
    ax_main.add_patch(Rectangle((86, 160), 24, 240, color="#444", zorder=-3))

    # Nominal trajectory reference
    t_nom = np.linspace(0, 40, 200)
    x_nom = [nominal_horizontal(t) for t in t_nom]
    y_nom = [nominal_altitude(t) for t in t_nom]
    ax_main.plot(x_nom, y_nom, "--", color=NOMINAL_COLOR, alpha=0.35, lw=1.2,
                 label="Nominal trajectory", zorder=1)
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.3,
                    facecolor="#222", edgecolor="#555", labelcolor="white")

    trail_line, = ax_main.plot([], [], color=TRAIL_COLOR, alpha=0.85, lw=2.2, zorder=3)

    # Rocket
    rocket_patch = Polygon(rocket_body_polygon(0, 0, 0), closed=True,
                            facecolor=ROCKET_BODY, edgecolor=ROCKET_DARK,
                            linewidth=1.5, zorder=10)
    stripe_patch = Polygon(rocket_stripe_polygon(0, 0, 0), closed=True,
                            facecolor=ROCKET_STRIPE, edgecolor=None, zorder=11)
    window_patch = Polygon(rocket_window_polygon(0, 0, 0), closed=True,
                            facecolor="#1976d2", edgecolor="#000",
                            linewidth=1.0, zorder=12)
    flame_outer_patch = Polygon(flame_polygon(0, 0, 0, 0), closed=True,
                                  facecolor=FLAME_OUTER, alpha=0.85,
                                  edgecolor=None, zorder=9)
    flame_inner_patch = Polygon(flame_inner_polygon(0, 0, 0, 0), closed=True,
                                  facecolor=FLAME_INNER, alpha=0.95,
                                  edgecolor=None, zorder=9)
    ax_main.add_patch(flame_outer_patch)
    ax_main.add_patch(flame_inner_patch)
    ax_main.add_patch(rocket_patch)
    ax_main.add_patch(stripe_patch)
    ax_main.add_patch(window_patch)

    explosion_scatter = ax_main.scatter([], [], s=[], c=[], alpha=1.0, zorder=12)
    shockwave = Circle((0, 0), 0, fill=False, edgecolor="white",
                        linewidth=3, alpha=0, zorder=11)
    ax_main.add_patch(shockwave)

    event_text = ax_main.text(
        0.5, 0.92, "", color="white", fontsize=17, fontweight="bold",
        ha="center", va="center", transform=ax_main.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#000",
                   edgecolor="#fff", lw=2, alpha=0),
        zorder=20,
    )

    # ---------------- Telemetry panel ----------------
    ax_telem.set_facecolor("#0a0e27")
    ax_telem.set_xlim(0, 1)
    ax_telem.set_ylim(0, 1)
    ax_telem.axis("off")
    ax_telem.set_title("FLIGHT STATE  +  BH PRECISION ANALYSIS",
                        color="white", fontsize=11, fontweight="bold",
                        pad=6, loc="left")

    # -------- helper to add a row --------
    text_objs = {}

    def add_row(key, label, y, label_color="#8899bb", value_color=TEXT_OK,
                 label_x=0.03, value_x=0.44, bold=False):
        ax_telem.text(label_x, y, label, color=label_color, fontsize=10,
                      fontfamily="monospace", transform=ax_telem.transAxes,
                      fontweight="bold" if bold else "normal")
        text_objs[key] = ax_telem.text(
            value_x, y, "---", color=value_color, fontsize=10,
            fontfamily="monospace", transform=ax_telem.transAxes,
        )

    def add_header(label, y, color="#64b5f6"):
        ax_telem.text(0.03, y, label, color=color, fontsize=9,
                      fontfamily="monospace", fontweight="bold",
                      transform=ax_telem.transAxes)

    # Flight state
    add_header("FLIGHT STATE", 0.96, color=TEXT_INFO)
    add_row("T+",             "  T+",                         0.90)
    add_row("alt",            "  Altitude",                    0.85)
    add_row("pitch",          "  Pitch",                       0.80)

    # BH 3 precisions
    add_header("BH (horizontal velocity)  --  3 precisions", 0.72,
                color=TEXT_INFO)
    add_row("bh_f64", "  float64  (truth)", 0.66, value_color=F64_COLOR)
    add_row("bh_f32", "  float32  drift",   0.61, value_color=F32_COLOR)
    add_row("bh_f16", "  float16  gap=",    0.56, value_color=F16_COLOR)

    # int16 cast results
    add_header("Cast to int16 (Ariane 5 bug)", 0.47, color=TEXT_INFO)
    add_row("cast_f64", "  int16(float64)",  0.41)
    add_row("cast_f32", "  int16(float32)",  0.36)
    add_row("cast_f16", "  int16(float16)",  0.31)

    # Float integer gap indicator
    add_header("Integer gap (current BH)",   0.22, color=TEXT_INFO)
    add_row("gap16", "  float16 gap",        0.16, value_color=F16_COLOR)
    add_row("gap32", "  float32 gap",        0.11, value_color=F32_COLOR)

    # Status banner
    status_banner = ax_telem.text(
        0.5, 0.02, "SRI NOMINAL", color=TEXT_OK, fontsize=12, fontweight="bold",
        ha="center", transform=ax_telem.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#002200",
                   edgecolor=TEXT_OK, lw=2),
    )

    # ---------------- BH graph ----------------
    ax_graph.set_facecolor("#0a0e27")
    ax_graph.set_xlim(0, SIM_END)
    ax_graph.set_ylim(-2000, 50000)
    ax_graph.set_xlabel("Time (s)", color="white", fontsize=9)
    ax_graph.set_ylabel("BH value", color="white", fontsize=9)
    ax_graph.tick_params(colors="white", labelsize=8)
    for spine in ax_graph.spines.values():
        spine.set_edgecolor("#555")
    ax_graph.set_title("BH in float64 (smooth) vs float16 (staircase)  "
                        "[integer-gap effect]",
                        color="white", fontsize=10, pad=6)
    ax_graph.axhline(y=INT16_MAX, color=TEXT_BAD, ls="--", lw=1.3,
                      label=f"int16 max = {INT16_MAX}")
    ax_graph.axhline(y=65504, color="#888888", ls=":", lw=1.0,
                      label="float16 max")
    ax_graph.axhspan(INT16_MAX, 50000, color=TEXT_BAD, alpha=0.08)
    ax_graph.grid(True, alpha=0.2, color="#334466")

    # Pre-plot full float16 staircase as a reference (translucent)
    ax_graph.plot(_BH_TS, _BH_F16, color=F16_COLOR, lw=1.2, alpha=0.35,
                   label="float16 (full)")
    # Dynamic lines for float64 and float16 (highlighted animated portion)
    bh64_line, = ax_graph.plot([], [], color=F64_COLOR, lw=2.2,
                                label="float64 (live)")
    bh16_line, = ax_graph.plot([], [], color=F16_COLOR, lw=2.0,
                                label="float16 (live)")
    bh64_point = ax_graph.scatter([], [], s=70, c=F64_COLOR, zorder=5,
                                    edgecolors="white", linewidths=1.2)
    bh16_point = ax_graph.scatter([], [], s=55, c=F16_COLOR, zorder=5,
                                    edgecolors="white", linewidths=1.0)
    ax_graph.legend(loc="upper left", fontsize=7, framealpha=0.35,
                     facecolor="#222", edgecolor="#555", labelcolor="white",
                     ncol=1)

    bh64_hist_t, bh64_hist_v = [], []
    bh16_hist_t, bh16_hist_v = [], []

    # =================== update ===================
    def update(frame):
        t = frame * FRAME_DT
        bh_true, bh64, bh32, bh16 = sim.step(t)

        # Trail
        trail_line.set_data(sim.traj_x, sim.traj_y)

        # Rocket / flame
        if sim.breakup_time is None:
            rocket_patch.set_xy(rocket_body_polygon(sim.x, sim.y, sim.pitch))
            stripe_patch.set_xy(rocket_stripe_polygon(sim.x, sim.y, sim.pitch))
            window_patch.set_xy(rocket_window_polygon(sim.x, sim.y, sim.pitch))
            flame_intensity = 1.0 if sim.sri_working else 1.8
            flame_outer_patch.set_xy(
                flame_polygon(sim.x, sim.y, sim.pitch, t, intensity=flame_intensity))
            flame_inner_patch.set_xy(
                flame_inner_polygon(sim.x, sim.y, sim.pitch, t, intensity=flame_intensity))
            rocket_patch.set_alpha(1.0)
            stripe_patch.set_alpha(1.0)
            window_patch.set_alpha(1.0)
            flame_outer_patch.set_alpha(0.9)
            flame_inner_patch.set_alpha(0.95)
            explosion_scatter.set_offsets(np.empty((0, 2)))
            shockwave.set_alpha(0)
        else:
            rocket_patch.set_alpha(0)
            stripe_patch.set_alpha(0)
            window_patch.set_alpha(0)
            flame_outer_patch.set_alpha(0)
            flame_inner_patch.set_alpha(0)
            if sim.explosion is not None:
                result = sim.explosion.get_particles(t)
                if result is not None:
                    px, py, size, alpha, colors = result
                    mask = alpha > 0.01
                    if np.any(mask):
                        rgba = np.array([mcolors.to_rgba(c) for c in
                                          np.array(colors)[mask]])
                        rgba[:, 3] = alpha[mask]
                        explosion_scatter.set_offsets(
                            np.column_stack([px[mask], py[mask]]))
                        explosion_scatter.set_sizes(size[mask])
                        explosion_scatter.set_facecolors(rgba)
                    else:
                        explosion_scatter.set_offsets(np.empty((0, 2)))
                    shockwave.center = (sim.explosion.x0, sim.explosion.y0)
                    shockwave.set_radius(sim.explosion.ring_radius)
                    dt_since = t - sim.breakup_time
                    shockwave.set_alpha(max(0, 0.8 - dt_since * 0.5))

        # Telemetry: flight state
        text_objs["T+"].set_text(f"{t:+7.2f} s")
        text_objs["alt"].set_text(f"{max(0, sim.y):>9.0f} m")
        text_objs["pitch"].set_text(f"{np.degrees(sim.pitch):>9.1f} deg")

        # Telemetry: BH 3 precisions
        text_objs["bh_f64"].set_text(f"{bh64:>12,.2f}")
        drift_f32 = bh32 - bh64
        text_objs["bh_f32"].set_text(f"{bh32:>12,.2f}  ({drift_f32:+.3f})")
        drift_f16 = bh16 - bh64
        text_objs["bh_f16"].set_text(f"{bh16:>12,.2f}  ({drift_f16:+.1f})")

        # Cast-to-int16 results
        for key, val in [("cast_f64", bh64), ("cast_f32", bh32),
                         ("cast_f16", bh16)]:
            casted, overflowed = cast_to_int16_unsafe(val)
            if overflowed:
                text_objs[key].set_text("   OVERFLOW!")
                text_objs[key].set_color(TEXT_BAD)
            else:
                text_objs[key].set_text(f"{casted:>12,d}")
                # Colour by magnitude
                if abs(val) > 25000:
                    text_objs[key].set_color(TEXT_WARN)
                else:
                    text_objs[key].set_color(TEXT_OK)

        # Integer-gap indicators
        g16 = float16_gap_at(max(abs(bh64), 1.0))
        g32 = float32_gap_at(max(abs(bh64), 1.0))
        text_objs["gap16"].set_text(f"{g16:>11.4f}" +
                                      ("  !!" if g16 >= 1 else ""))
        text_objs["gap32"].set_text(f"{g32:>11.6f}")

        # Colour BH f64 red when overflowing
        if bh64 > INT16_MAX:
            text_objs["bh_f64"].set_color(TEXT_BAD)
        elif bh64 > 25000:
            text_objs["bh_f64"].set_color(TEXT_WARN)
        else:
            text_objs["bh_f64"].set_color(F64_COLOR)

        # Status banner
        if sim.breakup_time is not None:
            status_banner.set_text("*** VEHICLE DESTROYED ***")
            status_banner.set_color(TEXT_BAD)
            status_banner.get_bbox_patch().set(
                facecolor="#330000", edgecolor=TEXT_BAD)
        elif not sim.sri_working:
            status_banner.set_text("!!! SRI EXCEPTION - OPERAND ERROR !!!")
            status_banner.set_color(TEXT_BAD)
            status_banner.get_bbox_patch().set(
                facecolor="#330000", edgecolor=TEXT_BAD)
        elif bh64 > 25000:
            status_banner.set_text("WARNING: int16 LIMIT APPROACHING")
            status_banner.set_color(TEXT_WARN)
            status_banner.get_bbox_patch().set(
                facecolor="#332200", edgecolor=TEXT_WARN)
        elif g16 >= 1:
            status_banner.set_text(
                f"FLOAT16 INTEGER GAP = {g16:.0f}  (observe staircase)")
            status_banner.set_color(TEXT_WARN)
            status_banner.get_bbox_patch().set(
                facecolor="#332200", edgecolor=TEXT_WARN)
        else:
            status_banner.set_text("SRI NOMINAL")
            status_banner.set_color(TEXT_OK)
            status_banner.get_bbox_patch().set(
                facecolor="#002200", edgecolor=TEXT_OK)

        # Graph: accumulate history
        bh64_hist_t.append(t)
        bh64_hist_v.append(min(bh64, 50000))
        bh16_hist_t.append(t)
        bh16_hist_v.append(min(bh16, 50000))
        bh64_line.set_data(bh64_hist_t, bh64_hist_v)
        bh16_line.set_data(bh16_hist_t, bh16_hist_v)
        bh64_point.set_offsets([[t, min(bh64, 50000)]])
        bh16_point.set_offsets([[t, min(bh16, 50000)]])

        # Event banner on main
        if sim.breakup_time is not None:
            dt_b = t - sim.breakup_time
            if dt_b < 4.0:
                event_text.set_text("*** RAPID UNSCHEDULED DISASSEMBLY ***")
                event_text.set_color(TEXT_BAD)
                bbox = event_text.get_bbox_patch()
                bbox.set(facecolor="#330000", edgecolor=TEXT_BAD,
                          alpha=min(1.0, dt_b * 2))
            else:
                event_text.set_text("")
                event_text.get_bbox_patch().set(alpha=0)
        elif sim.overflow_time is not None:
            dt_o = t - sim.overflow_time
            if dt_o < T_BREAKUP_DELAY:
                event_text.set_text("!  INT16 OVERFLOW - OPERAND ERROR  !")
                event_text.set_color(TEXT_WARN)
                bbox = event_text.get_bbox_patch()
                bbox.set(facecolor="#332200", edgecolor=TEXT_WARN,
                          alpha=min(1.0, dt_o * 2))
            else:
                event_text.set_text("")
                event_text.get_bbox_patch().set(alpha=0)

        return (trail_line, rocket_patch, stripe_patch, window_patch,
                flame_outer_patch, flame_inner_patch,
                explosion_scatter, shockwave,
                bh64_line, bh16_line, bh64_point, bh16_point,
                event_text, status_banner,
                *text_objs.values())

    interval_ms = max(1, int(1000 / FPS / max(speed, 0.1)))
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                          interval=interval_ms, blit=False, repeat=False)

    if save_gif:
        gif_path = os.path.join(output_dir, "ariane5_disaster.gif")
        print(f"Saving GIF to:\n  {gif_path}")
        print("This may take 60-120 seconds...")
        writer = PillowWriter(fps=FPS)
        anim.save(gif_path, writer=writer, dpi=90)
        print("GIF saved successfully.")

        poster_path = os.path.join(output_dir, "ariane5_poster.png")
        fig.savefig(poster_path, dpi=120, facecolor=SKY_TOP)
        print(f"Poster saved to:\n  {poster_path}")
    else:
        plt.show()


def save_snapshots(times=(15.0, 30.0, 37.0, 40.0)):
    """Render one figure per time and save as PNG (for documentation)."""
    for t_target in times:
        sim = Ariane5Simulation()
        # Fast-forward frame-by-frame so the trajectory trail fills in
        fig_shot = plt.figure(figsize=(17, 10), facecolor=SKY_TOP)
        # Re-use create_animation's body by inline re-running -- here we
        # just do a simpler render: set up the figure and step to t_target.
        plt.close(fig_shot)


# ---------- simpler snapshot impl using FuncAnimation state ----------
def save_static_snapshots():
    """Generate 4 still images at key moments for documentation."""
    target_times = [15.0, 30.0, 37.0, 40.0]
    labels = ["t15_climbing", "t30_gap_visible", "t37_overflow", "t40_explosion"]

    for t_target, label in zip(target_times, labels):
        sim = Ariane5Simulation()
        # Build fresh figure with same layout by reusing create_animation
        # via a trick: set TOTAL_FRAMES to exactly the target frame count
        # and save only the final frame of that animation.
        target_frame = int(t_target * FPS)

        fig = plt.figure(figsize=(17, 10), facecolor=SKY_TOP)
        # Re-use by manual setup (compact duplicate):
        _render_single_frame(fig, sim, t_target)
        out_path = os.path.join(output_dir, f"snapshot_{label}.png")
        fig.savefig(out_path, dpi=110, facecolor=SKY_TOP)
        plt.close(fig)
        print(f"  Saved {out_path}")


def _render_single_frame(fig, sim, t_target):
    """
    Step the simulation to t_target and draw one static snapshot
    onto `fig`.  This is a trimmed version of create_animation().
    """
    # Step sim one frame at a time so trail accumulates
    n_frames = int(round(t_target * FPS))
    for i in range(1, n_frames + 1):
        sim.step(i * FRAME_DT)
    t = n_frames * FRAME_DT
    bh_true, bh64, bh32, bh16 = bh_values_at(t)

    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.22, wspace=0.1,
                           left=0.05, right=0.98, top=0.94, bottom=0.07)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_telem = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[1, 1])

    # Main view
    ax_main.set_facecolor(SKY_COLOR)
    ax_main.set_xlim(-1500, 7500)
    ax_main.set_ylim(-500, 9500)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("Downrange distance (m)", color="white", fontsize=10)
    ax_main.set_ylabel("Altitude (m)", color="white", fontsize=10)
    ax_main.tick_params(colors="white", labelsize=9)
    for spine in ax_main.spines.values():
        spine.set_edgecolor("#555")
    ax_main.grid(True, alpha=0.15, color=GRID_COLOR)
    ax_main.set_title(f"Ariane 5 Flight 501  --  snapshot at t = {t:.1f}s",
                       color="white", fontsize=15, fontweight="bold", pad=10)

    rng = np.random.default_rng(1)
    star_x = rng.uniform(-1500, 7500, 100)
    star_y = rng.uniform(4000, 9500, 100)
    star_s = rng.uniform(1, 14, 100)
    ax_main.scatter(star_x, star_y, s=star_s, color="white", alpha=0.7, zorder=-8)
    ax_main.axhspan(-500, 0, color=GROUND_COLOR, zorder=-5)
    ax_main.plot([-1500, 7500], [0, 0], color="#5a3a1e", lw=2.5, zorder=-4)
    ax_main.add_patch(Rectangle((-150, 0), 300, 160, color=PAD_COLOR, zorder=-3))

    # Nominal trajectory
    t_nom = np.linspace(0, 40, 200)
    x_nom = [nominal_horizontal(tt) for tt in t_nom]
    y_nom = [nominal_altitude(tt) for tt in t_nom]
    ax_main.plot(x_nom, y_nom, "--", color=NOMINAL_COLOR, alpha=0.35, lw=1.2)

    # Trail
    ax_main.plot(sim.traj_x, sim.traj_y, color=TRAIL_COLOR, alpha=0.85, lw=2.2)

    # Rocket
    if sim.breakup_time is None:
        ax_main.add_patch(Polygon(flame_polygon(sim.x, sim.y, sim.pitch, t),
                                    facecolor=FLAME_OUTER, alpha=0.9))
        ax_main.add_patch(Polygon(flame_inner_polygon(sim.x, sim.y, sim.pitch, t),
                                    facecolor=FLAME_INNER, alpha=0.95))
        ax_main.add_patch(Polygon(rocket_body_polygon(sim.x, sim.y, sim.pitch),
                                    facecolor=ROCKET_BODY, edgecolor=ROCKET_DARK,
                                    linewidth=1.5))
        ax_main.add_patch(Polygon(rocket_stripe_polygon(sim.x, sim.y, sim.pitch),
                                    facecolor=ROCKET_STRIPE))
        ax_main.add_patch(Polygon(rocket_window_polygon(sim.x, sim.y, sim.pitch),
                                    facecolor="#1976d2", edgecolor="#000"))
    else:
        # Explosion (use particle state at this moment)
        if sim.explosion is not None:
            sim.explosion.start_t = sim.breakup_time
            result = sim.explosion.get_particles(t)
            if result is not None:
                px, py, size, alpha, colors = result
                mask = alpha > 0.01
                if np.any(mask):
                    rgba = np.array([mcolors.to_rgba(c) for c in
                                      np.array(colors)[mask]])
                    rgba[:, 3] = alpha[mask]
                    ax_main.scatter(px[mask], py[mask], s=size[mask],
                                     c=rgba, zorder=12)

    # Telemetry
    ax_telem.set_facecolor("#0a0e27")
    ax_telem.set_xlim(0, 1); ax_telem.set_ylim(0, 1); ax_telem.axis("off")
    ax_telem.set_title("FLIGHT STATE  +  BH PRECISION",
                        color="white", fontsize=11, fontweight="bold",
                        pad=6, loc="left")

    rows = [
        ("FLIGHT STATE", None, TEXT_INFO, True),
        ("  T+",         f"{t:+7.2f} s",                F64_COLOR, False),
        ("  Altitude",   f"{max(0, sim.y):>9.0f} m",    TEXT_OK,   False),
        ("  Pitch",      f"{np.degrees(sim.pitch):>9.1f} deg", TEXT_OK, False),
        ("BH - 3 PRECISIONS", None, TEXT_INFO, True),
        ("  float64",    f"{bh64:>12,.2f}",
            TEXT_BAD if bh64 > INT16_MAX else F64_COLOR, False),
        ("  float32",    f"{bh32:>12,.2f}  (d {bh32-bh64:+.3f})", F32_COLOR, False),
        ("  float16",    f"{bh16:>12,.2f}  (d {bh16-bh64:+.1f})", F16_COLOR, False),
        ("CAST to int16", None, TEXT_INFO, True),
    ]
    casts = []
    for key, val, lbl in [("float64", bh64, "  int16(f64)"),
                           ("float32", bh32, "  int16(f32)"),
                           ("float16", bh16, "  int16(f16)")]:
        c, ov = cast_to_int16_unsafe(val)
        if ov:
            casts.append((lbl, "   OVERFLOW!", TEXT_BAD))
        else:
            casts.append((lbl, f"{c:>12,d}",
                           TEXT_WARN if abs(val) > 25000 else TEXT_OK))

    g16 = float16_gap_at(max(abs(bh64), 1.0))
    g32 = float32_gap_at(max(abs(bh64), 1.0))

    all_rows = rows + casts + [
        ("INT-GAP @ BH", None, TEXT_INFO, True),
        ("  float16 gap", f"{g16:>11.4f}" + ("  !!" if g16 >= 1 else ""),
            F16_COLOR, False),
        ("  float32 gap", f"{g32:>11.6f}",                F32_COLOR, False),
    ]

    y_pos = 0.96
    for row in all_rows:
        if len(row) == 4:
            label, val, color, is_header = row
        else:
            label, val, color = row
            is_header = False
        if is_header:
            ax_telem.text(0.03, y_pos, label, color=color, fontsize=9,
                           fontweight="bold", fontfamily="monospace",
                           transform=ax_telem.transAxes)
        else:
            ax_telem.text(0.03, y_pos, label, color="#8899bb", fontsize=10,
                           fontfamily="monospace", transform=ax_telem.transAxes)
            if val is not None:
                ax_telem.text(0.45, y_pos, val, color=color, fontsize=10,
                               fontfamily="monospace",
                               transform=ax_telem.transAxes)
        y_pos -= 0.055

    # Status
    if sim.breakup_time is not None:
        status, col, fb, eb = "*** VEHICLE DESTROYED ***", TEXT_BAD, "#330000", TEXT_BAD
    elif not sim.sri_working:
        status, col, fb, eb = "!!! SRI OPERAND ERROR !!!", TEXT_BAD, "#330000", TEXT_BAD
    elif bh64 > 25000:
        status, col, fb, eb = "WARNING: int16 LIMIT APPROACHING", TEXT_WARN, "#332200", TEXT_WARN
    elif g16 >= 1:
        status, col, fb, eb = f"FLOAT16 INT-GAP = {g16:.0f}", TEXT_WARN, "#332200", TEXT_WARN
    else:
        status, col, fb, eb = "SRI NOMINAL", TEXT_OK, "#002200", TEXT_OK
    ax_telem.text(0.5, 0.02, status, color=col, fontsize=12, fontweight="bold",
                   ha="center", transform=ax_telem.transAxes,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=fb,
                              edgecolor=eb, lw=2))

    # BH graph
    ax_graph.set_facecolor("#0a0e27")
    ax_graph.set_xlim(0, SIM_END)
    ax_graph.set_ylim(-2000, 50000)
    ax_graph.set_xlabel("Time (s)", color="white", fontsize=9)
    ax_graph.set_ylabel("BH value", color="white", fontsize=9)
    ax_graph.tick_params(colors="white", labelsize=8)
    for spine in ax_graph.spines.values():
        spine.set_edgecolor("#555")
    ax_graph.set_title("BH in float64 vs float16 (integer-gap staircase)",
                        color="white", fontsize=10, pad=6)
    ax_graph.axhline(y=INT16_MAX, color=TEXT_BAD, ls="--", lw=1.3,
                      label=f"int16 max = {INT16_MAX}")
    ax_graph.axhline(y=65504, color="#888888", ls=":", lw=1.0,
                      label="float16 max")
    ax_graph.axhspan(INT16_MAX, 50000, color=TEXT_BAD, alpha=0.08)
    ax_graph.grid(True, alpha=0.2, color="#334466")
    # Plot full f64/f16 up to current t
    mask = _BH_TS <= t
    ax_graph.plot(_BH_TS[mask], _BH_F64[mask], color=F64_COLOR, lw=2.2,
                   label="float64 (live)")
    ax_graph.plot(_BH_TS[mask], _BH_F16[mask], color=F16_COLOR, lw=2.0,
                   label="float16 (staircase)")
    ax_graph.scatter([t], [min(bh64, 50000)], s=70, c=F64_COLOR, zorder=5,
                      edgecolors="white", linewidths=1.2)
    ax_graph.scatter([t], [min(bh16, 50000)], s=55, c=F16_COLOR, zorder=5,
                      edgecolors="white", linewidths=1.0)
    ax_graph.legend(loc="upper left", fontsize=7, framealpha=0.35,
                     facecolor="#222", edgecolor="#555", labelcolor="white")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ariane 5 Flight 501 disaster simulation"
    )
    parser.add_argument("--save-gif", action="store_true",
                         help="Save as GIF instead of displaying")
    parser.add_argument("--snapshots", action="store_true",
                         help="Save 4 still PNGs at key moments")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed multiplier (default: 1.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ARIANE 5 FLIGHT 501 DISASTER SIMULATION (June 4, 1996)")
    print("=" * 70)
    print(f"  BH overflow expected at t ~= 36.7s")
    print(f"  Vehicle breakup expected at t ~= {36.7 + T_BREAKUP_DELAY:.1f}s")
    print(f"  Showing BH accumulated in float64 / float32 / float16")
    print(f"  Float16 integer-gap staircase visible from t ~= 10s")
    print("=" * 70)

    if args.snapshots:
        print("Rendering 4 static snapshots ...")
        save_static_snapshots()
        print("Done.")
    else:
        create_animation(save_gif=args.save_gif, speed=args.speed)
