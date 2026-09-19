# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Render the U-Net input/output schematic for the TRB conference paper.

The figure explains the tensor contract of the accident speed-forecasting
U-Net: three 36 (space) x 36 (time) input channels are consumed by a
four-level encoder-decoder, and a 36 (space) x 30 (time) post-report speed
field is emitted after a sigmoid activation.

The output is a hand-authored, dependency-free SVG so that every glyph,
proportion and colour is deterministic and vector-clean at print resolution.

Examples:
    uv run scripts/visualization/plot_unet_io_schematic.py
    uv run scripts/visualization/plot_unet_io_schematic.py --out-dir /tmp/fig
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "paper_trb_2026" / "figures" / "architecture"

# ---------------------------------------------------------------------------
# Canvas and palette
# ---------------------------------------------------------------------------

W, H = 1600.0, 900.0

NAVY = "#12294A"
NAVY_MID = "#274B7A"
TEAL = "#0E7C86"
TEAL_LIGHT = "#3FA3A8"
AMBER = "#D99A2B"
GRAY = "#8A94A6"
GRAY_DARK = "#5B6577"
GRAY_ZERO = "#DCE0E6"
RULE = "#C6CCD6"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

# Slow -> free-flow speed ramp, kept inside the navy/teal/amber family.
SPEED_STOPS: list[tuple[float, str]] = [
    (0.00, "#7B3A11"),
    (0.28, "#C2761B"),
    (0.52, "#E5BA63"),
    (0.76, "#77B4B4"),
    (1.00, "#0E7C86"),
]

# ---------------------------------------------------------------------------
# Tensor geometry
# ---------------------------------------------------------------------------

N_SPACE = 36
N_TIME_IN = 36
N_PRE = 6
N_POST = 30

IN_CELL = 4.9
IN_W = N_TIME_IN * IN_CELL          # 176.4
IN_X0 = 140.0
IN_TOPS = (138.0, 380.0, 622.0)

OUT_CELL = 6.6
OUT_W = N_POST * OUT_CELL           # 198.0
OUT_H = N_SPACE * OUT_CELL          # 237.6
OUT_X0 = 1230.0

AXIS_Y = 380.0                      # main horizontal flow axis
OUT_TOP = AXIS_Y - OUT_H / 2.0

ENC_CX, DEC_CX = 700.0, 960.0
LEVEL_Y = (380.0, 470.0, 560.0, 650.0)
LEVEL_W = (120.0, 104.0, 88.0, 72.0)
LEVEL_H = (54.0, 46.0, 38.0, 30.0)
ENC_LABELS = ("64", "128", "256", "512")
DEC_LABELS = ("64", "128", "256", "512")

COLLECT_X = 400.0


def lerp_color(a: str, b: str, t: float) -> str:
    ar, ag, ab = (int(a[i : i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(b[i : i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(
        round(ar + (br - ar) * t),
        round(ag + (bg - ag) * t),
        round(ab + (bb - ab) * t),
    )


def speed_color(v: float) -> str:
    """Map a normalised speed in [0, 1] (0 = stopped, 1 = free flow) to a hex colour."""
    v = min(1.0, max(0.0, v))
    for (lo, c_lo), (hi, c_hi) in zip(SPEED_STOPS, SPEED_STOPS[1:]):
        if v <= hi:
            span = hi - lo
            return lerp_color(c_lo, c_hi, 0.0 if span == 0 else (v - lo) / span)
    return SPEED_STOPS[-1][1]


# ---------------------------------------------------------------------------
# Synthetic-but-plausible field content
# ---------------------------------------------------------------------------

ACCIDENT_ROW = 20
RECURRENT_ROWS = range(3, 9)


def _jitter(rng: random.Random, scale: float = 0.035) -> float:
    return rng.uniform(-scale, scale)


def observed_speed_field(rng: random.Random) -> list[list[float]]:
    """36 space x 6 pre-report intervals of realised speed before the report."""
    field = []
    for r in range(N_SPACE):
        row = []
        for t in range(N_PRE):
            v = 0.86
            if r in RECURRENT_ROWS:
                v = 0.63 + 0.02 * math.sin(r * 0.9)
            # Faint onset of the incident disturbance in the final intervals.
            if t >= 4 and ACCIDENT_ROW - 2 <= r <= ACCIDENT_ROW + 1:
                v -= 0.16 * (t - 3) / 2.0
            row.append(min(1.0, max(0.0, v + _jitter(rng))))
        field.append(row)
    return field


def validity_field(rng: random.Random) -> list[list[bool]]:
    """True = valid observation (white), False = unavailable (black)."""
    dead_rows = {12, 29}
    field = []
    for r in range(N_SPACE):
        row = []
        for _ in range(N_PRE):
            if r in dead_rows:
                row.append(False)
            else:
                row.append(rng.random() > 0.045)
        field.append(row)
    # A detector that drops out part-way through the window.
    for t in range(3, N_PRE):
        field[6][t] = False
    return field


def historical_speed_field(rng: random.Random) -> list[list[float]]:
    """Causal historical profile, populated across the full 36-interval window."""
    field = []
    for r in range(N_SPACE):
        row = []
        for t in range(N_TIME_IN):
            v = 0.87 - 0.05 * (t / (N_TIME_IN - 1))  # gentle peak-hour build-up
            if r in RECURRENT_ROWS:
                v = 0.66 - 0.14 * (t / (N_TIME_IN - 1)) + 0.02 * math.sin(r * 0.9)
            elif 24 <= r <= 27:
                v -= 0.10
            row.append(min(1.0, max(0.0, v + _jitter(rng, 0.03))))
        field.append(row)
    return field


def predicted_speed_field(rng: random.Random) -> list[list[float]]:
    """36 space x 30 post-report intervals: queue growth, plateau and recovery."""

    def extent(t: int) -> float:
        if t < 1:
            return 0.0
        if t <= 11:
            return 13.0 * (t - 1) / 10.0
        if t <= 15:
            return 13.0
        if t <= 25:
            return 13.0 * (25 - t) / 10.0
        return 0.0

    field = []
    for r in range(N_SPACE):
        row = []
        for t in range(N_POST):
            v = 0.86 - 0.04 * (t / (N_POST - 1))
            if r in RECURRENT_ROWS:
                v = 0.66 - 0.10 * (t / (N_POST - 1))
            if r > ACCIDENT_ROW:
                v = min(1.0, v + 0.06)  # metered discharge downstream of the incident
            e = extent(t)
            if e > 0 and ACCIDENT_ROW - e <= r <= ACCIDENT_ROW:
                depth = (ACCIDENT_ROW - r) / max(e, 1.0)
                v = min(v, 0.13 + 0.55 * depth)
            row.append(min(1.0, max(0.0, v + _jitter(rng, 0.025))))
        field.append(row)
    return field


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 11.0,
    fill: str = NAVY,
    weight: str = "400",
    anchor: str = "start",
    spacing: float = 0.0,
    style: str = "normal",
) -> str:
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    it = ' font-style="italic"' if style == "italic" else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}{it}>{esc(s)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, *, stroke: str = "none", sw: float = 1.0, rx: float = 0.0) -> str:
    r = f' rx="{rx}"' if rx else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{r}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, sw: float = 1.0, marker: str | None = None) -> str:
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="butt"{m}/>'
    )


def path(d: str, stroke: str, sw: float = 1.0, marker: str | None = None, fill: str = "none") -> str:
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"{m}/>'


def bracket(x0: float, x1: float, y: float, stroke: str = GRAY, tick: float = 4.0) -> str:
    return (
        path(
            f"M {x0:.2f} {y - tick:.2f} L {x0:.2f} {y:.2f} L {x1:.2f} {y:.2f} L {x1:.2f} {y - tick:.2f}",
            stroke,
            0.9,
        )
    )


# ---------------------------------------------------------------------------
# Composite pieces
# ---------------------------------------------------------------------------


def draw_input_grid(
    parts: list[str],
    top: float,
    title: str,
    *,
    speed: list[list[float]] | None = None,
    valid: list[list[bool]] | None = None,
    full_speed: list[list[float]] | None = None,
) -> None:
    x0, cell = IN_X0, IN_CELL
    x1 = x0 + IN_W
    bottom = top + IN_W
    xdiv = x0 + N_PRE * cell

    # --- cells -------------------------------------------------------------
    if full_speed is not None:
        for r in range(N_SPACE):
            for t in range(N_TIME_IN):
                parts.append(
                    rect(x0 + t * cell, top + r * cell, cell + 0.06, cell + 0.06, speed_color(full_speed[r][t]))
                )
    else:
        for r in range(N_SPACE):
            for t in range(N_PRE):
                if speed is not None:
                    fill = speed_color(speed[r][t])
                else:
                    fill = "#FFFFFF" if valid[r][t] else "#111820"
                parts.append(rect(x0 + t * cell, top + r * cell, cell + 0.06, cell + 0.06, fill))
        # Structural-zero placeholders over the 30 post-report intervals.
        parts.append(rect(xdiv, top, x1 - xdiv, IN_W, GRAY_ZERO))
        parts.append(
            txt(
                (xdiv + x1) / 2.0,
                top + IN_W / 2.0 + 4.0,
                "zero",
                size=13.0,
                fill=GRAY_DARK,
                anchor="middle",
                spacing=1.2,
            )
        )

    if valid is not None:
        # Hairline so that white valid cells stay legible against the page.
        parts.append(rect(x0, top, xdiv - x0, IN_W, "none", stroke=RULE, sw=0.6))

    parts.append(rect(x0, top, IN_W, IN_W, "none", stroke=NAVY, sw=1.1))
    parts.append(line(xdiv, top - 1.0, xdiv, bottom + 1.0, NAVY, 2.6))

    # --- annotations -------------------------------------------------------
    parts.append(txt(x0, top - 30.0, title, size=13.0, fill=NAVY, weight="600"))
    parts.append(bracket(x0, xdiv, top - 8.0))
    parts.append(bracket(xdiv, x1, top - 8.0))
    parts.append(txt(xdiv - 3.0, top - 13.0, "6 pre-report intervals", size=8.6, fill=GRAY_DARK, anchor="end"))
    parts.append(
        txt((xdiv + x1) / 2.0, top - 13.0, "30 post-report intervals", size=8.6, fill=GRAY_DARK, anchor="middle")
    )

    parts.append(txt((x0 + x1) / 2.0, bottom + 14.0, "36 time", size=9.5, fill=GRAY_DARK, anchor="middle"))
    cy = top + IN_W / 2.0
    parts.append(
        f'<g transform="translate({x0 - 11.0:.1f},{cy:.1f}) rotate(-90)">'
        + txt(0, 0, "36 space", size=9.5, fill=GRAY_DARK, anchor="middle")
        + "</g>"
    )


def draw_unet(parts: list[str]) -> None:
    for i in range(4):
        w, h, y = LEVEL_W[i], LEVEL_H[i], LEVEL_Y[i]
        ex, dx = ENC_CX - w / 2.0, DEC_CX - w / 2.0
        parts.append(rect(ex, y - h / 2.0, w, h, NAVY_MID, stroke=NAVY, sw=1.0, rx=2.5))
        parts.append(rect(dx, y - h / 2.0, w, h, TEAL, stroke="#0A5A62", sw=1.0, rx=2.5))
        parts.append(txt(ENC_CX, y + 4.2, ENC_LABELS[i], size=12.5, fill="#FFFFFF", weight="600", anchor="middle"))
        parts.append(txt(DEC_CX, y + 4.2, DEC_LABELS[i], size=12.5, fill="#FFFFFF", weight="600", anchor="middle"))

        # Skip connections / bottleneck link.
        parts.append(line(ex + w + 5.0, y, dx - 9.0, y, GRAY, 1.1, marker="arrow-gray"))

    # Encoder down / decoder up arrows.
    for i in range(3):
        y_a = LEVEL_Y[i] + LEVEL_H[i] / 2.0
        y_b = LEVEL_Y[i + 1] - LEVEL_H[i + 1] / 2.0
        parts.append(line(ENC_CX, y_a + 3.0, ENC_CX, y_b - 8.0, NAVY, 1.2, marker="arrow-navy"))
        parts.append(line(DEC_CX, y_b - 3.0, DEC_CX, y_a + 8.0, TEAL, 1.2, marker="arrow-teal"))

    parts.append(txt(830.0, LEVEL_Y[0] - 34.0, "skip connections", size=8.8, fill=GRAY_DARK, anchor="middle"))
    parts.append(txt(ENC_CX, LEVEL_Y[3] + 30.0, "encoder", size=9.5, fill=GRAY_DARK, anchor="middle", spacing=0.8))
    parts.append(txt(DEC_CX, LEVEL_Y[3] + 30.0, "decoder", size=9.5, fill=GRAY_DARK, anchor="middle", spacing=0.8))


def draw_output_grid(parts: list[str], field: list[list[float]]) -> None:
    x0, y0, cell = OUT_X0, OUT_TOP, OUT_CELL
    x1, y1 = x0 + OUT_W, y0 + OUT_H

    for r in range(N_SPACE):
        for t in range(N_POST):
            parts.append(rect(x0 + t * cell, y0 + r * cell, cell + 0.06, cell + 0.06, speed_color(field[r][t])))
    parts.append(rect(x0, y0, OUT_W, OUT_H, "none", stroke=NAVY, sw=1.1))

    parts.append(
        txt((x0 + x1) / 2.0, y0 - 16.0, "Predicted post-report speed", size=14.0, fill=NAVY, weight="600", anchor="middle")
    )
    parts.append(txt((x0 + x1) / 2.0, y1 + 15.0, "30 time", size=9.5, fill=GRAY_DARK, anchor="middle"))
    cy = (y0 + y1) / 2.0
    parts.append(
        f'<g transform="translate({x0 - 11.0:.1f},{cy:.1f}) rotate(-90)">'
        + txt(0, 0, "36 space", size=9.5, fill=GRAY_DARK, anchor="middle")
        + "</g>"
    )
    parts.append(txt((x0 + x1) / 2.0, y1 + 42.0, "36 space × 30 time", size=12.0, fill=NAVY, weight="600", anchor="middle"))
    parts.append(txt((x0 + x1) / 2.0, y1 + 61.0, "150-min forecast horizon", size=11.0, fill=GRAY_DARK, anchor="middle"))


def draw_legend(parts: list[str]) -> None:
    bar_x, bar_y, bar_w, bar_h = 668.0, 726.0, 200.0, 11.0
    steps = 80
    for i in range(steps):
        parts.append(
            rect(
                bar_x + bar_w * i / steps,
                bar_y,
                bar_w / steps + 0.4,
                bar_h,
                speed_color(i / (steps - 1)),
            )
        )
    parts.append(rect(bar_x, bar_y, bar_w, bar_h, "none", stroke=RULE, sw=0.8))
    parts.append(txt(bar_x, bar_y + 24.0, "congested", size=8.8, fill=GRAY_DARK))
    parts.append(txt(bar_x + bar_w, bar_y + 24.0, "free flow", size=8.8, fill=GRAY_DARK, anchor="end"))
    parts.append(txt(bar_x - 10.0, bar_y + 9.0, "speed", size=9.5, fill=NAVY, weight="600", anchor="end"))

    sx, sy = 900.0, bar_y
    swatches = (
        ("#FFFFFF", "valid"),
        ("#111820", "unavailable"),
        (GRAY_ZERO, "structural zero"),
    )
    for i, (fill, label) in enumerate(swatches):
        x = sx + i * 108.0
        parts.append(rect(x, sy, 11.0, 11.0, fill, stroke=RULE, sw=0.8))
        parts.append(txt(x + 16.0, sy + 9.0, label, size=8.8, fill=GRAY_DARK))


def build_svg(seed: int) -> str:
    rng = random.Random(seed)
    obs = observed_speed_field(rng)
    val = validity_field(rng)
    hist = historical_speed_field(rng)
    pred = predicted_speed_field(rng)

    p: list[str] = []
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

    # Title ribbon.
    p.append(rect(0, 0, W, 58, NAVY))
    p.append(
        txt(
            W / 2.0,
            36.0,
            "Three 36 × 36 input channels → 36 × 30 post-report prediction",
            size=21.0,
            fill="#FFFFFF",
            weight="600",
            anchor="middle",
            spacing=0.4,
        )
    )

    # Section captions.
    for x, s, anchor in ((IN_X0, "INPUT CHANNELS", "start"), (830.0, "U-NET", "middle"), (OUT_X0 + OUT_W / 2.0, "OUTPUT", "middle")):
        p.append(txt(x, 100.0, s, size=10.5, fill=GRAY, weight="600", anchor=anchor, spacing=1.7))

    # Input channels.
    draw_input_grid(p, IN_TOPS[0], "Observed speed", speed=obs)
    draw_input_grid(p, IN_TOPS[1], "Observed validity", valid=val)
    draw_input_grid(p, IN_TOPS[2], "Causal historical speed", full_speed=hist)

    in_bottom = IN_TOPS[2] + IN_W
    p.append(txt(IN_X0, in_bottom + 42.0, "3 channels × 36 space × 36 time", size=12.0, fill=NAVY, weight="600"))
    p.append(txt(IN_X0, in_bottom + 61.0, "−30 to −5 min | 150-min forecast horizon", size=11.0, fill=GRAY_DARK))

    # Collector from the three channels into the encoder.
    x1_in = IN_X0 + IN_W
    centres = [t + IN_W / 2.0 for t in IN_TOPS]
    for cy in centres:
        p.append(line(x1_in + 4.0, cy, COLLECT_X, cy, GRAY, 1.0))
    p.append(line(COLLECT_X, centres[0], COLLECT_X, centres[2], GRAY, 1.0))
    p.append(
        path(
            f"M {COLLECT_X:.1f} {centres[1]:.1f} C 500 {centres[1]:.1f}, 540 {AXIS_Y:.1f}, {ENC_CX - LEVEL_W[0] / 2.0 - 9.0:.1f} {AXIS_Y:.1f}",
            NAVY,
            1.4,
            marker="arrow-navy",
        )
    )

    draw_unet(p)

    # Sigmoid head.
    x_dec_r = DEC_CX + LEVEL_W[0] / 2.0
    p.append(line(x_dec_r + 5.0, AXIS_Y, OUT_X0 - 9.0, AXIS_Y, AMBER, 1.6, marker="arrow-amber"))
    p.append(txt((x_dec_r + OUT_X0) / 2.0, AXIS_Y - 10.0, "Sigmoid", size=12.5, fill=NAVY, weight="600", anchor="middle"))

    draw_output_grid(p, pred)
    draw_legend(p)

    defs = "".join(
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
        f'markerHeight="6.5" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>'
        for name, colour in (("navy", NAVY), ("teal", TEAL), ("gray", GRAY), ("amber", AMBER))
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.0f} {H:.0f}" shape-rendering="crispEdges">'
        f"<defs>{defs}</defs>" + "".join(p) + "</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="unet_io_schematic")
    parser.add_argument("--seed", type=int, default=20260727, help="Seed for the illustrative field textures.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.out_dir / f"{args.stem}.svg"
    svg_path.write_text(build_svg(args.seed), encoding="utf-8")
    # The project root contains CJK characters; guard against cp1252 consoles.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
