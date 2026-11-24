"""
Global configuration for the Bel Canto XAI pipeline.

Central place for paths, audio settings, the feature/label vocabulary,
and the shared plotting style (Times New Roman 11 pt, consistent palette).
"""
from __future__ import annotations
import os
import matplotlib

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("VOCALSET_DIR", os.path.join(ROOT, "data"))   # extracted VocalSet
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

FEATURE_CSV = os.path.join(OUT_DIR, "features.csv")

# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
SR = 22050              # analysis sample rate (resampled on load)
F0_MIN = 65.0           # Hz, lower bound for pitch tracking (~C2)
F0_MAX = 1200.0         # Hz, upper bound (~D6)
MIN_DURATION = 0.5      # s, skip clips shorter than this
FRAME_SEC = 0.025       # 25 ms analysis frame
HOP_SEC = 0.010         # 10 ms hop

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
# Standard 10-technique VocalSet classification benchmark
# (Wilkins et al., ISMIR 2018; Yamamoto et al.). Multi-word tokens first so
# that filename parsing matches "lip_trill" before "trill", etc.
TECHNIQUE_KEYWORDS = [
    "vocal_fry", "lip_trill", "messa_di_voce",
    "fast_forte", "fast_piano", "slow_forte", "slow_piano",
    "inhaled", "breathy", "straight", "vibrato",
    "trillo", "trill", "belt", "spoken", "forte",
]
# Classes retained for the supervised task (edit to use a different subset).
TECHNIQUES = [
    "straight", "vibrato", "breathy", "belt",
    "lip_trill", "trill", "trillo", "vocal_fry",
    "inhaled", "spoken",
]

# Human-readable names + Chinese counterparts (journal requires zh where applicable)
TECHNIQUE_ZH = {
    "straight":  ("Straight tone", "直声"),
    "vibrato":   ("Vibrato",       "颤音"),
    "breathy":   ("Breathy",       "气声"),
    "belt":      ("Belt",          "强声/贝尔特"),
    "lip_trill": ("Lip trill",     "唇颤音"),
    "trill":     ("Trill",         "颤音装饰"),
    "trillo":    ("Trillo",        "快速颤音"),
    "vocal_fry": ("Vocal fry",     "气泡音"),
    "inhaled":   ("Inhaled",       "吸气发声"),
    "spoken":    ("Spoken",        "说话声"),
}

# --------------------------------------------------------------------------- #
# Plot style
# --------------------------------------------------------------------------- #
# Real Times New Roman is used when present (e.g. the user's machine); the
# metric-compatible Liberation Serif / Tinos is the fallback in headless envs.
SERIF_STACK = ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"]
BASE_FONT_PT = 11

# Consistent, print-friendly categorical palette (muted, colour-blind aware).
PALETTE = [
    "#2E4A62",  # deep slate blue
    "#C0504D",  # brick red
    "#4F8A5B",  # moss green
    "#D9A441",  # ochre
    "#6E5B8A",  # muted violet
    "#3C8DAD",  # teal
    "#9C6B3F",  # sienna
    "#8C8C8C",  # grey
    "#B5546A",  # dusty rose
    "#5A7D4F",  # olive
]
SEQ_CMAP = "YlGnBu"       # sequential (heatmaps)
DIV_CMAP = "RdBu_r"       # diverging (SHAP / correlation)


def apply_style() -> None:
    """Apply the shared matplotlib style. Call once before plotting."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT - 1,
        "ytick.labelsize": BASE_FONT_PT - 1,
        "legend.fontsize": BASE_FONT_PT - 1,
        "figure.titlesize": BASE_FONT_PT + 1,
        "mathtext.fontset": "stix",       # Times-like math glyphs
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,               # embed TrueType (editable text)
        "ps.fonttype": 42,
    })
