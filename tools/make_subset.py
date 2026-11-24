"""
Build an upload-friendly VocalSet subset.

Walks an extracted VocalSet tree, keeps a balanced subset of the target
techniques, downsamples to 16 kHz mono FLAC (≈20x smaller than 44.1 kHz WAV),
and mirrors the folder structure so singer/technique labels are preserved.

Usage (Windows):
    pip install soundfile scipy numpy
    python make_subset.py "C:\\path\\to\\VocalSet\\FULL"  vocalset_subset

Then zip the output folder and upload it. If the zip is still large, run with
--per 8 (fewer files per technique) or zip the 'female*' and 'male*' folders
into two separate archives.
"""
from __future__ import annotations
import os, sys, re, glob, argparse
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff")

TECHNIQUE_KEYWORDS = [   # multi-word tokens first
    "vocal_fry", "lip_trill",
    "inhaled", "breathy", "straight", "vibrato",
    "trillo", "trill", "belt", "spoken",
]
TECHNIQUES = ["straight", "vibrato", "breathy", "belt", "lip_trill",
              "trill", "trillo", "vocal_fry", "inhaled", "spoken"]
_SINGER_RE = re.compile(r"(female|male|f|m)\s*_?(\d+)", re.IGNORECASE)


def parse_tech(path):
    key = path.lower().replace(os.sep, "_")
    for t in TECHNIQUE_KEYWORDS:
        if t in key:
            return t if t in TECHNIQUES else None
    return None


def parse_singer(path):
    for part in path.replace("\\", "/").split("/"):
        m = _SINGER_RE.search(part)
        if m:
            g = "female" if m.group(1).lower().startswith("f") else "male"
            return f"{g}{m.group(2)}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--per", type=int, default=15,
                    help="max files per singer per technique")
    args = ap.parse_args()

    files = []
    for ext in AUDIO_EXT:
        files += glob.glob(os.path.join(args.src, "**", f"*{ext}"), recursive=True)
    print(f"found {len(files)} audio files under {args.src}")

    counts, kept, errors = {}, 0, 0
    total_bytes = 0
    for path in sorted(files):
        tech = parse_tech(path); singer = parse_singer(path)
        if not tech or not singer:
            continue
        k = (singer, tech)
        if counts.get(k, 0) >= args.per:
            continue
        try:
            y, sr = sf.read(path)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != TARGET_SR:
                g = np.gcd(int(sr), TARGET_SR)
                y = resample_poly(y, TARGET_SR // g, sr // g)
            y = (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32)
            out_dir = os.path.join(args.dst, singer, tech)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(out_dir, base + ".flac")
            sf.write(out, y, TARGET_SR, format="FLAC")
            total_bytes += os.path.getsize(out)
            counts[k] = counts.get(k, 0) + 1
            kept += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print("  skip:", path, "->", e)

    singers = sorted({s for s, _ in counts})
    print(f"\nkept {kept} clips | {len(singers)} singers | "
          f"{len({t for _, t in counts})} techniques")
    print(f"output size: {total_bytes/1e6:.0f} MB  ->  {args.dst}")
    print("techniques:", sorted({t for _, t in counts}))
    if kept == 0:
        print("\n!! nothing matched. Point 'src' at the folder that contains "
              "the female1/ male1/ ... singer folders.")


if __name__ == "__main__":
    main()
