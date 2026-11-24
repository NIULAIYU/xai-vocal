"""
VocalSet discovery and labelling.

Walks an extracted VocalSet tree, and for each audio file derives:
    * technique  -- parsed from the filename / parent folder against the
                    known VocalSet technique vocabulary
    * singer     -- the "female<N>" / "male<N>" (or f<N>/m<N>) identifier,
                    used for *speaker-independent* train/test splits
    * gender     -- female / male

Robust to whichever subset the user uploads: files whose technique is not in
config.TECHNIQUES are skipped.
"""
from __future__ import annotations
import os
import re
import glob
import pandas as pd

import config as C

AUDIO_EXT = (".wav", ".flac", ".aiff", ".aif", ".mp3")
_SINGER_RE = re.compile(r"(female|male|f|m)\s*_?(\d+)", re.IGNORECASE)


def _parse_technique(path: str) -> str | None:
    key = path.lower().replace(os.sep, "_")
    for tech in C.TECHNIQUE_KEYWORDS:      # multi-word tokens first
        if tech in key:
            base = tech
            # normalise a couple of dynamic variants onto their class if needed
            return base if base in C.TECHNIQUES else base
    return None


def _parse_singer(path: str) -> tuple[str | None, str | None]:
    for part in path.replace("\\", "/").split("/"):
        m = _SINGER_RE.search(part)
        if m:
            kind = m.group(1).lower()
            num = m.group(2)
            gender = "female" if kind.startswith("f") else "male"
            return f"{gender}{num}", gender
    # fall back to filename prefix (f1_..., m5_...)
    base = os.path.basename(path).lower()
    m = _SINGER_RE.match(base)
    if m:
        gender = "female" if m.group(1).startswith("f") else "male"
        return f"{gender}{m.group(2)}", gender
    return None, None


def scan(data_dir: str | None = None) -> pd.DataFrame:
    """Return a DataFrame: path, technique, singer, gender."""
    data_dir = data_dir or C.DATA_DIR
    rows = []
    for ext in AUDIO_EXT:
        for path in glob.iglob(os.path.join(data_dir, "**", f"*{ext}"),
                               recursive=True):
            tech = _parse_technique(path)
            if tech is None or tech not in C.TECHNIQUES:
                continue
            singer, gender = _parse_singer(path)
            if singer is None:
                continue
            rows.append({"path": path, "technique": tech,
                         "singer": singer, "gender": gender})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No labelled audio found under {data_dir!r}. "
            f"Check the folder structure and config.TECHNIQUES.")
    return df.sort_values(["singer", "technique"]).reset_index(drop=True)


if __name__ == "__main__":
    df = scan()
    print(f"{len(df)} files | {df.singer.nunique()} singers | "
          f"{df.technique.nunique()} techniques")
    print(df.technique.value_counts())
