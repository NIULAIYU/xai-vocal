# Explainable AI for Bel Canto Vocal Training

Code and data for the paper *Explainable Artificial Intelligence for
Personalized Bel Canto Training: A Framework for Acoustic Feedback and Vocal
Skill Development*.

The pipeline describes each singing clip with 31 interpretable acoustic
features in four pedagogical families (pitch stability, vibrato, resonance,
breath control), classifies ten VocalSet techniques with gradient-boosted
trees under speaker-independent cross-validation, and explains every decision
with SHAP.

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # includes praat-parselmouth (Praat)
```

No GPU is needed; feature extraction and model training run on the CPU and
use all cores.

## Data

VocalSet is not redistributed here. Download it from Zenodo
(<https://zenodo.org/records/1442513>) and extract it.

The paper analyses a subset of VocalSet: at most 15 files per singer and
technique for the ten techniques in `config.TECHNIQUES`, converted to 16 kHz
mono FLAC (1,681 clips). To rebuild exactly that subset:

```bash
python tools/make_subset.py /path/to/VocalSet/FULL vocalset_subset --per 15
export VOCALSET_DIR=$PWD/vocalset_subset   # Windows: set VOCALSET_DIR=...
```

## Reproducing the paper

The extracted feature matrix is included (`outputs/features.csv`), so the
modelling and explanation steps run without audio. Steps that re-analyse the
audio (robustness tests, vibrato validation on real clips, pitch-tracker
comparison) need `VOCALSET_DIR`.

```bash
python run_all.py                     # features, CV evaluation, SHAP, base figures
python -m tools.revision_stats        # bootstrap CIs, McNemar, nested CV, baselines,
                                      #   attribution agreement, grouped SHAP, instance examples
python -m tools.revision_audio        # non-sinusoidal and real-vibrato validation, robustness
python -m tools.revision_extra        # focal loss, leave-one-singer-out, pYIN vibrato
python -m tools.revision_pyin_diag    # Praat vs pYIN on identical time spans
python -m tools.revision_figures      # Figures 2 and 4 to 9
python -m tools.fig_framework         # Figure 1
```

If `outputs/features.csv` exists, `run_all.py` reuses it instead of
re-extracting features; delete it to extract from the audio again.

## Validating the feature extractors (no data needed)

```bash
python -m tests.test_synthetic    # recovers known f0 and vibrato from synthetic tones
python -m tests.smoke_pipeline    # full pipeline on a synthetic VocalSet-shaped tree
```

## Repository layout

```
config.py                  paths, audio settings, labels, plot style
run_all.py                 end-to-end driver
src/data_loader.py         VocalSet discovery, singer and technique labels
src/features/              pitch.py, vibrato.py, resonance.py, breath.py, extract.py
src/model.py               feature matrix, singer-grouped CV, models
src/explain.py             SHAP (global, per class, per clip)
src/figures.py             base figures
tools/make_subset.py       builds the VocalSet subset used in the paper
tools/revision_*.py        statistical, robustness, and validation analyses
tools/fig_framework.py     framework diagram (Figure 1)
tests/                     synthetic validation and smoke test
outputs/                   results used in the paper (see below)
```

## Included results (`outputs/`)

| File | Content |
|---|---|
| `features.csv` | 31 features per clip with technique, singer, gender, and clip path (relative to the VocalSet folder) |
| `cv_results.json`, `tables/*.csv` | speaker-independent scores, confusion matrix, per-class metrics, SHAP importance |
| `shap_values.npz` | SHAP values of the final model |
| `revision/revision_stats.json` | singer-to-fold assignment, nested-CV selections, bootstrap intervals, tests, grouped SHAP |
| `revision/*.json`, `revision/*.csv` | vibrato validation, robustness, LOSO, focal loss, pitch-tracker comparison |
| `figures/` | the nine figures of the paper (PDF and PNG) |

## Method notes

- Evaluation is speaker-independent (`GroupKFold` by singer), so no singer
  appears in both training and test folds.
- Pitch, jitter, shimmer, HNR, and formants use Praat via parselmouth; the
  vibrato and spectral descriptors use custom DSP and librosa.
- XGBoost uses multithreaded histogram construction, so repeated runs can
  differ in the third decimal.
- Figures use Times New Roman when it is installed and a metric-compatible
  serif font otherwise.

## Citation

If you use this code, please cite the paper (citation details will be added
on publication).
