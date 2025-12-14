# Bull-flag detector (VITMMA19)

Deep-learning project for detecting **bull flag** and **bear flag** patterns in financial time series.

## What’s in this repo

- `data/` – data prep, label parsing, segment extraction, verification tools
- `src/bullflag_detector/` – model + detection utilities
- `models/` – trained weights (if any)

## Labeling workflow

### 1) Verify existing Label Studio labels

You can verify an exported Label Studio JSON with either:

- a Textual TUI: `main.py verify ... --tui`
- a matplotlib viewer: `main.py verify ...`

### 2) Export *potential* flags for Label Studio (non-interactive)

The interactive GUI labeler has been removed.

Instead, use the batch exporter which:

- scans all CSVs under `data/raw_data/`
- detects *potential* flag candidates
- writes per-CSV **Label Studio import JSON** files (unlabeled regions)

Run:

```bash
uv run bullflag-export-labelstudio --data-dir data --out-dir data/labelstudio_import
```

Output example:

- `data/labelstudio_import/XAU_1h_data_limited_potential_flags.json`

Each task contains a proposed `start`/`end` span with `timeserieslabels: []` so you can select one of the 6 classes inside Label Studio.

## Notes

This candidate detector is heuristic on purpose. It’s meant to speed up *manual* labeling, not replace it.
