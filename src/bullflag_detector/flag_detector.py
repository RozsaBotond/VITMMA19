"""bullflag_detector.flag_detector

Candidate discovery + verification helper.

This module is intended to help you *add more labels* by:
1) scanning OHLC time series for *candidate* flag-like segments (simple heuristic)
2) rendering a verification image that includes *context before/after* the candidate
3) saving accepted candidates into a Label Studio-compatible JSON you can merge/upload.

It intentionally does **not** aim to be a perfect detector; it’s a bootstrap tool to
speed up manual labeling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


VALID_LABELS = [
    "Bullish Normal",
    "Bullish Wedge",
    "Bullish Pennant",
    "Bearish Normal",
    "Bearish Wedge",
    "Bearish Pennant",
]


@dataclass(frozen=True)
class CandidateFlag:
    """A proposed flag segment in index-space + timestamps."""

    pole_start_idx: int
    pole_end_idx: int
    flag_start_idx: int
    flag_end_idx: int

    # Convenience metadata
    pole_return_pct: float
    direction_hint: str  # "Bullish" | "Bearish"


def find_raw_data_files(data_dir: str) -> List[str]:
    """Find all raw CSV data files under data/raw_data."""
    raw_data_path = os.path.join(data_dir, "raw_data")
    if not os.path.isdir(raw_data_path):
        return []

    return sorted(
        os.path.join(raw_data_path, f)
        for f in os.listdir(raw_data_path)
        if f.lower().endswith(".csv")
    )


def _coerce_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime index in UTC when possible."""
    if df.index.tz is None:
        # Label Studio exports show timestamps with Z; treat input as UTC.
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)
    return df


def load_ohlc_csv(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Load an OHLC CSV into a DataFrame indexed by timestamp.

    Accepts both:
    - timestamp,open,high,low,close (timestamp can be unix ms or ISO string)
    - Date/Open/High/Low/Close variants
    """
    df = pd.read_csv(file_path, sep=sep)

    column_mapping = {
        "Date": "timestamp",
        "date": "timestamp",
        "Timestamp": "timestamp",
        "timestamp": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    }
    df.rename(columns=column_mapping, inplace=True)

    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column")

    # Try unix ms first, then pandas datetime parsing.
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df.set_index("timestamp", inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.sort_index(inplace=True)
    return _coerce_utc_index(df)


def detect_candidate_flags(
    df: pd.DataFrame,
    pole_window: int = 15,
    pole_threshold_pct: float = 5.0,
    flag_window: int = 30,
    stride: int = 3,
) -> List[CandidateFlag]:
    """Very simple heuristic:
    - find windows where close moves more than pole_threshold_pct in pole_window bars
    - propose the next flag_window bars as the "flag".

    Returns de-duplicated candidates.
    """
    if len(df) < max(pole_window + 2, flag_window + 2):
        return []

    close = df["close"].to_numpy(dtype=float)

    candidates: List[CandidateFlag] = []

    # compute pole returns for end indices i
    for pole_end in range(pole_window, len(close), stride):
        pole_start = pole_end - pole_window
        p0 = close[pole_start]
        p1 = close[pole_end]
        if p0 == 0:
            continue
        ret = (p1 - p0) / p0 * 100.0
        if abs(ret) < pole_threshold_pct:
            continue

        flag_start = pole_end
        flag_end = min(len(close), flag_start + flag_window)
        if flag_end - flag_start < max(5, flag_window // 4):
            continue

        direction_hint = "Bullish" if ret > 0 else "Bearish"
        candidates.append(
            CandidateFlag(
                pole_start_idx=pole_start,
                pole_end_idx=pole_end,
                flag_start_idx=flag_start,
                flag_end_idx=flag_end,
                pole_return_pct=float(ret),
                direction_hint=direction_hint,
            )
        )

    # De-dup: keep best by absolute return per overlapping region.
    candidates.sort(key=lambda c: abs(c.pole_return_pct), reverse=True)
    kept: List[CandidateFlag] = []
    occupied: List[Tuple[int, int]] = []
    for c in candidates:
        overlap = False
        for a, b in occupied:
            if not (c.flag_end_idx <= a or c.flag_start_idx >= b):
                overlap = True
                break
        if overlap:
            continue
        kept.append(c)
        occupied.append((c.flag_start_idx, c.flag_end_idx))

    kept.sort(key=lambda c: c.flag_start_idx)
    return kept


def render_candidate_image(
    df: pd.DataFrame,
    candidate: CandidateFlag,
    output_path: str,
    context_bars: int = 50,
    handle_bars: int = 20,
    title_extra: str = "",
) -> None:
    """Render a candlestick image showing context/handle/flag regions."""
    # Determine context window
    context_start = max(0, candidate.flag_start_idx - context_bars - handle_bars)
    context_end = min(len(df), candidate.flag_end_idx + context_bars)
    context_df = df.iloc[context_start:context_end].copy()

    flag_start = candidate.flag_start_idx - context_start
    flag_end = candidate.flag_end_idx - context_start
    handle_start = max(0, flag_start - handle_bars)

    fig, ax = plt.subplots(figsize=(15, 7))
    n = len(context_df)
    if n == 0:
        plt.close(fig)
        return

    # Plot each candle
    for i, (_, row) in enumerate(context_df.iterrows()):
        open_p, high, low, close = (
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )

        if handle_start <= i < flag_start:
            color = "orange" if close >= open_p else "darkorange"
            alpha = 1.0
        elif flag_start <= i < flag_end:
            color = "lime" if close >= open_p else "red"
            alpha = 1.0
        else:
            color = "green" if close >= open_p else "darkred"
            alpha = 0.35

        ax.plot([i, i], [low, high], color="black", linewidth=0.5, alpha=alpha)

        body_bottom = min(open_p, close)
        body_height = abs(close - open_p)
        if body_height < 1e-6:
            body_height = 1e-6

        rect = Rectangle(
            (i - 0.3, body_bottom),
            0.6,
            body_height,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha,
        )
        ax.add_patch(rect)

    # Highlight regions
    if handle_start < flag_start:
        ax.axvspan(handle_start - 0.5, flag_start - 0.5, alpha=0.12, color="orange")
    ax.axvspan(flag_start - 0.5, flag_end - 0.5, alpha=0.12, color="blue")

    price_min = float(context_df["low"].min())
    price_max = float(context_df["high"].max())
    padding = (price_max - price_min) * 0.1 if price_max > price_min else 1.0
    ax.set_xlim(-1, n)
    ax.set_ylim(price_min - padding, price_max + padding)
    ax.grid(True, alpha=0.25)

    # Put a helpful title: include timestamps of the proposed flag section.
    start_ts = context_df.index[min(flag_start, n - 1)]
    end_ts = context_df.index[min(max(flag_end - 1, 0), n - 1)]
    title = (
        f"Candidate {candidate.direction_hint} flag | pole={candidate.pole_return_pct:.2f}% | "
        f"flag: {start_ts.isoformat()} → {end_ts.isoformat()}"
    )
    if title_extra:
        title += f" | {title_extra}"
    ax.set_title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _iso_z(ts: pd.Timestamp) -> str:
    # Label Studio export uses .000Z; use Z-suffixed UTC.
    ts = ts.tz_convert(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _ls_time(ts: pd.Timestamp) -> str:
    """Match this project's existing Label Studio export time format.

    `data/labels.json` stores timestamps like: "2006-06-09 22:00" (no timezone, space separator).
    We export in the same shape for compatibility.
    """
    try:
        ts = ts.tz_convert(timezone.utc)
    except Exception:
        pass
    return ts.strftime("%Y-%m-%d %H:%M")


def write_labelstudio_candidates_json(
    output_json_path: str,
    source_csv_filename: str,
    candidates: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
    uploader_prefix: str = "",
) -> None:
    """Write a Label Studio-style export JSON with timeserieslabels.

    We only emit the minimal structure our `data/label_parser.py` can read.
    """
    tasks: List[Dict[str, Any]] = []
    for i, (start_ts, end_ts, label) in enumerate(candidates):
        tasks.append(
            {
                "file_upload": source_csv_filename,
                "data": {
                    # Keep it close to Label Studio exports: they often store a URL-ish path.
                    "csv": f"{uploader_prefix}{source_csv_filename}",
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "id": f"cand_{i}",
                                "type": "timeserieslabels",
                                "from_name": "label",
                                "to_name": "ts",
                                "value": {
                                    "start": _iso_z(start_ts),
                                    "end": _iso_z(end_ts),
                                    "instant": False,
                                    "timeserieslabels": [label],
                                },
                            }
                        ]
                    }
                ],
            }
        )

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def write_labelstudio_potential_flags_json(
    output_json_path: str,
    source_csv_filename: str,
    candidates: List[Tuple[pd.Timestamp, pd.Timestamp]],
    uploader_prefix: str = "",
    from_name: str = "label",
    to_name: str = "ts",
    start_task_id: int = 1,
) -> None:
    """Write Label Studio import JSON for *potential* (unlabeled) flag regions.

    Goal: generate tasks you can import into Label Studio, where each task contains
    a pre-created timespan region (start/end) but **no label** selected yet.

    Notes:
    - Label Studio supports imports without annotations; however, providing an
      empty-labeled "timeserieslabels" region makes the candidate span visible
      immediately in the UI (you just pick the class).
    - The exact names (from_name/to_name) must match your Label Studio labeling
      config. Defaults are consistent with the existing exporter and parser.
    """

    tasks: List[Dict[str, Any]] = []
    for i, (start_ts, end_ts) in enumerate(candidates):
        tasks.append(
            {
                "id": start_task_id + i,
                "file_upload": source_csv_filename,
                "data": {"csv": f"{uploader_prefix}{source_csv_filename}"},
                "annotations": [
                    {
                        "result": [
                            {
                                "id": f"cand_{start_task_id + i}",
                                "type": "timeserieslabels",
                                "from_name": from_name,
                                "to_name": to_name,
                                "value": {
                                    "start": _ls_time(start_ts),
                                    "end": _ls_time(end_ts),
                                    "instant": False,
                                    # Intentionally empty: user will choose label in LS
                                    "timeserieslabels": [],
                                },
                            }
                        ]
                    }
                ],
            }
        )

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def _prompt_label(default_direction: str) -> Optional[str]:
    print("Valid labels:")
    for j, lab in enumerate(VALID_LABELS, start=1):
        hint = "" if default_direction not in lab else " (hint)"
        print(f"  {j}: {lab}{hint}")

    raw = input(f"Choose label [1-{len(VALID_LABELS)}] (empty to cancel): ").strip()
    if raw == "":
        return None
    try:
        idx = int(raw) - 1
    except ValueError:
        return None
    if 0 <= idx < len(VALID_LABELS):
        return VALID_LABELS[idx]
    return None


def interactive_label_candidates(
    csv_path: str,
    output_dir: str = "data/candidate_labels",
    plots_dir: str = "data/candidate_plots",
    max_candidates: int = 100,
    pole_window: int = 15,
    pole_threshold_pct: float = 5.0,
    flag_window: int = 30,
    context_bars: int = 50,
    handle_bars: int = 20,
) -> str:
    """Run candidate detection and an interactive prompt to accept/reject.

    Returns the path to the written JSON.
    """
    df = load_ohlc_csv(csv_path)
    cands = detect_candidate_flags(
        df,
        pole_window=pole_window,
        pole_threshold_pct=pole_threshold_pct,
        flag_window=flag_window,
    )[:max_candidates]

    if not cands:
        raise RuntimeError("No candidates found with current thresholds")

    accepted: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    source_csv_filename = os.path.basename(csv_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(cands, start=1):
        plot_path = os.path.join(plots_dir, f"{Path(source_csv_filename).stem}_cand_{i:03d}.png")
        render_candidate_image(
            df,
            c,
            plot_path,
            context_bars=context_bars,
            handle_bars=handle_bars,
            title_extra=f"{source_csv_filename} | {i}/{len(cands)}",
        )

        print("\n" + "=" * 70)
        print(f"Candidate {i}/{len(cands)} | {source_csv_filename}")
        print(f"Direction hint: {c.direction_hint} | pole return: {c.pole_return_pct:.2f}%")
        print(f"Open image: {os.path.abspath(plot_path)}")

        while True:
            action = input("Accept? [y=yes, n=no, q=quit]: ").strip().lower()
            if action in {"n", "no"}:
                break
            if action in {"q", "quit"}:
                # Stop early
                i = len(cands)
                break
            if action in {"y", "yes"}:
                label = _prompt_label(c.direction_hint)
                if not label:
                    print("No label selected; not saved.")
                    break

                start_ts = df.index[c.flag_start_idx]
                end_ts = df.index[min(c.flag_end_idx - 1, len(df.index) - 1)]
                accepted.append((start_ts, end_ts, label))
                print(f"Accepted: {label} | {_iso_z(start_ts)} → {_iso_z(end_ts)}")
                break

            print("Invalid input.")

        if action in {"q", "quit"}:
            break

    out_path = os.path.join(output_dir, f"{Path(source_csv_filename).stem}_candidates.json")
    write_labelstudio_candidates_json(
        out_path,
        source_csv_filename=source_csv_filename,
        candidates=accepted,
    )
    print(f"\nWrote {len(accepted)} accepted candidates to: {out_path}")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect candidate flags and help label them.")
    parser.add_argument("csv", help="Path to OHLC CSV")
    parser.add_argument("--out", default="data/candidate_labels", help="Output dir for candidate JSON")
    parser.add_argument("--plots", default="data/candidate_plots", help="Output dir for PNG plots")
    parser.add_argument("--max", type=int, default=100, help="Max candidates to review")
    parser.add_argument("--pole-window", type=int, default=15)
    parser.add_argument("--pole-pct", type=float, default=5.0)
    parser.add_argument("--flag-window", type=int, default=30)
    parser.add_argument("--context", type=int, default=50)
    parser.add_argument("--handle", type=int, default=20)
    args = parser.parse_args()

    interactive_label_candidates(
        args.csv,
        output_dir=args.out,
        plots_dir=args.plots,
        max_candidates=args.max,
        pole_window=args.pole_window,
        pole_threshold_pct=args.pole_pct,
        flag_window=args.flag_window,
        context_bars=args.context,
        handle_bars=args.handle,
    )


if __name__ == "__main__":
    main()
