"""bullflag_detector.ui.candidate_tui

Headless-friendly candidate labeling TUI.

This is designed for environments where matplotlib GUI backends aren't available.
It still:
- detects flag candidates
- renders a candlestick PNG with context-before/after (so you can open it elsewhere)
- shows an ASCII candlestick chart in-terminal with context/handle/flag markers
- lets you quickly label with single keys

Run:
    uv run python -m bullflag_detector.ui.candidate_tui path/to/ohlc.csv
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from ..flag_detector import (
    VALID_LABELS,
    CandidateFlag,
    detect_candidate_flags,
    load_ohlc_csv,
    render_candidate_image,
    write_labelstudio_candidates_json,
)


def _render_ascii_candles(
    ohlc: pd.DataFrame,
    flag_start: int,
    flag_end: int,
    handle_start: int,
    width: int = 80,
    height: int = 18,
) -> str:
    """Lightweight ASCII candlestick rendering (derived from data/label_verifier.py).

    Uses Rich markup so regions are colored in Textual:
    - Handle: yellow/orange
    - Flag: cyan/blue
    - Context: dim
    - Bull bodies: green, Bear bodies: red
    """
    if ohlc is None or len(ohlc) == 0:
        return "No data"

    data = ohlc[["open", "high", "low", "close"]].to_numpy(dtype=float)
    n_bars = len(data)

    highs = data[:, 1]
    lows = data[:, 2]
    pmin = float(lows.min())
    pmax = float(highs.max())
    prange = pmax - pmin
    if prange == 0:
        prange = 1.0

    chart = [[" " for _ in range(width)] for _ in range(height)]
    bar_width = max(1, width // n_bars)

    def price_to_y(price: float) -> int:
        return height - 1 - int((price - pmin) / prange * (height - 1))

    for i, bar in enumerate(data):
        x = min(i * bar_width + bar_width // 2, width - 1)
        open_p, high, low, close = bar
        y_high = price_to_y(high)
        y_low = price_to_y(low)
        y_open = price_to_y(open_p)
        y_close = price_to_y(close)

        in_handle = handle_start <= i < flag_start
        in_flag = flag_start <= i < flag_end

        if in_handle:
            bull_char, bear_char, wick_char = "H", "h", "|"
        elif in_flag:
            bull_char, bear_char, wick_char = "F", "f", ":"
        else:
            bull_char, bear_char, wick_char = "█", "░", "│"

        for y in range(min(y_high, y_low), max(y_high, y_low) + 1):
            if 0 <= y < height:
                chart[y][x] = wick_char

        body_top = min(y_open, y_close)
        body_bot = max(y_open, y_close)
        body_char = bull_char if close >= open_p else bear_char
        for y in range(body_top, body_bot + 1):
            if 0 <= y < height:
                chart[y][x] = body_char

    def style_char(ch: str) -> str:
        # Body/region chars
        if ch in {"H", "h", "|"}:
            return f"[yellow]{ch}[/yellow]"
        if ch in {"F", "f", ":"}:
            return f"[cyan]{ch}[/cyan]"
        # bullish/bearish in context
        if ch == "█":
            return f"[green]{ch}[/green]"
        if ch == "░":
            return f"[red]{ch}[/red]"
        if ch == "│":
            return f"[dim]{ch}[/dim]"
        return ch

    lines: List[str] = []
    for r, row in enumerate(chart):
        line = "".join(style_char(c) for c in row)
        if r in {0, height // 2, height - 1}:
            price = pmax - (r / (height - 1)) * prange
            line += f"  [dim]{price:.2f}[/dim]"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class AcceptedLabel:
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    label: str


class CandidateTuiApp(App):
    """Textual TUI for candidate labeling."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "next", "Next"),
        Binding("k", "prev", "Prev"),
        Binding("s", "skip", "Skip"),
        Binding("w", "write", "Save"),
        # Labels 1-6
        Binding("1", "label_1", "1"),
        Binding("2", "label_2", "2"),
        Binding("3", "label_3", "3"),
        Binding("4", "label_4", "4"),
        Binding("5", "label_5", "5"),
        Binding("6", "label_6", "6"),
    ]

    def __init__(
        self,
        csv_path: str,
        output_json_path: str,
        plots_dir: str = "data/candidate_plots",
        max_candidates: int = 200,
        pole_window: int = 15,
        pole_threshold_pct: float = 5.0,
        flag_window: int = 30,
        context_bars: int = 50,
        handle_bars: int = 20,
        autosave_every: int = 1,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.source_csv_filename = os.path.basename(csv_path)
        self.output_json_path = output_json_path
        self.plots_dir = plots_dir
        self.autosave_every = max(1, autosave_every)

        self.df = load_ohlc_csv(csv_path)
        self.candidates: List[CandidateFlag] = detect_candidate_flags(
            self.df,
            pole_window=pole_window,
            pole_threshold_pct=pole_threshold_pct,
            flag_window=flag_window,
        )[:max_candidates]
        if not self.candidates:
            raise RuntimeError("No candidates found with current thresholds")

        self.context_bars = context_bars
        self.handle_bars = handle_bars

        self.accepted: List[AcceptedLabel] = []
        self.current_idx = 0
        self._load_existing_output()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                yield Static("", id="meta")
                yield Static("", id="help")
            with Vertical(id="right"):
                yield Static("", id="chart")
        yield Footer()

    def _load_existing_output(self) -> None:
        if not os.path.exists(self.output_json_path):
            return
        try:
            with open(self.output_json_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            for task in tasks:
                for ann in task.get("annotations", []):
                    for res in ann.get("result", []):
                        val = res.get("value", {})
                        labels = val.get("timeserieslabels", [])
                        if not labels:
                            continue
                        start = pd.to_datetime(val.get("start", ""), utc=True, errors="coerce")
                        end = pd.to_datetime(val.get("end", ""), utc=True, errors="coerce")
                        if pd.isna(start) or pd.isna(end):
                            continue
                        self.accepted.append(AcceptedLabel(start_ts=start, end_ts=end, label=labels[0]))
        except Exception:
            return

    def _save(self) -> None:
        candidates = [(a.start_ts, a.end_ts, a.label) for a in self.accepted]
        write_labelstudio_candidates_json(
            self.output_json_path,
            source_csv_filename=self.source_csv_filename,
            candidates=candidates,
        )

    def _current_plot_path(self) -> str:
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(self.source_csv_filename).stem
        return os.path.join(self.plots_dir, f"{stem}_cand_{self.current_idx + 1:03d}.png")

    def _render_current(self) -> None:
        cand = self.candidates[self.current_idx]

        # Render PNG (open it outside the TUI if needed)
        plot_path = self._current_plot_path()
        render_candidate_image(
            self.df,
            cand,
            plot_path,
            context_bars=0,
            handle_bars=self.handle_bars,
            title_extra=f"{self.source_csv_filename} | {self.current_idx + 1}/{len(self.candidates)}",
        )

        # ASCII view: only show pole + (optional) handle + flag (no extra context)
        show_start = max(0, cand.pole_start_idx)
        show_end = min(len(self.df), cand.flag_end_idx)
        ctx_df = self.df.iloc[show_start:show_end].copy()
        flag_start = cand.flag_start_idx - show_start
        flag_end = cand.flag_end_idx - show_start
        handle_start = max(0, flag_start - self.handle_bars)

        meta = (
            f"[b]Candidate[/b] {self.current_idx + 1}/{len(self.candidates)}\n"
            f"CSV: {self.source_csv_filename}\n"
            f"Hint: {cand.direction_hint} | pole={cand.pole_return_pct:.2f}%\n"
            f"Accepted: {len(self.accepted)}\n"
            f"PNG: {plot_path}\n"
        )
        help_text = (
            "[b]Keys[/b]: j=next k=prev s=skip w=save q=quit\n"
            "Label: 1-6\n\n"
            "1 Bullish Normal\n"
            "2 Bullish Wedge\n"
            "3 Bullish Pennant\n"
            "4 Bearish Normal\n"
            "5 Bearish Wedge\n"
            "6 Bearish Pennant\n"
        )

        # Stretch the chart to available width.
        # Textual widget sizes become available after mount; we fall back safely.
        chart_widget = self.query_one("#chart", Static)
        # Leave some margin for the right-hand price labels and avoid overly tiny charts.
        chart_width = max(40, int((getattr(chart_widget, "size", None).width or 90) - 12))

        chart = _render_ascii_candles(
            ctx_df,
            flag_start=flag_start,
            flag_end=flag_end,
            handle_start=handle_start,
            width=chart_width,
            height=18,
        )
        chart_header = (
            "[b]Chart[/b] "
            "([yellow]H[/yellow]=handle, [cyan]F[/cyan]=flag, "
            "[green]█[/green]/[red]░[/red]=context bodies)\n"
            + ("-" * 90)
            + "\n"
        )

        self.query_one("#meta", Static).update(meta)
        self.query_one("#help", Static).update(help_text)
        self.query_one("#chart", Static).update(chart_header + chart)

    async def on_mount(self) -> None:
        self.title = "Candidate Labeler (TUI)"
        self.sub_title = os.path.basename(self.csv_path)
        self._render_current()

    async def on_resize(self) -> None:
        # Re-render so the ASCII chart stretches/shrinks with the terminal.
        self._render_current()

    def action_next(self) -> None:
        if self.current_idx < len(self.candidates) - 1:
            self.current_idx += 1
            self._render_current()

    def action_prev(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            self._render_current()

    def action_skip(self) -> None:
        self.action_next()

    def action_write(self) -> None:
        self._save()

    def _accept_label(self, label: str) -> None:
        cand = self.candidates[self.current_idx]
        start_ts = self.df.index[cand.flag_start_idx]
        end_ts = self.df.index[min(cand.flag_end_idx - 1, len(self.df.index) - 1)]
        self.accepted.append(AcceptedLabel(start_ts=start_ts, end_ts=end_ts, label=label))
        if len(self.accepted) % self.autosave_every == 0:
            self._save()
        self.action_next()

    def action_label_1(self) -> None:
        self._accept_label(VALID_LABELS[0])

    def action_label_2(self) -> None:
        self._accept_label(VALID_LABELS[1])

    def action_label_3(self) -> None:
        self._accept_label(VALID_LABELS[2])

    def action_label_4(self) -> None:
        self._accept_label(VALID_LABELS[3])

    def action_label_5(self) -> None:
        self._accept_label(VALID_LABELS[4])

    def action_label_6(self) -> None:
        self._accept_label(VALID_LABELS[5])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TUI for labeling detected flag candidates.")
    parser.add_argument("csv", help="Path to OHLC CSV")
    parser.add_argument("--out", default="data/candidate_labels/candidates.json", help="Output JSON path")
    parser.add_argument("--plots", default="data/candidate_plots", help="Directory to store rendered images")
    parser.add_argument("--max", type=int, default=200, help="Max candidates")
    parser.add_argument("--pole-window", type=int, default=15)
    parser.add_argument("--pole-pct", type=float, default=5.0)
    parser.add_argument("--flag-window", type=int, default=30)
    parser.add_argument("--context", type=int, default=50)
    parser.add_argument("--handle", type=int, default=20)
    parser.add_argument("--autosave", type=int, default=1, help="Autosave every N accepted labels")
    args = parser.parse_args()

    out_path = args.out
    if out_path.endswith(os.sep) or out_path.endswith("/"):
        out_path = os.path.join(out_path, f"{Path(args.csv).stem}_candidates.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    app = CandidateTuiApp(
        csv_path=args.csv,
        output_json_path=out_path,
        plots_dir=args.plots,
        max_candidates=args.max,
        pole_window=args.pole_window,
        pole_threshold_pct=args.pole_pct,
        flag_window=args.flag_window,
        context_bars=args.context,
        handle_bars=args.handle,
        autosave_every=args.autosave,
    )
    app.run()


if __name__ == "__main__":
    main()
