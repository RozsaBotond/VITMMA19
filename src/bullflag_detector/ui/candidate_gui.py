"""bullflag_detector.ui.candidate_gui

This module previously contained a matplotlib GUI labeler.

It has been intentionally removed per project requirements.
Use `bullflag-export-labelstudio` to generate Label Studio import JSONs from raw CSVs.
"""


def main() -> None:
    raise SystemExit(
        "The GUI labeler has been removed. "
        "Use 'bullflag-export-labelstudio' to export candidates for Label Studio."
    )


def main() -> None:
    import argparse
    import os as _os
    import matplotlib

    parser = argparse.ArgumentParser(description="GUI for labeling detected flag candidates.")
    parser.add_argument("csv", help="Path to OHLC CSV")
    parser.add_argument("--out", default="data/candidate_labels/candidates.json", help="Output JSON path")
    parser.add_argument("--plots", default="data/candidate_plots", help="Directory to store rendered images")
    parser.add_argument("--tui", action="store_true", help="Force Textual TUI (headless-friendly)")
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

    # Auto-fallback to TUI when a GUI can't be shown.
    backend = matplotlib.get_backend().lower()
    gui_backend = any(k in backend for k in ("tk", "qt", "wx", "macosx"))
    has_display = bool(_os.environ.get("DISPLAY"))

    if (not args.tui) and (not gui_backend or not has_display):
        # Give an actionable explanation. This commonly happens when matplotlib is
        # configured for a non-interactive backend (Agg) or when DISPLAY isn't set.
        reasons: list[str] = []
        if not has_display:
            reasons.append("$DISPLAY is not set")
        if not gui_backend:
            reasons.append(f"matplotlib backend is '{matplotlib.get_backend()}' (non-interactive)")
        reason_txt = "; ".join(reasons) if reasons else "unknown"
        print(
            "[bullflag-label] GUI can't be shown; falling back to TUI (headless mode).\n"
            f"Reason: {reason_txt}.\n\n"
            "To run the GUI, you need an interactive backend + a display. Options:\n"
            "  • If you're on SSH: use X-forwarding (ssh -X / -Y) so $DISPLAY is set.\n"
            "  • Install a GUI backend and force it, e.g.:\n"
            "      MPLBACKEND=TkAgg   (requires python-tk / tkinter)\n"
            "      MPLBACKEND=Qt5Agg  (requires PyQt5/PySide6)\n"
            "  • Or keep using the TUI: add --tui.\n"
        )

    if args.tui or (not gui_backend) or (not has_display):
        from .candidate_tui import CandidateTuiApp

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
        return

    app = CandidateLabelerMPL(
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
