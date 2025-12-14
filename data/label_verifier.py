"""
Label Verifier TUI

Interactive TUI application for verifying bull/bear flag pattern labels.
Shows the flag pattern with handle and surrounding context for visual verification.

Usage:
    uv run python label_verifier.py <labels.json> <data_dir>
    uv run python main.py verify <labels.json> <data_dir>
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Button, Label, 
    ListView, ListItem, DataTable, TextArea
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding
from textual.screen import Screen
from textual import events

# Import our modules
try:
    from label_parser import LabeledSegment, parse_label_studio_export
    from segment_extractor import load_timeseries_csv, extract_segment_from_df
except ImportError:
    # Add parent directory to path
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    if DATA_DIR not in sys.path:
        sys.path.insert(0, DATA_DIR)
    from label_parser import LabeledSegment, parse_label_studio_export
    from segment_extractor import load_timeseries_csv, extract_segment_from_df


@dataclass
class VerificationResult:
    """Result of label verification."""
    segment_idx: int
    original_label: str
    verified: bool = False
    correct: Optional[bool] = None
    new_label: Optional[str] = None
    notes: str = ""


@dataclass
class SegmentWithContext:
    """A segment with surrounding context for verification."""
    segment: LabeledSegment
    idx: int
    
    # Context data (before, flag, handle, after)
    context_before: Optional[np.ndarray] = None
    handle_data: Optional[np.ndarray] = None  # The pole/handle
    flag_data: Optional[np.ndarray] = None    # The flag pattern
    context_after: Optional[np.ndarray] = None
    
    # Full data for visualization
    full_data: Optional[np.ndarray] = None
    full_timestamps: Optional[List[str]] = None
    
    # Indices within full data
    flag_start_idx: int = 0
    flag_end_idx: int = 0
    handle_start_idx: int = 0
    
    # Verification
    verification: Optional[VerificationResult] = None


def extract_segment_with_context(
    segment: LabeledSegment,
    idx: int,
    data_dir: str,
    context_bars: int = 50,
    handle_bars: int = 20
) -> SegmentWithContext:
    """
    Extract a labeled segment along with surrounding context.
    
    Args:
        segment: The labeled segment
        idx: Index of the segment
        data_dir: Directory containing CSV files
        context_bars: Number of bars before/after to include
        handle_bars: Estimated handle/pole size
        
    Returns:
        SegmentWithContext with all data loaded
    """
    result = SegmentWithContext(segment=segment, idx=idx)
    
    # Load the CSV
    csv_path = os.path.join(data_dir, segment.csv_path)
    if not os.path.exists(csv_path):
        return result
    
    try:
        df = load_timeseries_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return result
    
    # Parse segment timestamps
    try:
        start_dt = pd.to_datetime(segment.start)
        end_dt = pd.to_datetime(segment.end)
    except Exception:
        return result
    
    # Find indices in DataFrame
    start_idx = df.index.searchsorted(start_dt)
    end_idx = df.index.searchsorted(end_dt)
    
    if start_idx >= len(df) or end_idx > len(df):
        return result
    
    # Calculate context range
    context_start_idx = max(0, start_idx - context_bars - handle_bars)
    context_end_idx = min(len(df), end_idx + context_bars)
    
    # Extract full context data
    context_df = df.iloc[context_start_idx:context_end_idx]
    result.full_data = context_df[['open', 'high', 'low', 'close']].values
    result.full_timestamps = [str(ts) for ts in context_df.index]
    
    # Calculate relative indices within full_data
    result.flag_start_idx = start_idx - context_start_idx
    result.flag_end_idx = end_idx - context_start_idx
    result.handle_start_idx = max(0, result.flag_start_idx - handle_bars)
    
    # Extract individual parts
    result.context_before = result.full_data[:result.handle_start_idx] if result.handle_start_idx > 0 else None
    result.handle_data = result.full_data[result.handle_start_idx:result.flag_start_idx]
    result.flag_data = result.full_data[result.flag_start_idx:result.flag_end_idx]
    result.context_after = result.full_data[result.flag_end_idx:] if result.flag_end_idx < len(result.full_data) else None
    
    return result


def render_candlestick_ascii(
    data: np.ndarray,
    width: int = 80,
    height: int = 20,
    flag_start: int = -1,
    flag_end: int = -1,
    handle_start: int = -1
) -> str:
    """
    Render OHLC data as ASCII candlestick chart.
    
    Args:
        data: OHLC data array (n, 4)
        width: Chart width in characters
        height: Chart height in characters
        flag_start: Start index of flag region (for highlighting)
        flag_end: End index of flag region
        handle_start: Start index of handle region
        
    Returns:
        ASCII string representation of the chart
    """
    if data is None or len(data) == 0:
        return "No data available"
    
    n_bars = len(data)
    
    # Calculate price range
    all_highs = data[:, 1]
    all_lows = data[:, 2]
    price_min = np.min(all_lows)
    price_max = np.max(all_highs)
    price_range = price_max - price_min
    
    if price_range == 0:
        price_range = 1
    
    # Create chart grid
    chart = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map bars to x positions
    bar_width = max(1, width // n_bars)
    
    def price_to_y(price):
        return height - 1 - int((price - price_min) / price_range * (height - 1))
    
    for i, bar in enumerate(data):
        x = min(i * bar_width + bar_width // 2, width - 1)
        open_p, high, low, close = bar
        
        y_high = price_to_y(high)
        y_low = price_to_y(low)
        y_open = price_to_y(open_p)
        y_close = price_to_y(close)
        
        # Determine character based on region
        if handle_start <= i < flag_start:
            # Handle region
            bull_char, bear_char, wick_char = 'H', 'h', '|'
        elif flag_start <= i < flag_end:
            # Flag region
            bull_char, bear_char, wick_char = 'F', 'f', ':'
        else:
            # Context
            bull_char, bear_char, wick_char = '█', '░', '│'
        
        # Draw wick
        for y in range(y_high, y_low + 1):
            if 0 <= y < height and 0 <= x < width:
                chart[y][x] = wick_char
        
        # Draw body
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)
        
        for y in range(body_top, body_bottom + 1):
            if 0 <= y < height and 0 <= x < width:
                if close >= open_p:
                    chart[y][x] = bull_char
                else:
                    chart[y][x] = bear_char
    
    # Add price labels on the right
    lines = []
    for i, row in enumerate(chart):
        price = price_max - (i / (height - 1)) * price_range
        line = ''.join(row)
        if i == 0 or i == height - 1 or i == height // 2:
            lines.append(f"{line} {price:.2f}")
        else:
            lines.append(line)
    
    return '\n'.join(lines)


def get_label_color(label: str) -> str:
    """Get rich color for a label."""
    label_lower = label.lower()
    if 'bullish' in label_lower:
        return "green"
    elif 'bearish' in label_lower:
        return "red"
    return "white"


class SegmentView(Static):
    """Widget to display a single segment with context."""
    
    def __init__(self, segment_ctx: Optional[SegmentWithContext] = None, **kwargs):
        super().__init__(**kwargs)
        self.segment_ctx = segment_ctx
    
    def update_segment(self, segment_ctx: SegmentWithContext):
        """Update the displayed segment."""
        self.segment_ctx = segment_ctx
        self.refresh()
    
    def render(self) -> str:
        if self.segment_ctx is None or self.segment_ctx.full_data is None:
            return "No segment data loaded"
        
        ctx = self.segment_ctx
        seg = ctx.segment
        
        # Build the display
        lines = []
        lines.append(f"[bold]Segment #{ctx.idx + 1}[/bold]")
        lines.append(f"[bold]Label: [{get_label_color(seg.label)}]{seg.label}[/][/bold]")
        lines.append(f"File: {seg.csv_path}")
        lines.append(f"Start: {seg.start}")
        lines.append(f"End: {seg.end}")
        lines.append("")
        
        # Statistics
        if ctx.flag_data is not None and len(ctx.flag_data) > 0:
            flag_high = np.max(ctx.flag_data[:, 1])
            flag_low = np.min(ctx.flag_data[:, 2])
            flag_range = flag_high - flag_low
            lines.append(f"[cyan]Flag bars: {len(ctx.flag_data)}, Range: {flag_range:.2f} ({flag_low:.2f} - {flag_high:.2f})[/cyan]")
        
        if ctx.handle_data is not None and len(ctx.handle_data) > 0:
            handle_high = np.max(ctx.handle_data[:, 1])
            handle_low = np.min(ctx.handle_data[:, 2])
            handle_open = ctx.handle_data[0, 0]
            handle_close = ctx.handle_data[-1, 3]
            handle_change = handle_close - handle_open
            direction = "▲" if handle_change > 0 else "▼"
            lines.append(f"[yellow]Handle bars: {len(ctx.handle_data)}, Change: {direction} {abs(handle_change):.2f}[/yellow]")
        
        lines.append("")
        lines.append("[bold]Chart[/bold] (H=Handle, F=Flag, █/░=Context)")
        lines.append("─" * 80)
        
        # Render ASCII chart
        chart = render_candlestick_ascii(
            ctx.full_data,
            width=75,
            height=18,
            flag_start=ctx.flag_start_idx,
            flag_end=ctx.flag_end_idx,
            handle_start=ctx.handle_start_idx
        )
        lines.append(chart)
        
        lines.append("─" * 80)
        
        # Verification status
        if ctx.verification:
            v = ctx.verification
            if v.verified:
                status = "[green]✓ Verified[/green]" if v.correct else "[red]✗ Incorrect[/red]"
                lines.append(f"Status: {status}")
                if v.new_label:
                    lines.append(f"New Label: {v.new_label}")
                if v.notes:
                    lines.append(f"Notes: {v.notes}")
        else:
            lines.append("[dim]Not yet verified (press Y/N/E)[/dim]")
        
        return '\n'.join(lines)


class LabelListItem(ListItem):
    """List item for a labeled segment."""
    
    def __init__(self, segment_ctx: SegmentWithContext, **kwargs):
        super().__init__(**kwargs)
        self.segment_ctx = segment_ctx
    
    def compose(self) -> ComposeResult:
        seg = self.segment_ctx.segment
        idx = self.segment_ctx.idx
        
        # Status indicator
        if self.segment_ctx.verification and self.segment_ctx.verification.verified:
            if self.segment_ctx.verification.correct:
                status = "✓"
            else:
                status = "✗"
        else:
            status = "○"
        
        color = get_label_color(seg.label)
        yield Label(f"{status} {idx+1:3d}. [{color}]{seg.label}[/]")


class VerifierApp(App):
    """TUI Application for verifying labels."""
    
    CSS = """
    #main-container {
        layout: horizontal;
    }
    
    #sidebar {
        width: 35;
        border: solid green;
        height: 100%;
    }
    
    #content {
        width: 1fr;
        border: solid blue;
        padding: 1;
    }
    
    ListView {
        height: 100%;
    }
    
    SegmentView {
        height: 100%;
    }
    
    #status-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        padding: 1;
    }
    
    #help-text {
        text-style: dim;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("y", "mark_correct", "Mark Correct"),
        Binding("n", "mark_incorrect", "Mark Incorrect"),
        Binding("e", "edit_label", "Edit Label"),
        Binding("s", "save", "Save Results"),
        Binding("j", "next", "Next"),
        Binding("k", "prev", "Previous"),
        Binding("down", "next", "Next", show=False),
        Binding("up", "prev", "Previous", show=False),
        Binding("g", "goto", "Go to #"),
        Binding("f", "filter", "Filter"),
        Binding("?", "help", "Help"),
    ]
    
    def __init__(
        self, 
        labels_path: str, 
        data_dir: str,
        output_path: Optional[str] = None,
        context_bars: int = 50,
        handle_bars: int = 20
    ):
        super().__init__()
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.output_path = output_path or labels_path.replace('.json', '_verified.json')
        self.context_bars = context_bars
        self.handle_bars = handle_bars
        
        # Data
        self.segments: List[SegmentWithContext] = []
        self.current_idx = 0
        self.filter_mode = "all"  # all, unverified, correct, incorrect
        
        # Load existing verification if present
        self.verification_results: Dict[int, VerificationResult] = {}
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="main-container"):
            with Vertical(id="sidebar"):
                yield Label("[bold]Segments[/bold]", id="sidebar-title")
                yield ListView(id="segment-list")
            
            with Vertical(id="content"):
                yield SegmentView(id="segment-view")
        
        yield Static(
            "[bold]Keys:[/bold] Y=Correct N=Incorrect E=Edit S=Save J/K=Nav Q=Quit ?=Help",
            id="status-bar"
        )
        yield Footer()
    
    async def on_mount(self) -> None:
        """Load data when app starts."""
        self.title = "Label Verifier"
        self.sub_title = f"Verifying: {os.path.basename(self.labels_path)}"
        
        # Load labels
        await self.load_segments()
        
        # Populate list
        await self.refresh_list()
        
        # Show first segment
        if self.segments:
            await self.show_segment(0)
    
    async def load_segments(self) -> None:
        """Load all segments with context."""
        segments = parse_label_studio_export(self.labels_path)
        
        self.segments = []
        for idx, seg in enumerate(segments):
            seg_ctx = extract_segment_with_context(
                seg, idx, self.data_dir,
                self.context_bars, self.handle_bars
            )
            self.segments.append(seg_ctx)
        
        # Load existing verification results
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    saved = json.load(f)
                for item in saved.get('verifications', []):
                    idx = item['segment_idx']
                    if idx < len(self.segments):
                        v = VerificationResult(
                            segment_idx=idx,
                            original_label=item['original_label'],
                            verified=item['verified'],
                            correct=item.get('correct'),
                            new_label=item.get('new_label'),
                            notes=item.get('notes', '')
                        )
                        self.verification_results[idx] = v
                        self.segments[idx].verification = v
            except Exception as e:
                self.notify(f"Could not load saved results: {e}", severity="warning")
    
    async def refresh_list(self) -> None:
        """Refresh the segment list."""
        list_view = self.query_one("#segment-list", ListView)
        await list_view.clear()
        
        for seg_ctx in self.get_filtered_segments():
            item = LabelListItem(seg_ctx)
            await list_view.append(item)
    
    def get_filtered_segments(self) -> List[SegmentWithContext]:
        """Get segments based on current filter."""
        if self.filter_mode == "all":
            return self.segments
        elif self.filter_mode == "unverified":
            return [s for s in self.segments if not s.verification or not s.verification.verified]
        elif self.filter_mode == "correct":
            return [s for s in self.segments if s.verification and s.verification.correct]
        elif self.filter_mode == "incorrect":
            return [s for s in self.segments if s.verification and s.verification.correct == False]
        return self.segments
    
    async def show_segment(self, idx: int) -> None:
        """Show a specific segment."""
        if 0 <= idx < len(self.segments):
            self.current_idx = idx
            seg_ctx = self.segments[idx]
            
            segment_view = self.query_one("#segment-view", SegmentView)
            segment_view.update_segment(seg_ctx)
            
            # Update subtitle
            verified = sum(1 for s in self.segments if s.verification and s.verification.verified)
            self.sub_title = f"{idx + 1}/{len(self.segments)} | Verified: {verified}"
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if isinstance(event.item, LabelListItem):
            await self.show_segment(event.item.segment_ctx.idx)
    
    async def action_next(self) -> None:
        """Go to next segment."""
        if self.current_idx < len(self.segments) - 1:
            await self.show_segment(self.current_idx + 1)
    
    async def action_prev(self) -> None:
        """Go to previous segment."""
        if self.current_idx > 0:
            await self.show_segment(self.current_idx - 1)
    
    async def action_mark_correct(self) -> None:
        """Mark current segment as correctly labeled."""
        if 0 <= self.current_idx < len(self.segments):
            seg_ctx = self.segments[self.current_idx]
            v = VerificationResult(
                segment_idx=self.current_idx,
                original_label=seg_ctx.segment.label,
                verified=True,
                correct=True
            )
            self.verification_results[self.current_idx] = v
            seg_ctx.verification = v
            
            await self.refresh_list()
            await self.show_segment(self.current_idx)
            self.notify(f"Marked #{self.current_idx + 1} as CORRECT", severity="information")
            
            # Auto-advance to next
            await self.action_next()
    
    async def action_mark_incorrect(self) -> None:
        """Mark current segment as incorrectly labeled."""
        if 0 <= self.current_idx < len(self.segments):
            seg_ctx = self.segments[self.current_idx]
            v = VerificationResult(
                segment_idx=self.current_idx,
                original_label=seg_ctx.segment.label,
                verified=True,
                correct=False
            )
            self.verification_results[self.current_idx] = v
            seg_ctx.verification = v
            
            await self.refresh_list()
            await self.show_segment(self.current_idx)
            self.notify(f"Marked #{self.current_idx + 1} as INCORRECT", severity="warning")
    
    async def action_edit_label(self) -> None:
        """Edit the label for current segment."""
        self.notify("Edit mode: Enter new label in terminal", severity="information")
        # For simplicity, we'll use a basic approach
        # In a full implementation, you'd show a modal dialog
        labels = [
            "Bullish Normal", "Bullish Wedge", "Bullish Pennant",
            "Bearish Normal", "Bearish Wedge", "Bearish Pennant"
        ]
        
        # Show available labels
        msg = "Available labels:\n" + "\n".join(f"  {i+1}. {l}" for i, l in enumerate(labels))
        self.notify(msg, timeout=10)
    
    async def action_save(self) -> None:
        """Save verification results."""
        output = {
            "labels_file": self.labels_path,
            "data_dir": self.data_dir,
            "verified_at": datetime.now().isoformat(),
            "total_segments": len(self.segments),
            "verified_count": sum(1 for v in self.verification_results.values() if v.verified),
            "correct_count": sum(1 for v in self.verification_results.values() if v.correct),
            "incorrect_count": sum(1 for v in self.verification_results.values() if v.correct == False),
            "verifications": [
                {
                    "segment_idx": v.segment_idx,
                    "original_label": v.original_label,
                    "verified": v.verified,
                    "correct": v.correct,
                    "new_label": v.new_label,
                    "notes": v.notes
                }
                for v in self.verification_results.values()
            ]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.notify(f"Saved to {self.output_path}", severity="information")
    
    async def action_filter(self) -> None:
        """Cycle through filter modes."""
        modes = ["all", "unverified", "correct", "incorrect"]
        current_idx = modes.index(self.filter_mode)
        self.filter_mode = modes[(current_idx + 1) % len(modes)]
        
        await self.refresh_list()
        self.notify(f"Filter: {self.filter_mode}", severity="information")
    
    async def action_goto(self) -> None:
        """Go to a specific segment number."""
        self.notify("Go to: Use number keys and press Enter", severity="information")
    
    async def action_help(self) -> None:
        """Show help."""
        help_text = """
[bold]Label Verifier Help[/bold]

[cyan]Navigation:[/cyan]
  J / ↓       Next segment
  K / ↑       Previous segment
  G           Go to segment #

[cyan]Verification:[/cyan]
  Y           Mark as CORRECT
  N           Mark as INCORRECT  
  E           Edit/change label

[cyan]Chart Legend:[/cyan]
  H / h       Handle (pole) - bullish/bearish
  F / f       Flag region - bullish/bearish
  █ / ░       Context - bullish/bearish
  │ / : / |   Wicks

[cyan]Other:[/cyan]
  S           Save results
  F           Filter (all/unverified/correct/incorrect)
  Q           Quit
  ?           This help
"""
        self.notify(help_text, timeout=15)


def run_verifier(
    labels_path: str,
    data_dir: str,
    output_path: Optional[str] = None,
    context_bars: int = 50,
    handle_bars: int = 20
):
    """Run the label verifier TUI."""
    app = VerifierApp(
        labels_path=labels_path,
        data_dir=data_dir,
        output_path=output_path,
        context_bars=context_bars,
        handle_bars=handle_bars
    )
    app.run()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: uv run python label_verifier.py <labels.json> <data_dir> [output.json]")
        print("\nArguments:")
        print("  labels.json  - Label Studio JSON export")
        print("  data_dir     - Directory containing CSV files")
        print("  output.json  - Output file for verification results (optional)")
        sys.exit(1)
    
    labels_path = sys.argv[1]
    data_dir = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_verifier(labels_path, data_dir, output_path)
