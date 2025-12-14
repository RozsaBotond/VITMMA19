"""
Label Verifier - Matplotlib Version

Interactive matplotlib-based tool for verifying bull/bear flag pattern labels.
Shows the flag pattern with handle and surrounding context for visual verification.

Usage:
    uv run python label_verifier_mpl.py <labels.json> <data_dir> [output.json]
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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as mpatches

# Import our modules
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

from label_parser import LabeledSegment, parse_label_studio_export
from segment_extractor import load_timeseries_csv


@dataclass
class VerificationResult:
    """Result of label verification."""
    segment_idx: int
    original_label: str
    verified: bool = False
    correct: Optional[bool] = None
    new_label: Optional[str] = None
    notes: str = ""


class LabelVerifierMPL:
    """Matplotlib-based label verifier."""
    
    LABELS = [
        "Bullish Normal", "Bullish Wedge", "Bullish Pennant",
        "Bearish Normal", "Bearish Wedge", "Bearish Pennant"
    ]
    
    def __init__(
        self,
        labels_path: str,
        data_dir: str,
        output_path: Optional[str] = None,
        context_bars: int = 50,
        handle_bars: int = 20
    ):
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.output_path = output_path or labels_path.replace('.json', '_verified.json')
        self.context_bars = context_bars
        self.handle_bars = handle_bars
        
        # Load segments
        self.segments: List[LabeledSegment] = []
        self.df_cache: Dict[str, pd.DataFrame] = {}
        self.verification_results: Dict[int, VerificationResult] = {}
        
        self.current_idx = 0
        
        # Load data
        self._load_segments()
        self._load_existing_verification()
        
        # Setup figure
        self.fig = None
        self.ax = None
    
    def _load_segments(self):
        """Load all segments from label file."""
        self.segments = parse_label_studio_export(self.labels_path)
        print(f"Loaded {len(self.segments)} segments")
    
    def _load_existing_verification(self):
        """Load existing verification results."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    saved = json.load(f)
                for item in saved.get('verifications', []):
                    idx = item['segment_idx']
                    v = VerificationResult(
                        segment_idx=idx,
                        original_label=item['original_label'],
                        verified=item['verified'],
                        correct=item.get('correct'),
                        new_label=item.get('new_label'),
                        notes=item.get('notes', '')
                    )
                    self.verification_results[idx] = v
                print(f"Loaded {len(self.verification_results)} existing verifications")
            except Exception as e:
                print(f"Could not load saved results: {e}")
    
    def _load_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        """Load CSV with caching."""
        full_path = os.path.join(self.data_dir, csv_path)
        
        if full_path not in self.df_cache:
            if not os.path.exists(full_path):
                print(f"Warning: CSV not found: {full_path}")
                return None
            try:
                self.df_cache[full_path] = load_timeseries_csv(full_path)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                return None
        
        return self.df_cache[full_path]
    
    def _get_segment_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get segment data with context."""
        if idx < 0 or idx >= len(self.segments):
            return None
        
        seg = self.segments[idx]
        df = self._load_csv(seg.csv_path)
        
        if df is None:
            return None
        
        try:
            start_dt = pd.to_datetime(seg.start)
            end_dt = pd.to_datetime(seg.end)
        except Exception:
            return None
        
        # Find indices
        start_idx = df.index.searchsorted(start_dt)
        end_idx = df.index.searchsorted(end_dt)
        
        if start_idx >= len(df):
            return None
        
        # Calculate context range
        context_start = max(0, start_idx - self.context_bars - self.handle_bars)
        context_end = min(len(df), end_idx + self.context_bars)
        
        # Extract data
        context_df = df.iloc[context_start:context_end].copy()
        
        return {
            'segment': seg,
            'df': context_df,
            'flag_start_idx': start_idx - context_start,
            'flag_end_idx': end_idx - context_start,
            'handle_start_idx': max(0, start_idx - context_start - self.handle_bars),
        }
    
    def _plot_candlesticks(self, ax, df, flag_start, flag_end, handle_start):
        """Plot candlestick chart with regions highlighted."""
        ax.clear()
        
        n = len(df)
        if n == 0:
            return
        
        # Plot each candle
        for i, (idx, row) in enumerate(df.iterrows()):
            open_p, high, low, close = row['open'], row['high'], row['low'], row['close']
            
            # Determine color and alpha based on region
            if handle_start <= i < flag_start:
                # Handle region
                color = 'orange' if close >= open_p else 'darkorange'
                alpha = 1.0
            elif flag_start <= i < flag_end:
                # Flag region
                color = 'lime' if close >= open_p else 'red'
                alpha = 1.0
            else:
                # Context
                color = 'green' if close >= open_p else 'darkred'
                alpha = 0.4
            
            # Draw wick
            ax.plot([i, i], [low, high], color='black', linewidth=0.5, alpha=alpha)
            
            # Draw body
            body_bottom = min(open_p, close)
            body_height = abs(close - open_p)
            if body_height < 0.001:
                body_height = 0.001
            
            rect = Rectangle(
                (i - 0.3, body_bottom),
                0.6, body_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=alpha
            )
            ax.add_patch(rect)
        
        # Add region highlights
        if handle_start < flag_start:
            ax.axvspan(handle_start - 0.5, flag_start - 0.5, 
                      alpha=0.15, color='orange', label='Handle')
        
        ax.axvspan(flag_start - 0.5, flag_end - 0.5, 
                  alpha=0.15, color='blue', label='Flag')
        
        # Set axis
        ax.set_xlim(-1, n)
        price_min = df['low'].min()
        price_max = df['high'].max()
        padding = (price_max - price_min) * 0.1
        ax.set_ylim(price_min - padding, price_max + padding)
        
        ax.set_xlabel('Bar Index')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    def _update_plot(self):
        """Update the plot with current segment."""
        data = self._get_segment_data(self.current_idx)
        
        if data is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"No data for segment {self.current_idx + 1}",
                        ha='center', va='center', transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        
        seg = data['segment']
        df = data['df']
        
        # Plot candlesticks
        self._plot_candlesticks(
            self.ax, df,
            data['flag_start_idx'],
            data['flag_end_idx'],
            data['handle_start_idx']
        )
        
        # Build title
        v = self.verification_results.get(self.current_idx)
        if v and v.verified:
            status = "✓ Correct" if v.correct else "✗ Incorrect"
            if v.new_label:
                status += f" → {v.new_label}"
        else:
            status = "Not verified"
        
        label_color = 'green' if 'Bullish' in seg.label else 'red'
        
        title = (
            f"Segment {self.current_idx + 1}/{len(self.segments)} | "
            f"Label: {seg.label} | {status}\n"
            f"File: {seg.csv_path} | {seg.start} to {seg.end}"
        )
        self.ax.set_title(title, fontsize=10)
        
        self.fig.canvas.draw_idle()
    
    def _on_prev(self, event):
        """Go to previous segment."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_plot()
    
    def _on_next(self, event):
        """Go to next segment."""
        if self.current_idx < len(self.segments) - 1:
            self.current_idx += 1
            self._update_plot()
    
    def _on_correct(self, event):
        """Mark as correct."""
        seg = self.segments[self.current_idx]
        self.verification_results[self.current_idx] = VerificationResult(
            segment_idx=self.current_idx,
            original_label=seg.label,
            verified=True,
            correct=True
        )
        print(f"Marked #{self.current_idx + 1} as CORRECT")
        self._on_next(event)
    
    def _on_incorrect(self, event):
        """Mark as incorrect."""
        seg = self.segments[self.current_idx]
        self.verification_results[self.current_idx] = VerificationResult(
            segment_idx=self.current_idx,
            original_label=seg.label,
            verified=True,
            correct=False
        )
        print(f"Marked #{self.current_idx + 1} as INCORRECT")
        self._update_plot()
    
    def _on_save(self, event):
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
                for v in sorted(self.verification_results.values(), key=lambda x: x.segment_idx)
            ]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved to {self.output_path}")
    
    def _on_goto(self, text):
        """Go to specific segment."""
        try:
            idx = int(text) - 1
            if 0 <= idx < len(self.segments):
                self.current_idx = idx
                self._update_plot()
        except ValueError:
            pass
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'left' or event.key == 'k':
            self._on_prev(event)
        elif event.key == 'right' or event.key == 'j':
            self._on_next(event)
        elif event.key == 'y':
            self._on_correct(event)
        elif event.key == 'n':
            self._on_incorrect(event)
        elif event.key == 's':
            self._on_save(event)
    
    def run(self):
        """Run the verifier."""
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Add buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.05])
        ax_next = plt.axes([0.21, 0.05, 0.1, 0.05])
        ax_correct = plt.axes([0.35, 0.05, 0.1, 0.05])
        ax_incorrect = plt.axes([0.46, 0.05, 0.1, 0.05])
        ax_save = plt.axes([0.6, 0.05, 0.1, 0.05])
        ax_goto = plt.axes([0.75, 0.05, 0.1, 0.05])
        
        btn_prev = Button(ax_prev, '← Prev (K)')
        btn_next = Button(ax_next, 'Next (J) →')
        btn_correct = Button(ax_correct, '✓ Correct (Y)', color='lightgreen')
        btn_incorrect = Button(ax_incorrect, '✗ Incorrect (N)', color='lightcoral')
        btn_save = Button(ax_save, 'Save (S)', color='lightyellow')
        text_goto = TextBox(ax_goto, 'Go to #:')
        
        btn_prev.on_clicked(self._on_prev)
        btn_next.on_clicked(self._on_next)
        btn_correct.on_clicked(self._on_correct)
        btn_incorrect.on_clicked(self._on_incorrect)
        btn_save.on_clicked(self._on_save)
        text_goto.on_submit(self._on_goto)
        
        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Show legend
        legend_elements = [
            mpatches.Patch(facecolor='orange', alpha=0.3, label='Handle (Pole)'),
            mpatches.Patch(facecolor='blue', alpha=0.3, label='Flag Pattern'),
            mpatches.Patch(facecolor='green', alpha=0.4, label='Context (Bullish)'),
            mpatches.Patch(facecolor='darkred', alpha=0.4, label='Context (Bearish)'),
        ]
        
        # Initial plot
        self._update_plot()
        
        # Print instructions
        print("\n" + "=" * 60)
        print("LABEL VERIFIER")
        print("=" * 60)
        print(f"Total segments: {len(self.segments)}")
        print(f"Already verified: {len(self.verification_results)}")
        print("\nKeyboard shortcuts:")
        print("  ← / K    Previous segment")
        print("  → / J    Next segment")
        print("  Y        Mark as CORRECT")
        print("  N        Mark as INCORRECT")
        print("  S        Save results")
        print("=" * 60 + "\n")
        
        plt.show()


def run_verifier_mpl(
    labels_path: str,
    data_dir: str,
    output_path: Optional[str] = None,
    context_bars: int = 50,
    handle_bars: int = 20
):
    """Run the matplotlib-based label verifier."""
    verifier = LabelVerifierMPL(
        labels_path=labels_path,
        data_dir=data_dir,
        output_path=output_path,
        context_bars=context_bars,
        handle_bars=handle_bars
    )
    verifier.run()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: uv run python label_verifier_mpl.py <labels.json> <data_dir> [output.json]")
        print("\nArguments:")
        print("  labels.json  - Label Studio JSON export")
        print("  data_dir     - Directory containing CSV files")
        print("  output.json  - Output file for verification results (optional)")
        print("\nKeyboard shortcuts:")
        print("  ← / K    Previous segment")
        print("  → / J    Next segment")
        print("  Y        Mark as CORRECT")
        print("  N        Mark as INCORRECT")
        print("  S        Save results")
        sys.exit(1)
    
    labels_path = sys.argv[1]
    data_dir = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_verifier_mpl(labels_path, data_dir, output_path)
