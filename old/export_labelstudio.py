"""bullflag_detector.export_labelstudio

Batch-export *potential* flag candidates from raw OHLC CSVs into Label Studio
import JSON.

This intentionally avoids any interactive GUI/TUI.

Outputs one JSON per CSV (default) so you can import them independently into
Label Studio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .flag_detector import (
	detect_candidate_flags,
	find_raw_data_files,
	load_ohlc_csv,
	write_labelstudio_potential_flags_json,
)


def export_potential_flags_for_csv(
	csv_path: str,
	out_path: str,
	max_candidates: int = 500,
	pole_window: int = 15,
	pole_threshold_pct: float = 5.0,
	flag_window: int = 30,
	uploader_prefix: str = "",
) -> int:
	"""Detect candidates in one CSV and write Label Studio import JSON.

	Returns number of candidates exported.
	"""
	# Raw files in this repo use both ',' and ';' separators.
	# Try comma first (default), then auto-fallback to semicolon.
	try:
		df = load_ohlc_csv(csv_path, sep=",")
	except ValueError as e:
		if "timestamp" not in str(e).lower():
			raise
		df = load_ohlc_csv(csv_path, sep=";")
	cands = detect_candidate_flags(
		df,
		pole_window=pole_window,
		pole_threshold_pct=pole_threshold_pct,
		flag_window=flag_window,
	)[:max_candidates]

	pairs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
	for c in cands:
		start_ts = df.index[c.flag_start_idx]
		end_ts = df.index[min(c.flag_end_idx - 1, len(df.index) - 1)]
		pairs.append((start_ts, end_ts))

	write_labelstudio_potential_flags_json(
		out_path,
		source_csv_filename=os.path.basename(csv_path),
		candidates=pairs,
		uploader_prefix=uploader_prefix,
		start_task_id=1,
	)
	return len(pairs)


def enrich_labels_json(
	labels_json_path: str,
	data_dir: str,
	out_path: str | None = None,
	max_candidates_per_csv: int = 500,
	pole_window: int = 15,
	pole_threshold_pct: float = 5.0,
	flag_window: int = 30,
) -> int:
	"""Enrich existing labels.json by detecting candidates and appending to existing tasks.

	For each task in the input JSON:
	1. Parse its data.csv field to find the referenced CSV basename
	2. Match that CSV to a file under data_dir/raw_data/
	3. Run candidate detection on the matched CSV
	4. Append detected regions (with empty timeserieslabels) to task's annotations[0].result

	Returns total number of candidate regions added.
	"""
	import json

	labels_path = Path(labels_json_path)
	tasks = json.loads(labels_path.read_text(encoding="utf-8"))
	if not tasks:
		raise SystemExit("Labels JSON is empty")

	# Build a map from CSV basename (stripped of prefix) to full path in raw_data
	raw_csv_map = {}
	for csv_path in find_raw_data_files(data_dir):
		basename = os.path.basename(csv_path)
		raw_csv_map[basename] = csv_path

	total_added = 0

	for task in tasks:
		# Extract CSV reference from task
		csv_ref = task.get("data", {}).get("csv", "")
		if not csv_ref:
			continue

		# csv_ref looks like "/data/upload/1/8e2ef3a4-XAU_1h_data_limited.csv"
		# We need to match the final part (e.g., "XAU_1h_data_limited.csv") to raw_data files
		csv_filename = os.path.basename(csv_ref)
		# The filename might have a prefix like "8e2ef3a4-", so try to match suffix
		matched_path = None
		for basename, fullpath in raw_csv_map.items():
			if csv_filename.endswith(basename) or basename.endswith(csv_filename.split("-")[-1] if "-" in csv_filename else csv_filename):
				matched_path = fullpath
				break
		if not matched_path:
			# Try exact match after removing UUID prefix (e.g., "8e2ef3a4-")
			stripped = csv_filename.split("-", 1)[-1] if "-" in csv_filename else csv_filename
			if stripped in raw_csv_map:
				matched_path = raw_csv_map[stripped]

		if not matched_path:
			print(f"[WARN] Could not match CSV for task {task.get('id')}: {csv_ref}")
			continue

		# Load the CSV and detect candidates
		try:
			df = load_ohlc_csv(matched_path, sep=",")
		except ValueError as e:
			if "timestamp" not in str(e).lower():
				raise
			df = load_ohlc_csv(matched_path, sep=";")

		cands = detect_candidate_flags(
			df,
			pole_window=pole_window,
			pole_threshold_pct=pole_threshold_pct,
			flag_window=flag_window,
		)[:max_candidates_per_csv]

		if not cands:
			continue

		# Prepare new result entries
		new_results = []
		task_id = task.get("id", 0)
		for i, c in enumerate(cands):
			start_ts = df.index[c.flag_start_idx]
			end_ts = df.index[min(c.flag_end_idx - 1, len(df.index) - 1)]
			rid = f"cand_{task_id}_{i}"
			new_results.append({
				"value": {
					"start": start_ts.strftime("%Y-%m-%d %H:%M"),
					"end": end_ts.strftime("%Y-%m-%d %H:%M"),
					"instant": False,
					"timeserieslabels": [],
				},
				"id": rid,
				"from_name": "label",
				"to_name": "ts",
				"type": "timeserieslabels",
				"origin": "manual",
			})

		# Append to existing annotations[0].result
		if not task.get("annotations"):
			task["annotations"] = [{"result": []}]
		if not task["annotations"][0].get("result"):
			task["annotations"][0]["result"] = []

		task["annotations"][0]["result"].extend(new_results)
		total_added += len(new_results)

	# Write output
	output_path = Path(out_path) if out_path else labels_path
	output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
	return total_added


def export_potential_flags_all(
	data_dir: str,
	out_path: str,
	max_candidates_per_csv: int = 500,
	pole_window: int = 15,
	pole_threshold_pct: float = 5.0,
	flag_window: int = 30,
	uploader_prefix: str = "/data/upload/1/",
) -> int:
	"""Export a single, combined Label Studio import JSON for all raw CSVs."""

	raw_csvs = find_raw_data_files(data_dir)
	if not raw_csvs:
		raise SystemExit(f"No CSVs found under: {Path(data_dir) / 'raw_data'}")

	out_file = Path(out_path)
	out_file.parent.mkdir(parents=True, exist_ok=True)

	task_id = 1
	total = 0
	# We append tasks by reusing the writer per CSV but offsetting ids.
	# To avoid reading/writing JSON repeatedly, we build payloads per CSV and merge in-memory.
	import json

	all_tasks = []
	for csv_path in raw_csvs:
		stem = Path(csv_path).stem
		tmp_out = out_file.parent / f"._tmp_{stem}.json"

		# Detect + write to tmp (uses correct per-task fields). Then load and merge.
		n = export_potential_flags_for_csv(
			csv_path,
			str(tmp_out),
			max_candidates=max_candidates_per_csv,
			pole_window=pole_window,
			pole_threshold_pct=pole_threshold_pct,
			flag_window=flag_window,
			uploader_prefix=uploader_prefix,
		)

		# Re-write ids to be globally unique and ensure data.csv includes prefix.
		tasks = json.loads(tmp_out.read_text(encoding="utf-8"))
		for t in tasks:
			t["id"] = task_id
			# keep file_upload as plain filename, but data.csv should include prefix
			t["data"]["csv"] = f"{uploader_prefix}{t['file_upload']}"
			# make result id stable too
			try:
				t["annotations"][0]["result"][0]["id"] = f"cand_{task_id}"
			except Exception:
				pass
			all_tasks.append(t)
			task_id += 1
		total += n

		try:
			tmp_out.unlink()
		except Exception:
			pass

	out_file.write_text(json.dumps(all_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
	return total


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(
		description="Export potential bull/bear flag candidates for Label Studio import (no GUI)."
	)
	parser.add_argument(
		"--data-dir",
		default="data",
		help="Project data directory that contains raw_data/*.csv",
	)
	parser.add_argument(
		"--out-dir",
		default="data/labelstudio_import",
		help="Directory to write Label Studio import JSONs",
	)
	parser.add_argument(
		"--out-file",
		default="",
		help="If set, write a single combined import JSON to this path (instead of per-CSV files)",
	)
	parser.add_argument(
		"--append-to",
		default="",
		help="If set, load an existing labels.json and append new tasks to it (path)",
	)
	parser.add_argument(
		"--enrich",
		default="",
		help="If set, enrich existing labels.json by appending detected candidates to each task's result array (path)",
	)
	parser.add_argument(
		"--uploader-prefix",
		default="",
		help="Optional prefix Label Studio uses to reference uploaded files (prepended to CSV filename)",
	)
	parser.add_argument("--max", type=int, default=500, help="Max candidates per CSV")
	parser.add_argument("--pole-window", type=int, default=15)
	parser.add_argument("--pole-pct", type=float, default=5.0)
	parser.add_argument("--flag-window", type=int, default=30)
	args = parser.parse_args()

	if args.enrich:
		total = enrich_labels_json(
			labels_json_path=args.enrich,
			data_dir=args.data_dir,
			out_path=args.out_file or None,
			max_candidates_per_csv=args.max,
			pole_window=args.pole_window,
			pole_threshold_pct=args.pole_pct,
			flag_window=args.flag_window,
		)
		out_dest = args.out_file if args.out_file else args.enrich
		print(f"Done. Added {total} candidate regions to existing tasks in: {out_dest}")
		return

	if args.out_file:
		total = export_potential_flags_all(
			data_dir=args.data_dir,
			out_path=args.out_file,
			max_candidates_per_csv=args.max,
			pole_window=args.pole_window,
			pole_threshold_pct=args.pole_pct,
			flag_window=args.flag_window,
			uploader_prefix=args.uploader_prefix or "/data/upload/1/",
		)
		print(f"Done. Exported {total} candidates into: {args.out_file}")
		return

	if args.append_to:
		# Build combined tasks in-memory and append to the provided labels.json
		import json
		existing = []
		try:
			existing = json.loads(Path(args.append_to).read_text(encoding="utf-8"))
		except Exception as e:
			raise SystemExit(f"Failed to read append file: {e}")

		# Use the first task's annotation as template
		if not existing:
			raise SystemExit("Append file appears empty; need an existing labels.json with at least one task")
		template_task = existing[0]
		try:
			annotation_template = template_task.get("annotations", [])[0]
		except Exception:
			raise SystemExit("Could not find annotation template in existing labels.json")

		# start ids from max existing id + 1
		max_task_id = max((t.get("id", 0) for t in existing), default=0)
		max_ann_id = 0
		for t in existing:
			for ann in t.get("annotations", []):
				max_ann_id = max(max_ann_id, ann.get("id", 0))

		raw_csvs = find_raw_data_files(args.data_dir)
		if not raw_csvs:
			raise SystemExit(f"No CSVs found under: {Path(args.data_dir) / 'raw_data'}")

		new_tasks = []
		task_id = max_task_id + 1
		ann_id = max_ann_id + 1
		import uuid

		for csv_path in raw_csvs:
			# Retry with semicolon separator if comma fails (same logic as export_potential_flags_for_csv)
			try:
				df = load_ohlc_csv(csv_path, sep=",")
			except ValueError as e:
				if "timestamp" not in str(e).lower():
					raise
				df = load_ohlc_csv(csv_path, sep=";")

			cands = detect_candidate_flags(
				df,
				pole_window=args.pole_window,
				pole_threshold_pct=args.pole_pct,
				flag_window=args.flag_window,
			)[: args.max]
			pairs = []
			for c in cands:
				start_ts = df.index[c.flag_start_idx]
				end_ts = df.index[min(c.flag_end_idx - 1, len(df.index) - 1)]
				pairs.append((start_ts, end_ts))

			# build result entries
			results = []
			for i, (s, e) in enumerate(pairs):
				rid = f"cand_{task_id}_{i}"
				results.append(
					{
						"value": {
							"start": s.strftime("%Y-%m-%d %H:%M"),
							"end": e.strftime("%Y-%m-%d %H:%M"),
							"instant": False,
							"timeserieslabels": [],
						},
						"id": rid,
						"from_name": annotation_template.get("result", [])[0].get("from_name", "label") if annotation_template.get("result") else "label",
						"to_name": annotation_template.get("result", [])[0].get("to_name", "ts") if annotation_template.get("result") else "ts",
						"type": "timeserieslabels",
						"origin": "manual",
					}
				)

			# clone template task shallowly
			new_task = dict(template_task)
			# replace metadata
			basename = os.path.basename(csv_path)
			new_task["id"] = task_id
			new_task["file_upload"] = basename
			new_task["data"] = {"csv": f"{args.uploader_prefix or '/data/upload/1/'}{basename}"}
			new_task["annotations"] = [
				dict(annotation_template, **{
					"id": ann_id,
					"task": task_id,
					"result": results,
					"result_count": len(results),
					"unique_id": str(uuid.uuid4()),
				})
			]

			new_tasks.append(new_task)
			task_id += 1
			ann_id += 1

		# append and write back
		combined = existing + new_tasks
		Path(args.append_to).write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"Appended {len(new_tasks)} tasks to: {args.append_to}")
		return

	raw_csvs = find_raw_data_files(args.data_dir)
	if not raw_csvs:
		raise SystemExit(f"No CSVs found under: {Path(args.data_dir) / 'raw_data'}")

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	total = 0
	for csv_path in raw_csvs:
		stem = Path(csv_path).stem
		out_path = out_dir / f"{stem}_potential_flags.json"
		n = export_potential_flags_for_csv(
			csv_path,
			str(out_path),
			max_candidates=args.max,
			pole_window=args.pole_window,
			pole_threshold_pct=args.pole_pct,
			flag_window=args.flag_window,
			uploader_prefix=args.uploader_prefix,
		)
		total += n
		print(f"{os.path.basename(csv_path)} -> {out_path}  ({n} candidates)")

	print(f"Done. Exported {total} candidates across {len(raw_csvs)} CSVs into: {out_dir}")


if __name__ == "__main__":
	main()
