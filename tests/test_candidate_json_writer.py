import json
from pathlib import Path

import pandas as pd

from bullflag_detector.flag_detector import (
    write_labelstudio_candidates_json,
    write_labelstudio_potential_flags_json,
)


def test_write_labelstudio_candidates_json_minimal(tmp_path: Path):
    out = tmp_path / "candidates.json"
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    end = pd.Timestamp("2025-01-01T01:00:00Z")

    write_labelstudio_candidates_json(
        str(out),
        source_csv_filename="XAU_1h_data_limited.csv",
        candidates=[(start, end, "Bullish Normal")],
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["file_upload"] == "XAU_1h_data_limited.csv"

    value = data[0]["annotations"][0]["result"][0]["value"]
    assert value["timeserieslabels"] == ["Bullish Normal"]
    assert value["start"].endswith("Z")
    assert value["end"].endswith("Z")


def test_write_labelstudio_potential_flags_json_minimal(tmp_path: Path):
    out = tmp_path / "potential.json"
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    end = pd.Timestamp("2025-01-01T01:00:00Z")

    write_labelstudio_potential_flags_json(
        str(out),
        source_csv_filename="XAU_1h_data_limited.csv",
        candidates=[(start, end)],
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["file_upload"] == "XAU_1h_data_limited.csv"

    res = data[0]["annotations"][0]["result"][0]
    assert res["type"] == "timeserieslabels"
    value = res["value"]
    assert "T" not in value["start"]
    assert value["start"].count(":") == 1
    assert "T" not in value["end"]
    assert value["end"].count(":") == 1
    assert value["timeserieslabels"] == []
