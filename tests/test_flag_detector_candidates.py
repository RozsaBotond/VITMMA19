import pandas as pd

from bullflag_detector.flag_detector import detect_candidate_flags


def test_detect_candidate_flags_finds_pole():
    # Synthetic series: strong up move over 15 bars then flat
    idx = pd.date_range("2025-01-01", periods=80, freq="min", tz="UTC")
    close = []
    price = 100.0
    for i in range(80):
        if i < 15:
            price *= 1.01  # ~16% up over 15 bars
        else:
            price *= 1.0002
        close.append(price)

    df = pd.DataFrame(
        {
            "open": close,
            "high": [c * 1.001 for c in close],
            "low": [c * 0.999 for c in close],
            "close": close,
        },
        index=idx,
    )

    cands = detect_candidate_flags(df, pole_window=15, pole_threshold_pct=5.0, flag_window=20)

    assert len(cands) >= 1
    assert cands[0].direction_hint in {"Bullish", "Bearish"}
    assert cands[0].direction_hint == "Bullish"
