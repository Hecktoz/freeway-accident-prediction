# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
# ]
# ///
"""Small deterministic validation for causal weather matching."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from build_incident_features import build_weather_feature_table


def main() -> None:
    context = pd.DataFrame({
        "task_id": ["A", "B", "C", "D"],
        "dispatch_timestamp": [
            "2025-01-01 10:00", "2025-01-01 10:00",
            "2025-01-01 10:00", "not-a-time",
        ],
    })
    weather = pd.DataFrame({
        "task_id": ["A", "A", "B", "C", "D"],
        "observation_time": [
            "2025-01-01 09:00",  # selected for A
            "2025-01-01 10:15",  # future row must never be selected
            "2025-01-01 08:00",  # prior but stale under a 90-min limit
            "2025-01-01 10:05",  # future-only
            "2025-01-01 09:30",
        ],
        "temp_c": [10, 30, 20, 20, 20],
        "humidity_pct": [90, 50, 50, 50, 50],
        "wind_speed_mps": [1, 1, 1, 1, 1],
        "rain_mm": [0, 0, 0, 0, 0],
        "visibility_km": [10, 10, 10, 10, 10],
    })

    with tempfile.TemporaryDirectory(prefix="trb_weather_match_") as temp_dir:
        weather_path = Path(temp_dir) / "weather.csv"
        weather.to_csv(weather_path, index=False)
        selected, matches, audit = build_weather_feature_table(
            weather_path,
            context,
            max_staleness_minutes=90,
        )

        assert selected["task_id"].tolist() == ["A"]
        assert int(selected.loc[0, "low_temp"]) == 1
        indexed = matches.set_index("task_id")
        assert indexed.loc["A", "weather_observation_timestamp"] == pd.Timestamp("2025-01-01 09:00")
        assert indexed.loc["A", "weather_staleness_minutes"] == 60.0
        assert bool(indexed.loc["A", "has_future_candidate"])
        assert indexed.loc["B", "weather_match_status"] == "stale_prior_observation"
        assert indexed.loc["C", "weather_match_status"] == "no_prior_observation"
        assert indexed.loc["D", "weather_match_status"] == "invalid_dispatch_timestamp"
        assert audit["mode"] == "causal_backward"
        assert audit["paper_valid"] is True
        assert audit["matched_timestamp_count"] == 1
        assert audit["future_timestamp_violations"] == 0
        assert audit["over_staleness_limit_violations"] == 0

        no_time_path = Path(temp_dir) / "weather_without_observation_time.csv"
        weather.drop(columns=["observation_time"]).to_csv(no_time_path, index=False)
        try:
            build_weather_feature_table(no_time_path, context)
        except ValueError as exc:
            assert "no actual observation timestamp" in str(exc)
        else:
            raise AssertionError("strict paper weather unexpectedly accepted an unverified CSV")

        _, _, legacy_audit = build_weather_feature_table(
            no_time_path,
            context,
            allow_unverified_weather=True,
        )
        assert legacy_audit["mode"] == "legacy_task_id_unverified"
        assert legacy_audit["paper_valid"] is False

    print("causal weather matching validation: PASS")


if __name__ == "__main__":
    main()
