"""Calendar helpers shared by historical-profile builders and viewers."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


# Official public/adjusted holiday dates used by the project day-class feature.
# The label exposed downstream is `national_holiday` to distinguish it from
# ordinary weekends. The set intentionally includes observed/adjusted closure
# days because traffic patterns follow the working calendar.
TAIWAN_PUBLIC_HOLIDAYS = frozenset(
    {
        # 2021
        "2021-01-01",
        "2021-02-10",
        "2021-02-11",
        "2021-02-12",
        "2021-02-13",
        "2021-02-14",
        "2021-02-15",
        "2021-02-16",
        "2021-02-28",
        "2021-03-01",
        "2021-04-02",
        "2021-04-04",
        "2021-04-05",
        "2021-05-01",
        "2021-06-14",
        "2021-09-20",
        "2021-09-21",
        "2021-10-10",
        "2021-10-11",
        "2021-12-31",
        # 2022
        "2022-01-01",
        "2022-01-31",
        "2022-02-01",
        "2022-02-02",
        "2022-02-03",
        "2022-02-04",
        "2022-02-28",
        "2022-04-04",
        "2022-04-05",
        "2022-05-01",
        "2022-06-03",
        "2022-09-09",
        "2022-09-10",
        "2022-10-10",
        # 2023
        "2023-01-01",
        "2023-01-02",
        "2023-01-20",
        "2023-01-21",
        "2023-01-22",
        "2023-01-23",
        "2023-01-24",
        "2023-01-25",
        "2023-01-26",
        "2023-01-27",
        "2023-02-27",
        "2023-02-28",
        "2023-04-03",
        "2023-04-04",
        "2023-04-05",
        "2023-05-01",
        "2023-06-22",
        "2023-06-23",
        "2023-09-29",
        "2023-10-09",
        "2023-10-10",
        # 2024
        "2024-01-01",
        "2024-02-08",
        "2024-02-09",
        "2024-02-10",
        "2024-02-11",
        "2024-02-12",
        "2024-02-13",
        "2024-02-14",
        "2024-02-28",
        "2024-04-04",
        "2024-04-05",
        "2024-05-01",
        "2024-06-10",
        "2024-09-17",
        "2024-10-10",
        # 2025
        "2025-01-01",
        "2025-01-27",
        "2025-01-28",
        "2025-01-29",
        "2025-01-30",
        "2025-01-31",
        "2025-02-28",
        "2025-04-03",
        "2025-04-04",
        "2025-05-01",
        "2025-05-30",
        "2025-05-31",
        "2025-09-28",
        "2025-09-29",
        "2025-10-06",
        "2025-10-10",
        "2025-10-24",
        "2025-10-25",
        "2025-12-25",
        # 2026
        "2026-01-01",
        "2026-02-15",
        "2026-02-16",
        "2026-02-17",
        "2026-02-18",
        "2026-02-19",
        "2026-02-20",
        "2026-02-27",
        "2026-02-28",
        "2026-04-03",
        "2026-04-04",
        "2026-04-05",
        "2026-04-06",
        "2026-05-01",
        "2026-06-19",
        "2026-09-25",
        "2026-09-28",
        "2026-10-09",
        "2026-10-10",
        "2026-10-25",
        "2026-10-26",
        "2026-12-25",
    }
)

TAIWAN_MAKEUP_WORKDAYS = frozenset(
    {
        "2021-02-20",
        "2021-09-11",
        "2022-01-22",
        "2023-01-07",
        "2023-02-04",
        "2023-02-18",
        "2023-03-25",
        "2023-06-17",
        "2023-09-23",
        "2024-02-17",
        "2025-02-08",
    }
)


def date_key(value: date | datetime | pd.Timestamp | str) -> str:
    """Return YYYY-MM-DD for a date-like value."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def day_class(value: date | datetime | pd.Timestamp | str) -> str:
    """Classify a Taiwan local date as weekday, weekend, or national_holiday."""
    ts = pd.Timestamp(value)
    key = ts.strftime("%Y-%m-%d")
    if key in TAIWAN_MAKEUP_WORKDAYS:
        return "weekday"
    if key in TAIWAN_PUBLIC_HOLIDAYS:
        return "national_holiday"
    if ts.dayofweek >= 5:
        return "weekend"
    return "weekday"
