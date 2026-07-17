"""Collect and flatten Falcon 9 launch records from the SpaceX REST API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


API_URL = "https://api.spacexdata.com/v4/launches/past"


def collect_launches(url: str = API_URL) -> list[dict]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def flatten_launches(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        cores = record.get("cores") or [{}]
        core = cores[0] or {}
        rows.append(
            {
                "FlightNumber": record.get("flight_number"),
                "Date": record.get("date_utc"),
                "Name": record.get("name"),
                "Payloads": record.get("payloads"),
                "LaunchSite": record.get("launchpad"),
                "Core": core.get("core"),
                "Flights": core.get("flight"),
                "GridFins": core.get("gridfins"),
                "Reused": core.get("reused"),
                "Legs": core.get("legs"),
                "LandingPad": core.get("landpad"),
                "LandingSuccess": core.get("landing_success"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.sort_values("Date").reset_index(drop=True)


def main(output: str = "spacex_launches.csv") -> Path:
    frame = flatten_launches(collect_launches())
    path = Path(output)
    frame.to_csv(path, index=False)
    print(f"Saved {len(frame)} launch records to {path}")
    return path


if __name__ == "__main__":
    main()
