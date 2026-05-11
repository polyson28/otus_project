from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sample JSON request body for the FastAPI /predict Swagger endpoint."
    )
    parser.add_argument("--days", type=int, default=40, help="Number of history records to generate.")
    parser.add_argument("--start-date", default="2023-01-01", help="First date in YYYY-MM-DD format.")
    parser.add_argument("--output", default=None, help="Optional output JSON file path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    return parser.parse_args()


def business_day_flag(day: date) -> int:
    return int(day.weekday() < 5)


def tax_flag(day: date, tax_days: set[int]) -> int:
    return int(day.day in tax_days)


def build_record(day: date, index: int) -> dict[str, Any]:
    return {
        "date": day.isoformat(),
        "balance": round(100.0 + index * 2.7 + (index % 5) * 1.3, 2),
        "income": round(950.0 + index * 8.0 + (index % 7) * 12.5, 2),
        "outcome": round(820.0 + index * 6.0 + (index % 4) * 10.0, 2),
        "рабочий_день_(1/0)": business_day_flag(day),
        "ндс": tax_flag(day, {25}),
        "ндфл": tax_flag(day, {28}),
        "акцизы": tax_flag(day, {15}),
        "страховые_взносы": tax_flag(day, {15, 28}),
        "налог_на_прибыль": tax_flag(day, {28}),
        "налог_на_имущество_организаций": tax_flag(day, {30}),
        "транспортный_налог": tax_flag(day, {1}),
        "налог_на_землю": tax_flag(day, {1}),
        "digitalization_level": round(70.0 + (index % 12) * 0.3, 2),
        "age_0_14": 17.5,
        "age_15_64": 67.3,
        "age_65_plus": 15.2,
        "population_density": 120.5,
        "gdp_per_capita": round(15000.0 + index * 15.0, 2),
        "unemployment_rate": round(4.5 + (index % 5) * 0.03, 2),
        "consumer_confidence_index": round(95.0 + (index % 6) * 0.2, 2),
        "inflation_rate": round(7.4 + (index % 4) * 0.05, 2),
        "trade_balance": round(500.0 + index * 4.0 - (index % 3) * 8.0, 2),
        "close": round(3200.0 + index * 3.5, 2),
        "usd_rate": round(92.5 + (index % 10) * 0.12, 2),
    }


def build_payload(days: int, start_date: str) -> dict[str, Any]:
    if days < 31:
        raise ValueError("Use at least 31 days so lag_7 and rolling_30 features are available.")

    current_date = date.fromisoformat(start_date)
    records = [
        build_record(current_date + timedelta(days=index), index)
        for index in range(days)
    ]
    return {"records": records}


def main() -> None:
    args = parse_args()
    payload = build_payload(days=args.days, start_date=args.start_date)
    output = json.dumps(payload, ensure_ascii=False, indent=args.indent)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote Swagger /predict payload to {output_path}")
        return

    print(output)


if __name__ == "__main__":
    main()
