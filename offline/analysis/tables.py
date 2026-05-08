"""Write analysis rows to CSV."""

from __future__ import annotations

import csv
import os
from typing import Any


def write_csv_dict_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
