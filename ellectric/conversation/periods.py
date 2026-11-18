"""Explicit dates and months named in Chinese / ISO text, and calendar shifts between them."""

from __future__ import annotations

import re
from datetime import date, timedelta

_ISO_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_ISO_MONTH = re.compile(r"(?<![\d-])(\d{4})-(\d{2})(?![\d-])")
_CN = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月(?:\s*(\d{1,2})\s*(?:日|号))?")


def _cn_matches(text: str):
    iso_spans = [(m.start(), m.end()) for m in _ISO_DAY.finditer(text)]
    year: int | None = None
    for match in _CN.finditer(text):
        if any(a <= match.start() < b for a, b in iso_spans):
            continue
        explicit = bool(match.group(1))
        if explicit:
            year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.group(3) else None
        if not 1 <= month <= 12:
            continue
        yield match.start(), (year if (explicit or year is not None) else None), month, day


def mentioned_periods(text: str) -> set[str]:
    """Dates (YYYY-MM-DD) and months (YYYY-MM) a text names explicitly.

    A Chinese date without a year ("至5月10日") takes the year of the previous date in the same text; a month
    or day with no year anywhere before it names no definite period and is left out (see ``partial_dates``).
    """
    out = {f"{m.group(1)}-{m.group(2)}-{m.group(3)}" for m in _ISO_DAY.finditer(text)}
    out |= {f"{m.group(1)}-{m.group(2)}" for m in _ISO_MONTH.finditer(text) if 1 <= int(m.group(2)) <= 12}
    for _, year, month, day in _cn_matches(text):
        if year is None:
            continue
        if day is None:
            out.add(f"{year}-{month:02d}")
            continue
        try:
            out.add(date(year, month, day).isoformat())
        except ValueError:
            continue
    return out


def partial_dates(text: str) -> set[tuple[int, int | None]]:
    """(month, day or None) for Chinese dates written without any year ("那7月31日呢")."""
    return {(month, day) for _, year, month, day in _cn_matches(text) if year is None}


def years_of(periods: set[str]) -> set[int]:
    return {int(p[:4]) for p in periods}


def shift(period: str, derivation: str, amount: int) -> str | None:
    """Apply a calendar shift to a YYYY-MM-DD or YYYY-MM period; None when it does not apply."""
    try:
        if len(period) == 10:
            day = date.fromisoformat(period)
            if derivation == "shift_days":
                return (day + timedelta(days=amount)).isoformat()
            if derivation == "shift_months":
                total = day.year * 12 + day.month - 1 + amount
                return date(total // 12, total % 12 + 1, day.day).isoformat()
            if derivation == "shift_years":
                return day.replace(year=day.year + amount).isoformat()
        elif len(period) == 7:
            year, month = int(period[:4]), int(period[5:])
            if derivation == "shift_months":
                total = year * 12 + month - 1 + amount
                return f"{total // 12}-{total % 12 + 1:02d}"
            if derivation == "shift_years":
                return f"{year + amount}-{month:02d}"
    except ValueError:
        return None
    return None
