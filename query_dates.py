"""Natural-language date ranges for memory search.

Pulls phrases like "yesterday", "last week", "in March", "since June" out of
a query and converts them to concrete UTC bounds. The remaining text is what
gets embedded; the range becomes a Qdrant filter. Stdlib only — no date
library dependency at the serving layer.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_start(day: datetime) -> str:
    return day.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _day_end(day: datetime) -> str:
    return day.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Ordered longest-first so "last month" wins over "month".
_PATTERNS = (
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), "this week"),
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), "last week"),
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), "this month"),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), "last month"),
    (re.compile(r"\bin\s+(%s)\b" % "|".join(_MONTH_NAMES), re.IGNORECASE), "in month"),
    (re.compile(r"\bsince\s+(%s)\b" % "|".join(_MONTH_NAMES), re.IGNORECASE), "since month"),
)


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
    end = datetime(end_year, end_month, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    return start, end


def _resolve(match_kind: str, match: re.Match) -> tuple[str, str] | None:
    now = _utc_now()
    if match_kind == "today":
        return _day_start(now), _day_end(now)
    if match_kind == "yesterday":
        return _day_start(now - timedelta(days=1)), _day_end(now - timedelta(days=1))
    if match_kind == "this week":
        return _day_start(now - timedelta(days=6)), _day_end(now)
    if match_kind == "last week":
        return _day_start(now - timedelta(days=7)), _day_end(now - timedelta(days=1))
    if match_kind == "this month":
        return _day_start(now.replace(day=1)), _day_end(now)
    if match_kind == "last month":
        return _day_start(now - timedelta(days=30)), _day_end(now - timedelta(days=1))
    month = _MONTH_NAMES[match.group(1).lower()]
    if match_kind == "in month":
        # A bare month means its most recent occurrence in the past.
        year = now.year if month <= now.month else now.year - 1
        start, end = _month_bounds(year, month)
        return start.isoformat(), end.isoformat()
    if match_kind == "since month":
        year = now.year if month <= now.month else now.year - 1
        start, _ = _month_bounds(year, month)
        return start.isoformat(), _day_end(now)
    return None


def parse_date_range(query: str) -> tuple[str, str | None, str | None]:
    """Strip the first recognized date phrase from ``query``.

    Returns ``(cleaned_query, date_from_iso, date_to_iso)``; the two bound
    strings are None when no phrase was found. Only one date phrase per
    query is interpreted — combining "last week in March" is ambiguous on
    purpose.
    """
    for pattern, kind in _PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        bounds = _resolve(kind, match)
        if bounds is None:
            continue
        cleaned = (query[: match.start()] + " " + query[match.end() :]).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned, bounds[0], bounds[1]
    return query.strip(), None, None
