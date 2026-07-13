"""
Marketplace timezone helpers — mirrors auroraBackend dashboardMetrics.js and
orderController.parseDateRangeForQuery so Profitability day boundaries match
Aurora Orders / Seller Central.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

# Keep in sync with auroraBackend/src/utils/dashboardMetrics.js
MARKETPLACE_TIMEZONES: dict[str, str] = {
    "A21TJRUUN4KGV": "Asia/Kolkata",
    "ATVPDKIKX0DER": "America/Los_Angeles",
    "A2EUQ1WTGCTBG2": "America/Toronto",
    "A1AM78C64UM0Y8": "America/Mexico_City",
    "A1F83G8C2ARO7P": "Europe/London",
    "A1PA6795UKMFR9": "Europe/Berlin",
    "A1RKKUPIHCS9HS": "Europe/Madrid",
    "A13V1IB3VIYZZH": "Europe/Paris",
    "A1VC38T7YXB528": "Asia/Tokyo",
    "A39IBJ37TRP1C6": "Australia/Sydney",
}

SALES_CHANNEL_TIMEZONES: dict[str, str] = {
    "Amazon.com": "America/Los_Angeles",
    "Amazon.in": "Asia/Kolkata",
    "Amazon.ca": "America/Toronto",
    "Amazon.com.mx": "America/Mexico_City",
    "Amazon.co.uk": "Europe/London",
    "Amazon.de": "Europe/Berlin",
    "Amazon.fr": "Europe/Paris",
    "Amazon.es": "Europe/Madrid",
    "Amazon.it": "Europe/Rome",
    "Amazon.co.jp": "Asia/Tokyo",
    "Amazon.com.au": "Australia/Sydney",
}

REGION_FALLBACK_TIMEZONES: dict[str, str] = {
    "NA": "America/Los_Angeles",
    "EU": "Europe/London",
    "FE": "Asia/Tokyo",
}

_PREFERRED_MARKETPLACES = (
    "ATVPDKIKX0DER",
    "A21TJRUUN4KGV",
    "A1F83G8C2ARO7P",
    "A2EUQ1WTGCTBG2",
)


def resolve_dashboard_timezone(
    user: dict,
    query_timezone: Optional[str] = None,
    top_sales_channel: Optional[str] = None,
) -> str:
    """Seller-level timezone for dashboard / profitability date filters."""
    if query_timezone and isinstance(query_timezone, str) and query_timezone.strip():
        return query_timezone.strip()

    if top_sales_channel and top_sales_channel in SALES_CHANNEL_TIMEZONES:
        return SALES_CHANNEL_TIMEZONES[top_sales_channel]

    marketplace_ids = [str(x) for x in (user.get("amazonMarketplaceIds") or [])]
    for marketplace_id in _PREFERRED_MARKETPLACES:
        if marketplace_id in marketplace_ids and marketplace_id in MARKETPLACE_TIMEZONES:
            return MARKETPLACE_TIMEZONES[marketplace_id]

    for marketplace_id in marketplace_ids:
        if marketplace_id in MARKETPLACE_TIMEZONES:
            return MARKETPLACE_TIMEZONES[marketplace_id]

    region = user.get("marketplace")
    if region and region in REGION_FALLBACK_TIMEZONES:
        return REGION_FALLBACK_TIMEZONES[region]

    return "UTC"


def resolve_order_timezone(order: dict, fallback_timezone: str = "UTC") -> str:
    """Timezone Seller Central uses for a single order."""
    marketplace_id = order.get("marketplaceId") or order.get("MarketplaceId")
    if marketplace_id and marketplace_id in MARKETPLACE_TIMEZONES:
        return MARKETPLACE_TIMEZONES[marketplace_id]

    channel = order.get("salesChannel") or order.get("marketplaceName")
    if channel and channel in SALES_CHANNEL_TIMEZONES:
        return SALES_CHANNEL_TIMEZONES[channel]

    return fallback_timezone or "UTC"


def parse_ymd_parts(value: Any) -> Optional[dict[str, int]]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        return None
    try:
        year, month, day = (int(part) for part in text.split("-"))
    except ValueError:
        return None
    if month < 1 or month > 12 or day < 1 or day > 31:
        return None
    return {"year": year, "month": month, "day": day}


def zoned_time_to_utc(parts: dict[str, int], timezone_name: str) -> datetime:
    """Local calendar instant in `timezone_name`, returned as UTC-aware datetime."""
    tz = ZoneInfo(timezone_name)
    dt_local = datetime(
        parts["year"],
        parts["month"],
        parts["day"],
        parts.get("hour", 0),
        parts.get("minute", 0),
        parts.get("second", 0),
        parts.get("microsecond", 0),
        tzinfo=tz,
    )
    return dt_local.astimezone(timezone.utc)


def parse_date_range_for_query(
    start_date_raw: Any,
    end_date_raw: Any,
    timezone_name: str,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Build UTC window from YYYY-MM-DD inputs in marketplace local time."""
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    tz = timezone_name or "UTC"

    start_parts = parse_ymd_parts(start_date_raw)
    if start_parts:
        start_dt = zoned_time_to_utc(
            {**start_parts, "hour": 0, "minute": 0, "second": 0, "microsecond": 0},
            tz,
        )

    end_parts = parse_ymd_parts(end_date_raw)
    if end_parts:
        end_dt = zoned_time_to_utc(
            {
                **end_parts,
                "hour": 23,
                "minute": 59,
                "second": 59,
                "microsecond": 999_000,
            },
            tz,
        )

    return start_dt, end_dt


def resolve_window_from_days_back(
    now: datetime,
    days_back: int,
    timezone_name: str,
) -> tuple[datetime, datetime, str, str]:
    """Last N calendar days in marketplace TZ (inclusive of today)."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(timezone_name or "UTC")
    now_local = now.astimezone(tz)
    span = max(int(days_back), 1)

    end_dt = zoned_time_to_utc(
        {
            "year": now_local.year,
            "month": now_local.month,
            "day": now_local.day,
            "hour": 23,
            "minute": 59,
            "second": 59,
            "microsecond": 999_000,
        },
        timezone_name,
    )
    if end_dt > now:
        end_dt = now

    start_local_date = now_local.date() - timedelta(days=span - 1)
    start_dt = zoned_time_to_utc(
        {
            "year": start_local_date.year,
            "month": start_local_date.month,
            "day": start_local_date.day,
            "hour": 0,
            "minute": 0,
            "second": 0,
            "microsecond": 0,
        },
        timezone_name,
    )
    return start_dt, end_dt, start_local_date.isoformat(), now_local.date().isoformat()


def utc_instant_to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
