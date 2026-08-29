"""Convert marketplace money to USD for profitability totals.

Sellers with US + CA (and MX, etc.) listings produce orders whose
``itemPrice.amount`` is CAD or MXN next to USD. Profitability used to
add those raw floats together, so January revenue/referral/FBA/fuel for
a mixed-ASIN (e.g. B09JZL4J8S) treated C$17.98 as $17.98 USD.

All profitability money is converted to USD *before* aggregating:

  USD amount = native amount × (USD per 1 unit of native currency)

Rates come from Frankfurter (ECB daily) with a static fallback so a
failed FX fetch never silently re-mixes currencies. Weekends/holidays
carry forward the last published business-day rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

import httpx

log = logging.getLogger(__name__)

_PENNY = Decimal("0.01")
TARGET_CURRENCY = "USD"
FRANKFURTER_URL = "https://api.frankfurter.dev/v1"

# Approximate USD per 1 unit — used when Frankfurter has no series
# (AED/SAR/EGP) or the HTTP fetch fails. MXN ~ Jan 2026 ECB (~0.0555).
FALLBACK_USD_PER_UNIT: dict[str, float] = {
    "USD": 1.0,
    "CAD": 0.72,
    "MXN": 0.055,
    "EUR": 1.08,
    "GBP": 1.27,
    "AUD": 0.66,
    "JPY": 0.0064,
    "INR": 0.011,
    "BRL": 0.18,
    "AED": 0.272,
    "SAR": 0.267,
    "SGD": 0.74,
    "PLN": 0.25,
    "SEK": 0.095,
    "TRY": 0.023,
    "EGP": 0.021,
    "HKD": 0.128,
    "CHF": 1.12,
    "CNY": 0.14,
    "NZD": 0.58,
}

SALES_CHANNEL_CURRENCY: dict[str, str] = {
    "amazon.com": "USD",
    "amazon.ca": "CAD",
    "amazon.com.mx": "MXN",
    "amazon.com.br": "BRL",
    "amazon.co.uk": "GBP",
    "amazon.de": "EUR",
    "amazon.fr": "EUR",
    "amazon.it": "EUR",
    "amazon.es": "EUR",
    "amazon.nl": "EUR",
    "amazon.se": "SEK",
    "amazon.pl": "PLN",
    "amazon.com.be": "EUR",
    "amazon.com.tr": "TRY",
    "amazon.eg": "EGP",
    "amazon.sa": "SAR",
    "amazon.ae": "AED",
    "amazon.sg": "SGD",
    "amazon.in": "INR",
    "amazon.co.jp": "JPY",
    "amazon.com.au": "AUD",
}

US_MARKETPLACE_ID = "ATVPDKIKX0DER"


def is_us_marketplace(marketplace_id: str | None) -> bool:
    """True for Amazon.com or a missing id (legacy All-Orders rows)."""
    mid = (marketplace_id or "").strip()
    return (not mid) or mid == US_MARKETPLACE_ID


SALES_CHANNEL_MARKETPLACE: dict[str, str] = {
    "amazon.com": "ATVPDKIKX0DER",
    "amazon.ca": "A2EUQ1WTGCTBG2",
    "amazon.com.mx": "A1AM78C64UM0Y8",
    "amazon.com.br": "A2Q3Y263D00KWC",
    "amazon.co.uk": "A1F83G8C2ARO7P",
    "amazon.de": "A1PA6795UKMFR9",
    "amazon.fr": "A13V1IB3VIYZZH",
    "amazon.it": "APJ6JRA9NG5V4",
    "amazon.es": "A1RKKUPIHCS9HS",
    "amazon.nl": "A1805IZSGTT6HS",
    "amazon.se": "A2NODRKZP88ZB9",
    "amazon.pl": "A1C3SOZRARQ6R3",
    "amazon.com.tr": "A33AVAJ2PDY3EV",
    "amazon.eg": "ARBP9OOSHTCHU",
    "amazon.sa": "A17E79C6D8DWNP",
    "amazon.ae": "A2VIGQ35RCS4UG",
    "amazon.sg": "A19VAU5U5O7RUS",
    "amazon.in": "A21TJRUUN4KGV",
    "amazon.co.jp": "A1VC38T7YXB528",
    "amazon.com.au": "A39IBJ37TRP1C6",
}

MARKETPLACE_CURRENCY: dict[str, str] = {
    "ATVPDKIKX0DER": "USD",
    "A2EUQ1WTGCTBG2": "CAD",
    "A1MQXOICRS2Z7M": "CAD",
    "A1AM78C64UM0Y8": "MXN",
    "A2Q3Y263D00KWC": "BRL",
    "A1F83G8C2ARO7P": "GBP",
    "A1PA6795UKMFR9": "EUR",
    "A13V1IB3VIYZZH": "EUR",
    "APJ6JRA9NG5V4": "EUR",
    "A1RKKUPIHCS9HS": "EUR",
    "A1805IZSGTT6HS": "EUR",
    "A2NODRKZP88ZB9": "SEK",
    "A1C3SOZRARQ6R3": "PLN",
    "A33AVAJ2PDY3EV": "TRY",
    "ARBP9OOSHTCHU": "EGP",
    "A3H6HPSLHAK3XG": "EGP",
    "A17E79C6D8DWNP": "SAR",
    "A2ZV50J4W1RKNI": "SAR",
    "A2VIGQ35RCS4UG": "AED",
    "A19VAU5U5O7RUS": "SGD",
    "AHRY1CZE9ZY4H": "SGD",
    "A21TJRUUN4KGV": "INR",
    "A1VC38T7YXB528": "JPY",
    "A39IBJ37TRP1C6": "AUD",
}

# (currency, start_iso, end_iso) → {date: usd_per_unit}
_FX_CACHE: dict[tuple[str, str, str], dict[date, float]] = {}


def _round_money(value: float | Decimal | str) -> float:
    try:
        d = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        return 0.0
    return float(d.quantize(_PENNY, rounding=ROUND_HALF_UP))


def _as_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.date()
    except ValueError:
        return None


def infer_order_date(order: dict | None) -> date | None:
    if not order:
        return None
    return _as_date(
        order.get("purchaseDate")
        or order.get("PurchaseDate")
        or order.get("purchase_date")
    )


def infer_marketplace_id(order: dict | None) -> str:
    """marketplaceId, or salesChannel → Amazon marketplace id.

    Aurora All-Orders sync often leaves ``marketplaceId`` empty and only
    fills ``salesChannel`` (Amazon.com / Amazon.ca). Profitability's
    marketplace filter and currency inference both need that mapping.
    """
    if not order:
        return ""
    mid = order.get("marketplaceId") or order.get("MarketplaceId") or ""
    if isinstance(mid, str) and mid.strip():
        return mid.strip()
    channel = str(
        order.get("salesChannel") or order.get("SalesChannel") or ""
    ).strip().lower()
    return SALES_CHANNEL_MARKETPLACE.get(channel, "")


def _iso_currency(value) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return ""


def infer_channel_currency(order: dict | None) -> str:
    """Amazon.ca / .mx / etc. → ISO code from salesChannel or marketplaceId."""
    if not order:
        return ""
    channel = str(
        order.get("salesChannel") or order.get("SalesChannel") or ""
    ).strip().lower()
    if channel in SALES_CHANNEL_CURRENCY:
        return SALES_CHANNEL_CURRENCY[channel]
    mid = infer_marketplace_id(order)
    return MARKETPLACE_CURRENCY.get(mid, "")


def _explicit_money_currency(order: dict | None, item: dict | None) -> str:
    if item:
        for field in (
            "itemSubtotal", "itemPrice", "promotionDiscount",
            "ItemPrice", "PromotionDiscount",
        ):
            block = item.get(field) or {}
            if not isinstance(block, dict):
                continue
            code = _iso_currency(
                block.get("currencyCode") or block.get("CurrencyCode")
            )
            if code:
                return code
    if order:
        top = order.get("orderTotal") or order.get("OrderTotal") or {}
        if isinstance(top, dict):
            code = _iso_currency(
                top.get("currencyCode") or top.get("CurrencyCode")
            )
            if code:
                return code
    return ""


def infer_line_currency(
    order: dict | None,
    item: dict | None = None,
    default: str = TARGET_CURRENCY,
) -> str:
    """Best-effort ISO code for one order line.

    ``default`` is USD for conversion (never empty). Pass ``default=""``
    when scanning for mixed currencies so a missing code is not counted
    as USD.

    Prefer the Amazon sales channel when a money field is labelled USD on a
    CAD/MXN marketplace. Order sync stamps US ``products.fees`` currency onto
    referral/FBA even when ``itemPrice`` is pesos — treating that USD label as
    truth added MXN 267 as $267 revenue/referral.
    """
    fallback = (default or "").strip().upper()
    channel_ccy = infer_channel_currency(order)
    money_ccy = _explicit_money_currency(order, item)
    if money_ccy and channel_ccy and money_ccy != channel_ccy:
        if money_ccy == TARGET_CURRENCY:
            return channel_ccy
        return money_ccy
    if money_ccy:
        return money_ccy
    if channel_ccy:
        return channel_ccy
    return fallback


def referral_fee_currency(order: dict | None, item: dict | None = None) -> str:
    """Referral is a % of line revenue — always the line's marketplace currency.

    Stored ``referralFee.currencyCode`` is often USD from the US fee map even
    when the 15% was taken of a CAD/MXN face value.
    """
    return infer_line_currency(order, item)


def money_field_currency(block, fallback: str = TARGET_CURRENCY) -> str:
    if not isinstance(block, dict):
        return (fallback or TARGET_CURRENCY).upper()
    code = block.get("currencyCode") or block.get("CurrencyCode")
    if isinstance(code, str) and code.strip():
        return code.strip().upper()
    return (fallback or TARGET_CURRENCY).upper()


@dataclass
class UsdFx:
    """USD-per-unit table keyed by (currency, date)."""

    rates: dict[tuple[str, date], float] = field(default_factory=dict)
    fallback: dict[str, float] = field(
        default_factory=lambda: dict(FALLBACK_USD_PER_UNIT)
    )
    source: str = "fallback"
    missing: set[str] = field(default_factory=set)

    def rate(self, currency: str | None, on: date | None = None) -> float:
        cur = (currency or TARGET_CURRENCY).strip().upper() or TARGET_CURRENCY
        if cur == TARGET_CURRENCY:
            return 1.0
        if on is not None:
            found = self.rates.get((cur, on))
            if found is None:
                for i in range(1, 8):
                    found = self.rates.get((cur, on - timedelta(days=i)))
                    if found is not None:
                        break
            if found is not None:
                return float(found)
        fb = self.fallback.get(cur)
        if fb is not None:
            return float(fb)
        self.missing.add(cur)
        return 1.0

    def to_usd(
        self,
        amount: float | None,
        currency: str | None,
        on: date | None = None,
    ) -> float:
        try:
            raw = float(amount or 0)
        except (TypeError, ValueError):
            return 0.0
        if raw == 0:
            return 0.0
        return _round_money(raw * self.rate(currency, on))

    def summary(self, currencies: Iterable[str]) -> dict:
        out: dict[str, dict] = {}
        for raw in currencies:
            cur = str(raw or "").strip().upper()
            if not cur or cur == TARGET_CURRENCY:
                continue
            dated = [
                (d, r) for (c, d), r in self.rates.items() if c == cur
            ]
            if dated:
                dated.sort(key=lambda x: x[0])
                mid = dated[len(dated) // 2]
                out[cur] = {
                    "usd_per_unit": round(float(mid[1]), 6),
                    "as_of": mid[0].isoformat(),
                    "source": self.source,
                }
            elif cur in self.fallback:
                out[cur] = {
                    "usd_per_unit": float(self.fallback[cur]),
                    "as_of": None,
                    "source": "fallback",
                }
        return out


def identity_fx() -> UsdFx:
    return UsdFx(source="identity")


def _fill_calendar(
    daily: dict[date, float], start: date, end: date,
) -> dict[date, float]:
    """Carry last known rate across weekends/holidays in [start, end]."""
    if not daily:
        return {}
    filled: dict[date, float] = {}
    last: float | None = None
    earliest = min(daily)
    # Seed last with the first published rate on/before start.
    probe = start
    while probe >= earliest - timedelta(days=10):
        if probe in daily:
            last = daily[probe]
            break
        probe -= timedelta(days=1)
    if last is None:
        last = daily[min(daily)]
    day = start
    while day <= end:
        if day in daily:
            last = daily[day]
        if last is not None:
            filled[day] = last
        day += timedelta(days=1)
    return filled


async def _fetch_frankfurter_range(
    currency: str, start: date, end: date,
) -> dict[date, float]:
    cur = currency.upper()
    key = (cur, start.isoformat(), end.isoformat())
    cached = _FX_CACHE.get(key)
    if cached is not None:
        return cached
    url = f"{FRANKFURTER_URL}/{start.isoformat()}..{end.isoformat()}"
    params = {"from": cur, "to": TARGET_CURRENCY}
    daily: dict[date, float] = {}
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json() or {}
    except Exception as e:
        log.warning("FX fetch failed for %s %s..%s: %s", cur, start, end, e)
        return {}
    rates_block = payload.get("rates") or {}
    # Single-day responses put rates at the top level.
    if "USD" in (payload.get("rates") or {}) and not any(
        isinstance(v, dict) for v in rates_block.values()
    ):
        as_of = _as_date(payload.get("date")) or start
        try:
            daily[as_of] = float(rates_block["USD"])
        except (TypeError, ValueError, KeyError):
            pass
    else:
        for day_s, pair in rates_block.items():
            d = _as_date(day_s)
            if d is None or not isinstance(pair, dict):
                continue
            try:
                daily[d] = float(pair.get(TARGET_CURRENCY))
            except (TypeError, ValueError):
                continue
    filled = _fill_calendar(daily, start, end)
    _FX_CACHE[key] = filled
    return filled


async def load_usd_fx(
    currencies: Iterable[str],
    start: date | datetime | None,
    end: date | datetime | None,
) -> UsdFx:
    """Load USD rates for every non-USD code in ``currencies``."""
    start_d = _as_date(start) or date.today()
    end_d = _as_date(end) or start_d
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    need = sorted({
        str(c).strip().upper()
        for c in (currencies or [])
        if str(c or "").strip() and str(c).strip().upper() != TARGET_CURRENCY
    })
    if not need:
        return identity_fx()

    table = UsdFx(source="fallback")
    fetched_any = False
    for cur in need:
        daily = await _fetch_frankfurter_range(cur, start_d, end_d)
        if daily:
            fetched_any = True
            for d, r in daily.items():
                table.rates[(cur, d)] = r
        elif cur not in table.fallback:
            table.missing.add(cur)
    table.source = "frankfurter" if fetched_any else "fallback"
    return table
