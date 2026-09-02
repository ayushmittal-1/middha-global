"""Convert marketplace money to USD for profitability totals.

Sellers with US + CA (and MX, etc.) listings produce orders whose
``itemPrice.amount`` is CAD or MXN next to USD. Profitability used to
add those raw floats together, so January revenue/referral/FBA/fuel for
a mixed-ASIN (e.g. B09JZL4J8S) treated C$17.98 as $17.98 USD.

All profitability money is converted to USD *before* aggregating:

  USD amount = native amount × (USD per 1 unit of native currency)

Rates come from Wise mid-market (same figures as
https://wise.com/in/currency-converter/cad-to-usd-rate/history/30-01-2026 )
for the *order's purchase date*. Frankfurter (ECB daily) fills days Wise
misses; a static fallback is last resort so a failed FX fetch never
silently re-mixes currencies. A December CAD order uses December's rate,
not today's. Weekends/holidays carry forward the last known rate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

import httpx

log = logging.getLogger(__name__)

_PENNY = Decimal("0.01")
TARGET_CURRENCY = "USD"
FRANKFURTER_URL = "https://api.frankfurter.dev/v1"
WISE_HISTORY_BASE = os.getenv(
    "WISE_HISTORY_BASE",
    "https://wise.com/in/currency-converter",
).rstrip("/")
WISE_FX_ENABLED = os.getenv("WISE_FX", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
_WISE_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)
_WISE_SEM = asyncio.Semaphore(8)
_WISE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AuroraFX/1.0; +https://wise.com/)"
    ),
    "Accept": "text/html,application/json",
}

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
# Per-day cache so a December CAD rate is never replaced by "today" when a
# later request loads a different window (or Frankfurter misses one range).
_DAILY_RATES: dict[tuple[str, date], float] = {}
_WISE_HITS: set[tuple[str, date]] = set()
LOOKBACK_DAYS = 14


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


def _money_codes_from_block(block) -> list[str]:
    if not isinstance(block, dict):
        return []
    code = _iso_currency(
        block.get("currencyCode") or block.get("CurrencyCode")
    )
    return [code] if code else []


def _collect_money_currencies(order: dict | None, item: dict | None) -> list[str]:
    """Every ISO code on the line / order, first-seen order preserved."""
    codes: list[str] = []
    seen: set[str] = set()

    def _add(block) -> None:
        for code in _money_codes_from_block(block):
            if code not in seen:
                seen.add(code)
                codes.append(code)

    if item:
        for field in (
            "itemSubtotal", "itemPrice", "promotionDiscount",
            "ItemPrice", "PromotionDiscount",
        ):
            _add(item.get(field))
    if order:
        _add(order.get("orderTotal") or order.get("OrderTotal"))
    return codes


def _explicit_money_currency(order: dict | None, item: dict | None) -> str:
    """Prefer a real marketplace currency over a USD fee-map stamp.

    Order sync often labels referral/itemPrice USD on Amazon.ca / .mx lines
    while ``orderTotal`` (or another field) is still CAD/MXN. Returning the
    first USD hit made 287.37 pesos look like $287.37.
    """
    codes = _collect_money_currencies(order, item)
    if not codes:
        return ""
    non_usd = [c for c in codes if c != TARGET_CURRENCY]
    if non_usd:
        return non_usd[0]
    return codes[0]


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
    if channel_ccy and (not money_ccy or money_ccy == TARGET_CURRENCY):
        # Channel wins over a missing/USD stamp so Amazon.com.mx is MXN
        # even when every money field was labelled USD by the fee map.
        if channel_ccy != TARGET_CURRENCY:
            return channel_ccy
    if money_ccy:
        return money_ccy
    if channel_ccy:
        return channel_ccy
    return fallback


def referral_fee_currency(order: dict | None, item: dict | None = None) -> str:
    """Referral is a % of line revenue — always the line's marketplace currency.

    Stored ``referralFee.currencyCode`` is often USD from the US fee map even
    when the 15% was taken of a CAD/MXN face value (B08HGLL647 24 Mar 2026:
    MXN 287.37 → referral 43.11 labelled USD).
    """
    return infer_line_currency(order, item)


def fba_fee_currency(order: dict | None, item: dict | None = None) -> str:
    """ISO code for a stored ``fulfillmentFee`` amount.

    US catalog FBA ($4.35) is stamped USD on CA/MX lines — keep USD so we
    do not treat dollars as pesos. A fee actually stored in CAD/MXN is
    converted. Referral must NOT use this helper (that stamp is a lie).
    """
    block = {}
    if item and isinstance(item, dict):
        block = item.get("fulfillmentFee") or item.get("FulfillmentFee") or {}
    stamped = money_field_currency(block, "")
    if stamped and stamped != TARGET_CURRENCY:
        return stamped
    return TARGET_CURRENCY


def looks_like_unconverted_foreign_face(
    sale_usd: float,
    listing_usd: float | None = None,
) -> bool:
    """True when a 'USD' unit price is almost certainly pesos/CAD as dollars.

    B08HGLL647 listing $13.49 vs unconverted MXN 287.37. A real $10–$50
    sale on the same SKU must still quote Fees API; 5× listing (or +$40)
    is the cut.
    """
    sale = float(sale_usd or 0)
    listing = float(listing_usd or 0)
    if sale <= 0:
        return False
    if listing > 0:
        return sale >= max(listing * 5.0, listing + 40.0)
    return sale >= 80.0


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
        """USD per 1 unit of ``currency`` on the *order* calendar date.

        Never substitutes a later day's rate (e.g. today) for a December
        order. Weekend/holiday gaps walk backward up to ``LOOKBACK_DAYS``.
        Static fallback is last resort when that date's series is missing.
        """
        cur = (currency or TARGET_CURRENCY).strip().upper() or TARGET_CURRENCY
        if cur == TARGET_CURRENCY:
            return 1.0
        on_d = _as_date(on) if on is not None else None
        if on_d is not None:
            found = self.rates.get((cur, on_d))
            if found is None:
                for i in range(1, LOOKBACK_DAYS + 1):
                    found = self.rates.get((cur, on_d - timedelta(days=i)))
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


def wise_history_url(currency: str, day: date) -> str:
    """Same URL shape as the Wise history page the seller uses."""
    cur = (currency or "").strip().lower()
    return (
        f"{WISE_HISTORY_BASE}/{cur}-to-usd-rate/history/"
        f"{day.strftime('%d-%m-%Y')}"
    )


def parse_wise_history_html(html: str) -> float | None:
    """Read mid-market USD-per-unit from Wise converter ``__NEXT_DATA__``."""
    if not html:
        return None
    m = _WISE_NEXT_DATA.search(html)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    model = ((payload.get("props") or {}).get("pageProps") or {}).get("model") or {}
    block = model.get("historicalRate") or model.get("rate") or {}
    try:
        value = float(block.get("value"))
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


async def _fetch_wise_day(
    client: httpx.AsyncClient, currency: str, day: date,
) -> tuple[date, float | None]:
    if (currency, day) in _WISE_HITS:
        cached = _DAILY_RATES.get((currency, day))
        return day, float(cached) if cached is not None else None
    url = wise_history_url(currency, day)
    async with _WISE_SEM:
        try:
            resp = await client.get(url, headers=_WISE_HEADERS)
            if resp.status_code >= 400:
                return day, None
            rate = parse_wise_history_html(resp.text)
            return day, rate
        except Exception as e:
            log.warning("Wise FX fetch failed for %s %s: %s", currency, day, e)
            return day, None


async def _fetch_wise_range(
    currency: str, start: date, end: date,
) -> dict[date, float]:
    """Daily Wise mid-market USD-per-unit for ``currency`` in [start, end]."""
    if not WISE_FX_ENABLED:
        return {}
    cur = currency.upper()
    days: list[date] = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=1)
    need = [d for d in days if (cur, d) not in _WISE_HITS]
    daily: dict[date, float] = {}
    for d in days:
        if (cur, d) in _WISE_HITS:
            hit = _DAILY_RATES.get((cur, d))
            if hit is not None:
                daily[d] = float(hit)
    if need:
        try:
            async with httpx.AsyncClient(timeout=18, follow_redirects=True) as client:
                rows = await asyncio.gather(
                    *[_fetch_wise_day(client, cur, d) for d in need]
                )
        except Exception as e:
            log.warning("Wise FX range failed for %s %s..%s: %s", cur, start, end, e)
            rows = []
        for d, rate in rows:
            if rate is None:
                continue
            daily[d] = float(rate)
            _DAILY_RATES[(cur, d)] = float(rate)
            _WISE_HITS.add((cur, d))
    return daily


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
    for d, r in filled.items():
        _DAILY_RATES.setdefault((cur, d), float(r))
    return filled


def _remember_daily(cur: str, daily: dict[date, float]) -> None:
    for d, r in daily.items():
        _DAILY_RATES.setdefault((cur, d), float(r))


def _hydrate_from_daily(cur: str, start: date, end: date) -> dict[date, float]:
    """Replay previously fetched days (and lookback) into a window."""
    out: dict[date, float] = {}
    day = start
    while day <= end:
        found = _DAILY_RATES.get((cur, day))
        if found is None:
            for i in range(1, LOOKBACK_DAYS + 1):
                found = _DAILY_RATES.get((cur, day - timedelta(days=i)))
                if found is not None:
                    break
        if found is not None:
            out[day] = float(found)
        day += timedelta(days=1)
    return out


async def load_usd_fx(
    currencies: Iterable[str],
    start: date | datetime | None,
    end: date | datetime | None,
) -> UsdFx:
    """Load USD rates for every non-USD code in ``currencies``.

    Fetches a padded window so a Saturday / Christmas order still gets the
    prior business-day rate, not today's fallback. Each order later converts
    with *its* ``purchaseDate``.
    """
    start_d = _as_date(start) or date.today()
    end_d = _as_date(end) or start_d
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    fetch_start = start_d - timedelta(days=LOOKBACK_DAYS)
    fetch_end = end_d + timedelta(days=1)
    need = sorted({
        str(c).strip().upper()
        for c in (currencies or [])
        if str(c or "").strip() and str(c).strip().upper() != TARGET_CURRENCY
    })
    if not need:
        return identity_fx()

    table = UsdFx(source="fallback")
    wise_any = False
    frank_any = False
    for cur in need:
        wise = await _fetch_wise_range(cur, fetch_start, fetch_end)
        if wise:
            wise_any = True
            _remember_daily(cur, wise)
        frank: dict[date, float] = {}
        needs_frank = not wise
        if wise:
            probe = fetch_start
            while probe <= fetch_end:
                if probe not in wise:
                    needs_frank = True
                    break
                probe += timedelta(days=1)
        if needs_frank:
            frank = await _fetch_frankfurter_range(cur, fetch_start, fetch_end)
            if frank:
                frank_any = True
                _remember_daily(cur, frank)
        cached = _hydrate_from_daily(cur, fetch_start, fetch_end)
        # Wise mid-market wins over ECB when both exist for the same day.
        merged = {**cached, **(frank or {}), **(wise or {})}
        if merged:
            merged = _fill_calendar(merged, fetch_start, fetch_end)
        for d, r in merged.items():
            table.rates[(cur, d)] = r
        if not merged and cur not in table.fallback:
            table.missing.add(cur)
    if wise_any:
        table.source = "wise"
    elif frank_any:
        table.source = "frankfurter"
    elif table.rates:
        table.source = "cache"
    else:
        table.source = "fallback"
    return table
