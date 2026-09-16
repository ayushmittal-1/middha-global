"""Audit every seller × month: FX first, then net-of-returns referral/FBA.

Uses Mongo orders + products.fees + Frankfurter. Does not call SP-API.
Run from aiModel/backend:

    python scripts/audit_all_asin_fee_math.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
from dotenv import load_dotenv

load_dotenv(BACKEND / ".env")
load_dotenv(BACKEND.parent / ".env")

PHASE_ASIN = "B07H4S83D8"
WATCH_ASINS = {PHASE_ASIN, "B07TJT135V"}
NON_USD = {"CAD", "MXN", "EUR", "GBP", "AUD", "JPY", "BRL", "INR"}
MX_MP = "A1AM78C64UM0Y8"
CA_MP = "A2EUQ1WTGCTBG2"


def _month_starts(min_dt: datetime, max_dt: datetime, tz_name: str) -> list[date]:
    tz = ZoneInfo(tz_name or "UTC")
    local_min = min_dt.astimezone(tz).date().replace(day=1)
    local_max = max_dt.astimezone(tz).date()
    out: list[date] = []
    y, m = local_min.year, local_min.month
    while date(y, m, 1) <= local_max:
        out.append(date(y, m, 1))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _last_day(d: date) -> date:
    return _next_month(d) - timedelta(days=1)


def _peeled_fba(pf: dict | None, include_fuel: bool) -> tuple[float, float]:
    from amazon_sp import split_bundled_fulfillment_total

    if not pf:
        return 0.0, 0.0
    fba_u = float(pf.get("fba_per_unit") or 0)
    fuel_u = float(pf.get("fuel_per_unit") or 0)
    ful_u = float(pf.get("fulfillment_per_unit") or 0)
    if not include_fuel:
        if fuel_u > 0:
            return round(fba_u, 2), 0.0
        bundled = ful_u if ful_u > 0 else fba_u
        if bundled > 0:
            base, _ = split_bundled_fulfillment_total(bundled)
            return base, 0.0
        return 0.0, 0.0
    if fuel_u <= 0 and (ful_u > 0 or fba_u > 0):
        return split_bundled_fulfillment_total(ful_u if ful_u > 0 else fba_u)
    return round(fba_u, 2), round(fuel_u, 2)


async def main() -> None:
    from auth import _db
    from aurora_data import (
        _pack_product_fee_doc,
        _usd_listing_fba_band,
        aggregate_sku_metrics_from_orders,
        fba_returns_by_sku,
        is_excluded_order_status,
        line_item_quantity,
        line_item_sales_amount,
        recompute_referral_totals,
        snap_referral_rate,
    )
    from agent import (
        _apply_returns_to_sku_data,
        apply_sale_price_fba,
        fuel_surcharge_applies_for_window,
        resolve_sku_referral_fba_fuel,
        usd_fba_price_band,
    )
    from currency_fx import (
        identity_fx,
        infer_line_currency,
        infer_marketplace_id,
        load_usd_fx,
    )
    from marketplace_timezone import parse_date_range_for_query, resolve_dashboard_timezone

    db = _db()
    print("=== inventory ===", flush=True)
    n_orders = await db.orders.estimated_document_count()
    n_products = await db.products.estimated_document_count()
    bounds = await db.orders.aggregate(
        [
            {
                "$group": {
                    "_id": None,
                    "min": {"$min": "$purchaseDate"},
                    "max": {"$max": "$purchaseDate"},
                    "sellers": {"$addToSet": "$sellerId"},
                }
            }
        ]
    ).to_list(1)
    if not bounds:
        print("no orders")
        return
    min_dt = bounds[0]["min"]
    max_dt = bounds[0]["max"]
    if min_dt.tzinfo is None:
        min_dt = min_dt.replace(tzinfo=timezone.utc)
    if max_dt.tzinfo is None:
        max_dt = max_dt.replace(tzinfo=timezone.utc)
    seller_ids = [s for s in bounds[0]["sellers"] if s is not None]
    print(
        f"orders~{n_orders} products~{n_products} sellers={len(seller_ids)} "
        f"purchaseDate {min_dt.isoformat()} -> {max_dt.isoformat()}",
        flush=True,
    )

    print("loading products.fees ...", flush=True)
    fees_by_seller_sku: dict[tuple[str, str], dict] = {}
    siblings_by_seller_asin: dict[tuple[str, str], list[dict]] = defaultdict(list)
    band_by_asin: dict[str, dict[str, tuple[float, float]]] = {}
    async for doc in db.products.find(
        {},
        {"sellerId": 1, "sku": 1, "asin": 1, "fees": 1, "price": 1, "fulfillmentType": 1},
    ):
        packed = _pack_product_fee_doc(doc)
        if not packed:
            continue
        sid = str(doc.get("sellerId") or "")
        sku = str(doc.get("sku") or "")
        asin = str(packed.get("asin") or doc.get("asin") or "").strip().upper()
        if sid and sku:
            fees_by_seller_sku[(sid, sku.lower())] = packed
        if sid and asin:
            siblings_by_seller_asin[(sid, asin)].append(packed)
        listing = float(packed.get("listing_price") or 0)
        fba = float(packed.get("fba_per_unit") or 0)
        fuel = float(packed.get("fuel_per_unit") or 0)
        if asin and listing > 0 and fba > 0:
            band = _usd_listing_fba_band(listing)
            dest = band_by_asin.setdefault(asin, {})
            dest.setdefault(band, (round(fba, 2), round(fuel, 2)))
    print(
        f"fee cards={len(fees_by_seller_sku)} asin bands={len(band_by_asin)}",
        flush=True,
    )

    print("preloading FX (CAD/MXN/EUR/GBP/AUD) for full history ...", flush=True)
    fx_global = await load_usd_fx(NON_USD, min_dt, max_dt)
    print(f"fx source={fx_global.source} rates={len(fx_global.rates)}", flush=True)

    users_by_id: dict[str, dict] = {}
    async for u in db.users.find(
        {},
        {"amazonMarketplaceIds": 1, "marketplace": 1, "email": 1},
    ):
        users_by_id[str(u["_id"])] = u

    hard: list[dict] = []
    watches: list[dict] = []
    stats = defaultdict(int)
    extra_fx: dict[frozenset[str], object] = {}

    for i, seller in enumerate(sorted(seller_ids, key=str), 1):
        sid = str(seller)
        user = users_by_id.get(sid) or {"_id": seller}
        tz_name = resolve_dashboard_timezone(user) or "UTC"
        months = _month_starts(min_dt, max_dt, tz_name)
        print(
            f"[{i}/{len(seller_ids)}] seller ...{sid[-6:]} tz={tz_name} months={len(months)}",
            flush=True,
        )
        for month in months:
            start_s = month.isoformat()
            end_s = _last_day(month).isoformat()
            start_dt, end_dt = parse_date_range_for_query(start_s, end_s, tz_name)
            if start_dt is None or end_dt is None:
                continue
            orders = [
                o
                async for o in db.orders.find(
                    {
                        "sellerId": seller,
                        "purchaseDate": {"$gte": start_dt, "$lte": end_dt},
                    }
                )
            ]
            if not orders:
                continue
            stats["seller_months"] += 1
            stats["orders"] += len(orders)

            currencies: set[str] = set()
            for o in orders:
                if is_excluded_order_status(o.get("orderStatus")):
                    continue
                for it in o.get("orderItems") or []:
                    currencies.add(infer_line_currency(o, it))
            need = {c for c in currencies if c and c != "USD"}
            fx = fx_global
            if need - NON_USD:
                key = frozenset(need)
                if key not in extra_fx:
                    extra_fx[key] = await load_usd_fx(need, start_dt, end_dt)
                fx = extra_fx[key]

            sku_data, _, _ = aggregate_sku_metrics_from_orders(orders, fx=fx)
            naive, _, _ = aggregate_sku_metrics_from_orders(orders, fx=identity_fx())
            naive_ordered_mp = {
                s: dict((d.get("units_by_marketplace") or {}))
                for s, d in naive.items()
            }
            if not sku_data:
                continue

            rate_by_sku: dict[str, float] = {}
            pf_by_sku: dict[str, dict] = {}
            for sku, d in sku_data.items():
                pf = fees_by_seller_sku.get((sid, str(sku).lower()))
                asin = str(d.get("asin") or (pf or {}).get("asin") or "").upper()
                if not pf and asin:
                    sibs = siblings_by_seller_asin.get((sid, asin)) or []
                    pf = next((p for p in sibs if float(p.get("fba_per_unit") or 0) > 0), None)
                    if not pf and sibs:
                        pf = sibs[0]
                if pf:
                    pf_by_sku[sku] = pf
                    rate_by_sku[sku] = snap_referral_rate(
                        pf.get("listing_price"), pf.get("referral_per_unit"),
                    )
                else:
                    rate_by_sku[sku] = 0.15

            recompute_referral_totals(orders, sku_data, rate_by_sku, fx=fx)
            recompute_referral_totals(orders, naive, rate_by_sku, fx=identity_fx())
            returns = await fba_returns_by_sku(
                {"_id": seller}, start_dt, end_dt, fx=fx,
            )
            _apply_returns_to_sku_data(sku_data, returns, rate_by_sku)
            _apply_returns_to_sku_data(naive, returns, rate_by_sku)

            include_fuel = fuel_surcharge_applies_for_window(
                display_start=start_s,
                display_end=end_s,
                start_dt=start_dt,
                end_dt=end_dt,
                marketplace_tz=tz_name,
            )

            for sku, d in sku_data.items():
                units = int(d.get("units") or 0)
                if units <= 0:
                    continue
                stats["sku_months"] += 1
                asin = str(d.get("asin") or "").upper()
                if asin:
                    stats["asin_months"] += 1
                ordered = int(d.get("ordered_units") or units)
                returned = int(d.get("returned_units") or 0)
                net = int(
                    d.get("net_units")
                    if d.get("net_units") is not None
                    else max(0, ordered - returned)
                )
                bill = net if net > 0 else 0
                converted_rev = float(d.get("revenue") or 0)
                naive_rev = float((naive.get(sku) or {}).get("revenue") or 0)
                rate = float(rate_by_sku.get(sku) or 0.15)
                mp_units = d.get("units_by_marketplace") or {}
                mx_u = int(mp_units.get(MX_MP) or 0)
                ca_u = int(mp_units.get(CA_MP) or 0)
                fx_gap = abs(converted_rev - naive_rev)
                has_fx = fx_gap > 0.5
                if has_fx:
                    stats["fx_affected"] += 1
                if returned > 0:
                    stats["with_returns"] += 1
                if mx_u or ca_u:
                    stats["foreign_mp"] += 1

                pf = pf_by_sku.get(sku)
                ref, fba, fuel, _src = resolve_sku_referral_fba_fuel(
                    line_referral=float(d.get("referral_total") or 0),
                    line_fba=float(d.get("fba_total") or 0),
                    bill_units=bill,
                    revenue=converted_rev,
                    product_fees=pf,
                    fee_estimate=None,
                    include_fuel=include_fuel,
                )
                catalog_fba, catalog_fuel = _peeled_fba(pf, include_fuel)
                asin_bands = band_by_asin.get(asin) or {}
                fba_per_band = {k: v[0] for k, v in asin_bands.items()}
                fuel_per_band = {k: v[1] for k, v in asin_bands.items()}
                if catalog_fba > 0 and pf:
                    listing_band = _usd_listing_fba_band(
                        float(pf.get("listing_price") or 0)
                    )
                    fba_per_band.setdefault(listing_band, catalog_fba)
                    if include_fuel:
                        fuel_per_band.setdefault(listing_band, catalog_fuel)
                rebuilt = apply_sale_price_fba(
                    bill_units=bill,
                    units_by_usd_price=d.get("units_by_usd_price"),
                    fba_per_usd_price={},
                    catalog_fba_per_unit=catalog_fba,
                    include_fuel=include_fuel,
                    fuel_per_usd_price={},
                    catalog_fuel_per_unit=catalog_fuel,
                    fba_per_band=fba_per_band,
                    fuel_per_band=fuel_per_band,
                )
                if rebuilt is not None:
                    fba, fuel = rebuilt

                issues: list[str] = []
                net_ref = round(converted_rev * rate, 2)
                native_ref = round(naive_rev * rate, 2)
                # Peso-as-dollar leak: referral matches 15% of unconverted net
                # AND misses 15% of converted by more than FX rounding.
                # Ignore tiny FX gaps where per-line rounding on thousands of
                # units is larger than 15% of the FX delta.
                if has_fx and converted_rev > 1 and fx_gap >= 5.0:
                    native_hit = abs(ref - native_ref) <= 0.51
                    converted_miss = abs(ref - net_ref) > max(2.0, 0.15 * fx_gap)
                    if native_hit and converted_miss:
                        issues.append("referral_tracks_native_not_usd")
                    if abs(converted_rev - naive_rev) < 0.01:
                        issues.append("fx_not_applied")
                if returned > 0 and catalog_fba > 0 and bill > 0:
                    fba_net = round(catalog_fba * bill, 2)
                    fba_ship = round(catalog_fba * ordered, 2)
                    if (
                        abs(fba_ship - fba_net) > 0.05
                        and abs(fba - fba_ship) <= 0.05
                        and abs(fba - fba_net) > 0.5
                    ):
                        issues.append("fba_billed_shipped_not_net")
                if returned > 0 and ordered > 0:
                    if net != max(0, ordered - returned):
                        issues.append("net_units_mismatch")
                    if int(d.get("units") or 0) != ordered:
                        issues.append("units_column_not_ordered")
                ordered_mp = naive_ordered_mp.get(sku) or {}
                naive_mx = int(ordered_mp.get(MX_MP) or 0)
                ret_row = returns.get(sku) or {}
                if not ret_row:
                    sku_l = str(sku).lower()
                    ret_row = next(
                        (v for k, v in returns.items() if str(k).lower() == sku_l),
                        {},
                    )
                ret_mx = int(
                    (ret_row.get("returned_units_by_marketplace") or {}).get(MX_MP)
                    or 0
                )
                expected_mx = max(0, naive_mx - ret_mx)
                if expected_mx > 0 and mx_u == 0:
                    issues.append("mexico_units_dropped")

                off_band = False
                by_price = d.get("units_by_usd_price") or {}
                listing_band = None
                if pf:
                    listing_band = _usd_listing_fba_band(
                        float(pf.get("listing_price") or 0)
                    )
                for price, qty in by_price.items():
                    if int(qty or 0) <= 0:
                        continue
                    band = usd_fba_price_band(float(price))
                    if listing_band and band != listing_band and band not in fba_per_band:
                        off_band = True
                if off_band:
                    stats["off_band_no_sibling"] += 1

                if not pf:
                    stats["missing_catalog_fees"] += 1

                row = {
                    "seller": sid[-6:],
                    "month": start_s[:7],
                    "sku": sku,
                    "asin": asin,
                    "ordered": ordered,
                    "returned": returned,
                    "net": net,
                    "mx": mx_u,
                    "ca": ca_u,
                    "converted_rev": round(converted_rev, 2),
                    "naive_rev": round(naive_rev, 2),
                    "fx_gap": round(fx_gap, 2),
                    "referral": round(ref, 2),
                    "fba": round(fba, 2),
                    "fuel": round(fuel, 2),
                    "rate": rate,
                    "catalog_fba": catalog_fba,
                    "issues": issues,
                }
                if issues:
                    stats["hard_fail_skus"] += 1
                    hard.append(row)
                if asin in WATCH_ASINS:
                    watches.append(row)
                    stats["watch"] += 1

    print("\n=== GLOBAL RULE CHECK ===", flush=True)
    print(json.dumps(dict(stats), indent=2, default=str), flush=True)

    print("\n=== watched ASINs (Phase Card / UNO) ===", flush=True)
    for w in watches:
        print(json.dumps(w, default=str), flush=True)

    print(f"\n=== hard failures ({len(hard)}) ===", flush=True)
    for row in hard[:80]:
        print(json.dumps(row, default=str), flush=True)
    if len(hard) > 80:
        print(f"... {len(hard) - 80} more", flush=True)

    out_path = BACKEND / "scripts" / "_audit_all_asin_fee_math.json"
    out_path.write_text(
        json.dumps({"stats": dict(stats), "hard": hard, "watch": watches}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}", flush=True)
    if hard:
        print("NOT once-for-all yet — hard failures above.", flush=True)
        raise SystemExit(1)
    print(
        "PASS: no peso-as-dollar / shipped-units / dropped-Mexico leaks "
        "on any seller-month SKU.",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
