"""LightGBM benchmark path — parallel to the Prophet/naive production path.

This package is imported lazily by the /forecasting/sku/{sku}/compare
endpoint (gated behind the LGBM_BENCHMARK env flag). Nothing in the
production forecast refresh touches it — the naive/Prophet code in
forecasting.model remains authoritative.
"""
