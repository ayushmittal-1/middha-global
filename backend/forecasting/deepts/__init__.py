"""Deep time-series forecasters: DeepAR + Temporal Fusion Transformer.

Wraps gluonts's torch-backed estimators in the same call shape as
`forecasting.lgbm.model` so the picker treats them as sibling
candidates. Heavy: requires torch + gluonts + pytorch-lightning
(all in requirements-dev.txt). Training uses MPS on Apple Silicon
and CUDA on Linux/NVIDIA; falls back to CPU.
"""
