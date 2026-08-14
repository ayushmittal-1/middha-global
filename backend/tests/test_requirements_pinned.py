"""All runtime dependencies must be exact-pinned (audit C5).

Un-pinned deps mean the next `pip install` could pull a breaking or
malicious upstream release. This test greps requirements.txt for the
`==` constraint on every non-comment, non-blank line."""

from pathlib import Path


REQ_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def test_requirements_file_exists():
    assert REQ_PATH.exists()


def test_every_package_line_is_exact_pinned():
    """Each real requirement line must contain '=='. Blank lines,
    comments, and lines starting with -r / --index-url are exempt."""
    unpinned: list[str] = []
    for raw in REQ_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "-c", "--")):
            continue
        # Extras syntax like `uvicorn[standard]==0.34.0` still has ==.
        if "==" not in line:
            unpinned.append(line)
    assert not unpinned, f"unpinned requirement(s): {unpinned!r}"


def test_slowapi_is_present_for_c2_rate_limiter():
    text = REQ_PATH.read_text().lower()
    assert "slowapi==" in text
