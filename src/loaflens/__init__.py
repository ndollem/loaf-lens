"""LoafLens: deteksi roti kempes di konveyor (POC, CPU-only)."""

import logging

__version__ = "0.0.1"


def setup_logging(level: str = "INFO") -> None:
    """Konfigurasi logging standar untuk CLI LoafLens."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
