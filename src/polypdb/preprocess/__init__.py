"""Dataset-specific preprocessing utilities."""

from .piba import convert_to_standard_layout as convert_piba
from .polar import convert_to_standard_layout as convert_polar
from .sun import convert_to_standard_layout as convert_sun
try:  # pragma: no cover - optional dependency
    from .piccolo import convert_to_standard_layout as convert_piccolo
except Exception:  # pragma: no cover
    convert_piccolo = None
from .polypgen import convert_to_standard_layout as convert_polypgen

CONVERTERS = {
    "piba": convert_piba,
    "polar": convert_polar,
    "sun": convert_sun,
    "polypgen": convert_polypgen,
}
if convert_piccolo is not None:
    CONVERTERS["piccolo"] = convert_piccolo
"""Mapping of dataset identifiers to their conversion functions."""
