"""Dataset-specific preprocessing utilities."""

from .piba import PibaConverter, convert_to_standard_layout as convert_piba
from .polar import PolarConverter, convert_to_standard_layout as convert_polar
from .sun import SunConverter, convert_to_standard_layout as convert_sun
try:  # pragma: no cover - optional dependency
    from .piccolo import PiccoloConverter, convert_to_standard_layout as convert_piccolo
except Exception:  # pragma: no cover
    PiccoloConverter = None
    convert_piccolo = None
from .polypgen import PolypGenConverter, convert_to_standard_layout as convert_polypgen

CONVERTERS = {
    "piba": PibaConverter,
    "polar": PolarConverter,
    "sun": SunConverter,
    "polypgen": PolypGenConverter,
}
if PiccoloConverter is not None:
    CONVERTERS["piccolo"] = PiccoloConverter
"""Mapping of dataset identifiers to their conversion functions."""
