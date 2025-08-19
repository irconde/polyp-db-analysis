"""Dataset-specific preprocessing utilities."""

from .piba import convert_to_standard_layout as convert_piba
from .polar import convert_to_standard_layout as convert_polar
from .piccolo import convert_to_standard_layout as convert_piccolo
from .sun import convert_to_standard_layout as convert_sun

CONVERTERS = {
    "piba": convert_piba,
    "polar": convert_polar,
    "piccolo": convert_piccolo,
    "sun": convert_sun,
}
