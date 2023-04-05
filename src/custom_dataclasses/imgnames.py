from dataclasses import dataclass
from collections.abc import Iterable


@dataclass
class ImageNames:
    imgnames: Iterable[str]
    axes_of_interest: Iterable[str]
    objnames: Iterable[str]
