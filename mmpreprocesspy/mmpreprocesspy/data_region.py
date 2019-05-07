from dataclasses import dataclass


@dataclass
class DataRegion:
    start: int = None
    end: int = None
    width: int = None
