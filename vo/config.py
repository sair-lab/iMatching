from dataclasses import dataclass

from omegaconf import MISSING

from .mapping.mapper import MapConfig
from .tracking.tracker import TrackerConfig


@dataclass
class VOConfig:
    tracker: TrackerConfig = MISSING
    mapper: MapConfig = MISSING
