from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
from dataclasses import dataclass

@dataclass
class TrainConfig:
    pass

class Trainer:
    def __init__(self, conf: TrainConfig):
        pass