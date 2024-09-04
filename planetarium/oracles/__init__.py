__all__ = ["blocksworld", "gripper", "rover_single", "floortile"]

from . import oracle

from . import blocksworld
from . import gripper
from . import rover_single
from . import floortile

ORACLES: dict[str, oracle.Oracle] = {
    "blocksworld": blocksworld.BlocksworldOracle,
    "gripper": gripper.GripperOracle,
    "rover-single": rover_single.RoverSingleOracle,
    "floor-tile": floortile.FloorTileOracle,
}
