import numpy as np
from abc import ABC,abstractmethod
from typing import List, Callable, Sequence, Any, Tuple, TYPE_CHECKING,Union,Literal
if TYPE_CHECKING:
    from Medium import Medium

def create_hard_boundary_conditions(medium:"Medium" = None,y_left:float = None,y_right:float = None) -> Sequence["BoundaryCondition"]:
    ret = []
    y_left = medium.y[0] if y_left is None else y_left
    y_right = medium.y[-1] if y_right is None else y_right
    ret.append(StationaryPointCondition(None,1,y_left))
    ret.append(StationaryPointCondition(-1,None,y_right))
    return ret

def create_open_boundary_conditions() -> Sequence["BoundaryCondition"]:
    ret = []
    ret.append(LeftOpenBounderyCondition())
    ret.append(RightOpenBounderyCondition())
    return ret

class BoundaryCondition(ABC):

    @abstractmethod
    def apply_condition(self,y:np.ndarray,v:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:...


class StationaryPointCondition(BoundaryCondition):
    lagrangian_slice: slice
    y0: float
    def __init__(self,inclusive_start_index, exclusive_end_index,y0):
        self.lagrangian_slice = slice(inclusive_start_index,exclusive_end_index,1)
        self.y0 = y0

    def apply_condition(self,y:np.ndarray,v:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        y[self.lagrangian_slice] = self.y0
        v[self.lagrangian_slice] = 0
        return y,v

class LeftOpenBounderyCondition(BoundaryCondition):
    def apply_condition(self, y: np.ndarray, v: np.ndarray):
        y[0] = y[1]
        v[0] = v[1]

class RightOpenBounderyCondition(BoundaryCondition):
    def apply_condition(self, y: np.ndarray, v: np.ndarray):
        y[-1] = y[-2]
        v[-1] = v[-2]