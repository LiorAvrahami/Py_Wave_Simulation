import numpy as np
from abc import ABC,abstractmethod
from typing import List, Callable, Sequence, Any, Tuple, TYPE_CHECKING,Union,Literal
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from itertools import count
if TYPE_CHECKING:
    from Medium import Medium

class BounderyCreationFunctor():
    def __init__(self,func:Callable[["Medium"],List["BoundaryCondition"]]):
        self.func = func

    def __call__(self,medium:"Medium") -> List["BoundaryCondition"]:
        return self.func(medium)

    @staticmethod
    def create_from_list(list:List["BounderyCreationFunctor"], medium:"Medium") -> List["BoundaryCondition"]:
        return [cond for conditions_creator in list for cond in conditions_creator(medium)]


def limited_segment_condition_creator(low_u:float, high_u:float, x_start:float = None, x_stop:float = None, x_start_index:float = None, x_stop_index:float = None) -> List[BounderyCreationFunctor]:
    def creat(medium:"Medium") -> List["BoundaryCondition"]:
        nonlocal x_start_index,x_stop_index,x_start,x_stop
        x_start_index = np.argmax(medium.x > x_start) if x_start_index is None else x_start_index
        x_stop_index = np.argmax(medium.x > x_stop) if x_stop_index is None else x_stop_index
        return [LimmitedY_SegmentCondition(medium,x_start_index,x_stop_index,low_u,high_u)]
    return [BounderyCreationFunctor(creat)]

def hard_boundary_conditions_creator(u_left:float = None,u_right:float = None) -> List[BounderyCreationFunctor]:
    def creat(medium: "Medium") -> List["BoundaryCondition"]:
        nonlocal u_left,u_right
        ret = []
        u_left = medium.u[0] if u_left is None else u_left
        u_right = medium.u[-1] if u_right is None else u_right
        ret.append(StationarySegmentCondition(medium,None, 1, u_left))
        ret.append(StationarySegmentCondition(medium,-1, None, u_right))
        return ret
    return [BounderyCreationFunctor(creat)]

def open_boundary_conditions_creator() -> List[BounderyCreationFunctor]:
    def creat(medium: "Medium") -> List["BoundaryCondition"]:
        ret = []
        ret.append(LeftOpenBounderyCondition(medium))
        ret.append(RightOpenBounderyCondition(medium))
        return ret
    return [BounderyCreationFunctor(creat)]

class BoundaryCondition(ABC):
    medium_host: "Medium"

    def __init__(self,medium_host: "Medium"):
        self.medium_host = medium_host

    @abstractmethod
    def apply_condition(self,u:np.ndarray,v:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:...

    def draw(self,axes):...


class LeftOpenBounderyCondition(BoundaryCondition):
    def apply_condition(self, u: np.ndarray, v: np.ndarray):
        u[0] = u[1]
        v[0] = v[1]

class RightOpenBounderyCondition(BoundaryCondition):
    def apply_condition(self, u: np.ndarray, v: np.ndarray):
        u[-1] = u[-2]
        v[-1] = v[-2]

class SegmentCondition(BoundaryCondition):
    lagrangian_slice: slice

    def __init__(self, medium_host: "Medium", inclusive_start_index, exclusive_end_index):
        super().__init__(medium_host)
        self.lagrangian_slice = slice(inclusive_start_index,exclusive_end_index,1)

    @property
    def lagrangian_range(self):
        return range(len(self.medium_host.x))[self.lagrangian_slice]

class StationarySegmentCondition(SegmentCondition):
    u0: float
    def __init__(self, medium_host: "Medium", inclusive_start_index, exclusive_end_index, u0):
        super().__init__(medium_host,inclusive_start_index, exclusive_end_index)
        self.u0 = u0

    def apply_condition(self,u:np.ndarray,v:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        u[self.lagrangian_slice] = self.u0
        v[self.lagrangian_slice] = 0
        return u,v

    def draw(self,axes:Axes):
        lagrangian_range = self.lagrangian_range
        x0 = self.medium_host.x[min(lagrangian_range)]
        x1 = self.medium_host.x[max(lagrangian_range)]
        axes.scatter([x0,x1],[self.u0,self.u0],marker='x',c=(0.7,0.2,0.1,0.5),s=70)
        # rect_low = Rectangle((x_min,y_min_low),width,height,color=(0.7,0.2,0.1))
        # rect_high = Rectangle((x_min, y_min_high), width, height,color=(0.7,0.2,0.1))
        # axes.add_artist(rect_low)
        # axes.add_artist(rect_high)

class LimmitedY_SegmentCondition(SegmentCondition):
    umin: float
    umax: float
    def __init__(self, medium_host: "Medium", inclusive_start_index, exclusive_end_index, umin,umax):
        super().__init__(medium_host,inclusive_start_index, exclusive_end_index)
        self.umin = umin
        self.umax = umax

    def apply_condition(self,y:np.ndarray,v:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        v[self.lagrangian_slice] = v[self.lagrangian_slice] * ((y[self.lagrangian_slice] <= self.umax)*(y[self.lagrangian_slice] >= self.umin))
        y[self.lagrangian_slice] = np.maximum(np.minimum(y[self.lagrangian_slice],self.umax),self.umin)
        return y,v

    def draw(self,axes:Axes,height=100):
        lagrangian_range = self.lagrangian_range
        x_min = self.medium_host.x[min(lagrangian_range)]
        width = self.medium_host.x[max(lagrangian_range)] - x_min
        y_min_low = self.umin - height
        y_min_high = self.umax
        rect_low = Rectangle((x_min,y_min_low),width,height,color=(0.7,0.2,0.1,0.5))
        rect_high = Rectangle((x_min, y_min_high), width, height,color=(0.7,0.2,0.1,0.5))
        axes.add_artist(rect_low)
        axes.add_artist(rect_high)