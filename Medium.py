from Boundery_Conditions import hard_boundary_conditions_creator,open_boundary_conditions_creator,BoundaryCondition,BounderyCreationFunctor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Callable, Sequence, Any,Tuple, Optional

#TODO smooth out velocity
#TODO Furier transform
#TODO control time flow - until this is done c is without meaning
#TODO implement pause
#TODO implement pause

class Medium:
    c : float
    time:float
    x : np.ndarray
    dx : np.ndarray
    u : np.ndarray
    v : np.ndarray
    boundery_conditions: Sequence[BoundaryCondition]
    def __init__(self,x=None,u=None,v=None,c=1,boundery_conditions_generators:List[BounderyCreationFunctor]=None):
        self.c = 1
        self.time = 0
        self.x = np.linspace(0, 1, 500) if x is None else x
        self.dx = self.x[1] - self.x[0]
        self.u = np.zeros(self.x.shape) if u is None else u(self.x)
        self.v = np.zeros(self.x.shape) if v is None else v(self.x)
        boundery_conditions_generators = hard_boundary_conditions_creator() if boundery_conditions_generators is None else boundery_conditions_generators
        self.boundery_conditions = BounderyCreationFunctor.create_from_list(boundery_conditions_generators,self)

    @property
    def energy_K(self) -> float:
        return sum(0.5*self.v**2)

    @property
    def energy_U(self) -> float:
        return 0.5*self.c**2*sum(np.gradient(self.u, self.x)**2)

    @property
    def energy_Tot(self) -> float:
        return self.energy_K + self.energy_U

    def step(self, dt=None):
        if dt == None:
            dt = (np.min(self.dx) / self.c) * 0.5
        dv = (self.c ** 2) * np.gradient(np.gradient(self.u, self.x), self.x) * dt
        self.u += (self.v + dv) * dt
        self.v += dv
        for boundery_condition in self.boundery_conditions:
            boundery_condition.apply_condition(self.u,self.v)
        self.time += dt

    def several_steps(self, n, dt=None):
        for i in range(n):
            self.step(dt)

    def plot(self,b_draw_u=True,b_draw_v=True,**kwargs) -> Tuple[Optional[Line2D],Optional[Axes],Optional[Line2D],Optional[Axes],Figure]:
        fig, ax1 = plt.subplots()
        l1, l2, ax2 = None,None,None
        plt.legend()
        plt.grid()
        if b_draw_u:
            l1 = plt.plot(self.x, self.u,"b",**kwargs)[0]
            for cond in self.boundery_conditions:
                cond.draw(ax1)
        if b_draw_v:
            if b_draw_u:
                ax2 = plt.twinx()
                l2 = plt.plot(self.x, self.v,"g", **kwargs)[0]
            else:
                l1 = plt.plot(self.x, self.v, "g", **kwargs)[0]

        # return lines to be updated in animation
        if b_draw_u and b_draw_v:
            return l1,ax1,l2,ax2,fig
        elif b_draw_u and (not b_draw_v):
            return l1,ax1,None,None,fig
        elif (not b_draw_u) and b_draw_v:
            return None,None,l1,ax1,fig
        elif (not b_draw_u) and (not b_draw_v):
            return None, None, None, None, fig



