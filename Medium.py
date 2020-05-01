# from Boundery_Conditions import BoundaryCondition
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Callable, Sequence, Any,Tuple

class BoundaryCondition:
    pass

class Medium:
    c : float
    time:float
    x : np.ndarray
    dx : np.ndarray
    y : np.ndarray
    v : np.ndarray
    boundery_conditions: Sequence[BoundaryCondition]
    def __init__(self,x=None,y=None,v=None,c=1):
        self.c = 1
        self.time = 0
        self.x = np.linspace(0, 1, 1000) if x is None else x
        self.dx = self.x[1] - self.x[0]
        self.y = np.zeros(self.x.shape) if y is None else y(self.x)
        self.v = np.zeros(self.x.shape) if v is None else v(self.x)

    def step(self, dt=None):
        if dt == None:
            dt = (np.min(self.dx) / self.c) * 0.5
        dv = self.c ** 2 * np.gradient(np.gradient(self.y, self.x), self.x) * dt
        self.y += (self.v + dv) / 2 * dt
        self.v += dv
        self.v[0], self.v[-1] = 0,0
        self.y[0], self.y[-1] = 0,0
        self.time += dt

    def several_steps(self, n, dt=None):
        for i in range(n):
            self.step(dt)

    def plot(self,**kwargs) -> List[Line2D]:
        return plt.plot(self.x, self.y,**kwargs)


