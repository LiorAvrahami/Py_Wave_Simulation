# from Boundery_Conditions import BoundaryCondition
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from TicToc import tic,toc
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

m = Medium(y = lambda x:np.sin(x*(np.pi)*1))
# m = Medium(v =lambda x: 0.05*np.sign(x - 0.5))

fig,ax = plt.subplots()
plt.legend()
plt.grid()
line = m.plot()[0]
ylims = float("inf"),-1*float("inf")
tic()
def update_anim(frame):
    global ylims
    toc()
    tic("calc")
    m.several_steps(30)
    line.set_data(m.x,m.y)
    ylims = min(min(line.get_ydata()),ylims[0]), max(max(line.get_ydata()),ylims[1])
    ylims_mid_dif = (ylims[1]+ylims[0])/2, (ylims[1]-ylims[0])/2
    ax.set_ylim(ylims_mid_dif[0]-ylims_mid_dif[1]*1.1,ylims_mid_dif[0]+ylims_mid_dif[1]*1.1)
    toc()
    tic("wait")



anim = FuncAnimation(fig,update_anim,interval=1000/40)


