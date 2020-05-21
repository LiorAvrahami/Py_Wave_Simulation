from Medium import Medium
from TicToc import tic,toc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from typing import List,Tuple

def reset_yaxis_limits(axes,y_vals,old_limits:Tuple[float,float]):
    limits = min(min(y_vals), old_limits[0]), max(max(y_vals), old_limits[1])
    limits_mid_dif = (limits[1] + limits[0]) / 2, (limits[1] - limits[0]) / 2
    axes.set_ylim(limits_mid_dif[0] - limits_mid_dif[1] * 1.1, limits_mid_dif[0] + limits_mid_dif[1] * 1.1)
    return limits


def run_animation(medium: Medium,fps:float,b_draw_u=True,b_draw_v=False):
    lineu ,axu ,linev ,axv ,fig = medium.plot(b_draw_u,b_draw_v)
    limitsu,limitsv = (float("inf"), -1 * float("inf")),(float("inf"), -1 * float("inf"))
    re = []
    if b_draw_u:
        re.append(lineu)
    if b_draw_v:
        re.append(linev)
    tic()
    def update_anim(frame):
        nonlocal limitsu,limitsv
        toc(False)
        tic("calc")
        medium.several_steps(10)#TODO control time flow - pass final time to reach and not number of steps
        if b_draw_u:
            lineu.set_data(medium.x, medium.u)
            limitsu = reset_yaxis_limits(axu, medium.u, limitsu)
        if b_draw_v:
            linev.set_data(medium.x, medium.v)
            limitsv = reset_yaxis_limits(axv, medium.v, limitsv)
        print(medium.energy_Tot,medium.energy_K,medium.energy_U)
        toc(False)
        tic("wait")
        return re

    anim = FuncAnimation(fig, update_anim, interval=1000 / fps)
    plt.show()
