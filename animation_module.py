import math
import time

from Medium import Medium
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from typing import List,Tuple
from pytictoc import TicToc

def reset_yaxis_limits(axes,y_vals,old_limits:Tuple[float,float]):
    limits = min(min(y_vals), old_limits[0]), max(max(y_vals), old_limits[1])
    limits_mid_dif = (limits[1] + limits[0]) / 2, (limits[1] - limits[0]) / 2
    axes.set_ylim(limits_mid_dif[0] - limits_mid_dif[1] * 1.1, limits_mid_dif[0] + limits_mid_dif[1] * 1.1)
    return limits


def run_animation(medium: Medium,fps:float,b_draw_u=True,b_draw_v=False,pause_at_start=True,f_edit_plot=None,on_animation_update=None):
    tic_toc = TicToc()
    target_simulation_time = 0

    lineu ,axu ,linev ,axv ,fig = medium.plot(b_draw_u,b_draw_v)
    re = []
    if f_edit_plot:
        re += f_edit_plot(lineu ,axu ,linev ,axv ,fig)
    if pause_at_start:
        plt.pause(0.5)
    limitsu,limitsv = (float("inf"), -1 * float("inf")),(float("inf"), -1 * float("inf"))
    if b_draw_u:
        re.append(lineu)
    if b_draw_v:
        re.append(linev)

    def update_anim(frame):
        nonlocal limitsu,limitsv,target_simulation_time
        target_simulation_time += tic_toc.tocvalue()
        target_simulation_time = 0 if math.isnan(target_simulation_time) else target_simulation_time
        medium.advance_to_time(target_simulation_time)
        tic_toc.tic()
        if b_draw_u:
            lineu.set_data(medium.x, medium.u)
            limitsu = reset_yaxis_limits(axu, medium.u, limitsu)
        if b_draw_v:
            linev.set_data(medium.x, medium.v)
            limitsv = reset_yaxis_limits(axv, medium.v, limitsv)
        if on_animation_update is not None:
            on_animation_update()
        return re

    anim = FuncAnimation(fig, update_anim, interval=1000 / fps)
    plt.show()
