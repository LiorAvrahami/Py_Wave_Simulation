import math
import time

import numpy as np

from Medium import Medium
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from typing import List, Tuple
from pytictoc import TicToc


def reset_yaxis_limits(axes, y_vals, old_limits: Tuple[float, float]):
    limits = min(min(y_vals), old_limits[0]), max(max(y_vals), old_limits[1])
    limits_mid_dif = (limits[1] + limits[0]) / 2, (limits[1] - limits[0]) / 2
    axes.set_ylim(limits_mid_dif[0] - limits_mid_dif[1] * 1.1, limits_mid_dif[0] + limits_mid_dif[1] * 1.1)
    return limits


def run_animation(medium: Medium, fps: float, b_draw_u=True, b_draw_v=False, pause_at_start=True, f_edit_plot=None, on_animation_update=None,
                  initial_limits_u=(float("inf"), -1 * float("inf")),
                  initial_limits_v=(float("inf"), -1 * float("inf")),
                  animation_length=float("inf")):
    num_frames = int(animation_length * fps) if np.isfinite(animation_length) else None

    update_animation_iter = AnimationUpdater(medium, b_draw_u, b_draw_v, pause_at_start, f_edit_plot, on_animation_update, initial_limits_u, initial_limits_v)

    anim = FuncAnimation(fig=update_animation_iter.fig, func=update_animation_iter, frames=num_frames, interval=1000 / fps)
    return anim


class AnimationUpdater:
    def __init__(self, medium, b_draw_u, b_draw_v, pause_at_start, f_edit_plot, on_animation_update, initial_limits_u, initial_limits_v):
        self.medium = medium
        self.b_draw_u = b_draw_u
        self.b_draw_v = b_draw_v
        self.pause_at_start = pause_at_start
        self.f_edit_plot = f_edit_plot
        self.on_animation_update = on_animation_update
        self.limits_u = initial_limits_u
        self.limits_v = initial_limits_v
        self.target_simulation_time = -1 if pause_at_start else 0
        self.tic_toc = TicToc()
        self.tic_toc.tic()

        self.lineu, self.axu, self.linev, self.axv, self.fig = medium.plot(b_draw_u, b_draw_v)
        self.objects_in_plot = []
        if f_edit_plot:
            items_to_plot = f_edit_plot(self.lineu, self.axu, self.linev, self.axv, self.fig)
            if items_to_plot is not None:
                self.objects_in_plot += items_to_plot
        if b_draw_u:
            self.objects_in_plot.append(self.lineu)
        if b_draw_v:
            self.objects_in_plot.append(self.linev)

    def __call__(self, *args, **kwargs):
        self.target_simulation_time += self.tic_toc.tocvalue()
        if self.target_simulation_time > 0:
            self.medium.advance_to_time(self.target_simulation_time)
        self.tic_toc.tic()
        if self.b_draw_u:
            self.lineu.set_data(self.medium.x, self.medium.u)
            self.limits_u = reset_yaxis_limits(self.axu, self.medium.u, self.limits_u)
        if self.b_draw_v:
            self.linev.set_data(self.medium.x, self.medium.v)
            self.limits_v = reset_yaxis_limits(self.axv, self.medium.v, self.limits_v)
        if self.on_animation_update is not None:
            self.on_animation_update()
        return self.objects_in_plot
