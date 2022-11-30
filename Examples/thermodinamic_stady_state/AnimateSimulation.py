# __package__ = "Examples.thermodinamic_stady_state.animate.py"

import sys

import matplotlib.pyplot as plt
import matplotlib.patches

# allows imports relative to project's base
base_name = "Py_Wave_Simulation"
sys.path.insert(0, __file__[:__file__.index(base_name) + len(base_name)])


from Medium import Medium
from Boundary_Conditions import hard_boundary_conditions_creator, limited_segment_condition_creator, open_boundary_conditions_creator, \
    flow_out_boundary_conditions_creator, BoundaryCondition, LimmitedY_SegmentCondition
from animation_module import run_animation
from unittest import TestCase
import numpy as np


obstacle_gen = limited_segment_condition_creator(-0.2, 0.2, 0.19)  # obstacle that limits u movement in the indexes 100-200
boundary_conditions = open_boundary_conditions_creator()
m = Medium(v=1,c=1, boundary_conditions_generators=boundary_conditions + obstacle_gen)

obstacle = [b for b in m.boundary_conditions if type(b) is LimmitedY_SegmentCondition][0]

side_fig:plt.Figure = plt.figure()
pos_ax,internal_energy_ax = side_fig.subplots(2,1)
com_energy_ax = internal_energy_ax.twinx()
pos_ax.set_ylim(-1,1)
com_energy_ax.set_ylim(0,6)
internal_energy_ax.set_ylim(0, 6)

com_vs_time_graph = pos_ax.plot([0],[0],label="position of center of mass")[0]
rightedge_vs_time_graph = pos_ax.plot([0],[0],label="position of right edge")[0]
boundedseg_vs_time_graph = pos_ax.plot([0],[0],label="position of bounded segment")[0]
total_energy_vs_time_graph = internal_energy_ax.plot([0], [0], label="total energy")[0]
internal_energy_vs_time_graph = internal_energy_ax.plot([0], [0], label="internal energy")[0]
com_velocity_energy_vs_time_graph = com_energy_ax.plot([0], [0], "g",label="energy of velocity of center of mass")[0]

stat_com_arr = [0]
stat_rightedge_arr = [0]
stat_boundedseg_arr = [0]
stat_total_energy_arr = [1]
stat_internal_energy_arr = [1]
stat_com_velocity_energy_arr = [1]

stat_time_arr = [m.time]

com_dot = None
def update_anim2(frame):
    com_vs_time_graph.set_data(stat_time_arr, stat_com_arr)
    rightedge_vs_time_graph.set_data(stat_time_arr, stat_rightedge_arr)
    boundedseg_vs_time_graph.set_data(stat_time_arr, stat_boundedseg_arr)
    total_energy_vs_time_graph.set_data(stat_time_arr, stat_total_energy_arr)
    internal_energy_vs_time_graph.set_data(stat_time_arr, stat_internal_energy_arr)
    com_velocity_energy_vs_time_graph.set_data(stat_time_arr, stat_com_velocity_energy_arr)

    pos_ax.set_xlim(-0.05*stat_time_arr[-1], stat_time_arr[-1])
    com_energy_ax.set_xlim(-0.05 * stat_time_arr[-1], stat_time_arr[-1])
    internal_energy_ax.set_xlim(-0.05 * stat_time_arr[-1], stat_time_arr[-1])
    len_time = len(stat_time_arr)
    com_energy_ax.set_ylim(-max(stat_com_velocity_energy_arr[len_time//2:])*0.01,max(stat_com_velocity_energy_arr[len_time//2:])*1.01)
    internal_energy_ax.set_ylim(min(stat_internal_energy_arr[len_time//2:])*0.99, max(stat_internal_energy_arr[len_time//2:])*1.01)

    return [com_vs_time_graph,rightedge_vs_time_graph,boundedseg_vs_time_graph,
            total_energy_vs_time_graph,internal_energy_vs_time_graph,com_velocity_energy_vs_time_graph]
from matplotlib.animation import FuncAnimation
anim2 = FuncAnimation(side_fig, update_anim2, interval=1000 / 20)

def f_edit_plot(lineu, axu, linev, axv, fig):
    global com_dot
    fig: plt.Figure
    ax: plt.Axes = fig.gca()
    com_dot = ax.plot([0,1],[stat_com_arr[-1],stat_com_arr[-1]],"g",ls="--")[0]
    return [com_dot]

def on_animation_update():
    stat_time_arr.append(m.time)
    stat_com_arr.append(np.average(m.u))
    stat_rightedge_arr.append(m.u[-1])
    stat_boundedseg_arr.append(m.u[obstacle.lagrangian_slice.start])
    stat_total_energy_arr.append(float(m.energy_Tot))
    com_vel = (stat_com_arr[-1] - stat_com_arr[-2]) / (stat_time_arr[-1] - stat_time_arr[-2])
    stat_com_velocity_energy_arr.append(np.sum((0.5 * m.z / m.c) * com_vel ** 2*np.gradient(m.x)) if np.isfinite(com_vel) else 1)
    stat_internal_energy_arr.append(stat_total_energy_arr[-1] - stat_com_velocity_energy_arr[-1])

    com_dot.set_data([0,1],[stat_com_arr[-1],stat_com_arr[-1]])

a = run_animation(m, 20,f_edit_plot=f_edit_plot,on_animation_update=on_animation_update)
plt.show()