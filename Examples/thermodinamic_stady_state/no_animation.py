import matplotlib.pyplot as plt
import matplotlib.patches

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
stat_total_energy_arr = [0]
stat_internal_energy_arr = [0]
stat_com_velocity_energy_arr = [0]

stat_time_arr = [m.time]

#todo generate histogram of com_energy/total_energy after boundery collition at steady state, also try and preserve energy better.

while True:
    m.step()
