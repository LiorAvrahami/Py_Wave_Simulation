import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches

import sys

# allows imports relative to project's base
base_name = "Py_Wave_Simulation"
sys.path.insert(0, __file__[:__file__.index(base_name) + len(base_name)])

import datetime
from SimulationCheckpoint import SimulationResultsCheckPoint
from Medium import Medium
from Boundary_Conditions import hard_boundary_conditions_creator, limited_segment_condition_creator, open_boundary_conditions_creator, \
    flow_out_boundary_conditions_creator, BoundaryCondition, LimmitedY_SegmentCondition
import numpy as np

class RunParameters:
    # num_cells - the number of cells to be used #L#
    num_cells = None
    # RUN_NAME - the name for the simulation run. this is used for checkpoints.
    RUN_NAME = None
    # target_time - the simulation time to which the simulation will be advanced.
    target_time = None
    # b_force_start_clean - whether or not to check if an old run's checkpoint exists,
    # if one exists and "b_force_start_clean" is false then the old checkpoint will be used.
    b_force_start_clean = None
    # the amount of simulation steps between prints to terminal
    frequency_of_print_eta = None
    # the amount of simulation steps between creation of checkpoints
    frequency_of_save_checkpoint = None

    def assert_full(self):
        assert self.num_cells is not None
        assert self.RUN_NAME is not None
        assert self.target_time is not None
        assert self.b_force_start_clean is not None
        assert self.frequency_of_print_eta is not None
        assert self.frequency_of_save_checkpoint is not None

    def unload(self):
        return self.num_cells, self.RUN_NAME, self.target_time, self.b_force_start_clean, self.frequency_of_print_eta, self.frequency_of_save_checkpoint

def runSimulationForAnalysis(run_parameters:RunParameters):
    num_cells, RUN_NAME, target_time, b_force_start_clean, frequency_of_print_eta, frequency_of_save_checkpoint = run_parameters.unload()
    def cycler(freq):
        i = 1
        while True:
            yield i % freq == 0
            i+=1

    print_eta_cycler = cycler(frequency_of_print_eta)
    checkpoint_cycler = cycler(frequency_of_save_checkpoint)

    # -- Get Simulation Results Data.
    # if simulation was already run and was saved, and if "b_force_start_clean" is false then results will be loaded.
    # otherwise simulation will be run from scratch.
    # after loading or creation of new run, the simulation is advanced up to "target_time", (unless simulation is already at "target_time").
    checkpoint:SimulationResultsCheckPoint
    temp_old_removable_checkpoint:SimulationResultsCheckPoint = None
    start_cp_progress = SimulationResultsCheckPoint.b_find_less_advanced_cp(RUN_NAME, progress=target_time)
    if start_cp_progress is None or b_force_start_clean:
        start_cp_progress = 0
        # make new run
        obstacle_gen = limited_segment_condition_creator(-0.2, 0.2, 0.19)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(x=num_cells,v=1, c=1, boundary_conditions_generators=boundary_conditions + obstacle_gen)

        stat_com_arr = []
        stat_rightedge_arr = []
        stat_boundedseg_arr = []
        stat_total_energy_arr = []

        stat_time_arr = []
    else:
        # load old run and advance
        checkpoint = SimulationResultsCheckPoint.load(RUN_NAME,progress=start_cp_progress)
        m, stat_time_arr, stat_com_arr, stat_rightedge_arr, stat_boundedseg_arr, stat_total_energy_arr = checkpoint.unload()
        stat_time_arr= list(stat_time_arr)
        stat_com_arr = list(stat_com_arr)
        stat_rightedge_arr = list(stat_rightedge_arr)
        stat_boundedseg_arr = list(stat_boundedseg_arr)
        stat_total_energy_arr = list(stat_total_energy_arr)

    obstacle = [b for b in m.boundary_conditions if type(b) is LimmitedY_SegmentCondition][0]

    # for eta
    start_time_irl = datetime.datetime.now()
    initial_simulation_time = start_cp_progress

    while m.time < target_time:
        m.step()
        stat_time_arr.append(m.time)
        stat_com_arr.append(np.average(m.u))
        stat_rightedge_arr.append(m.u[-1])
        stat_boundedseg_arr.append(m.u[obstacle.lagrangian_slice.start])
        stat_total_energy_arr.append(float(m.energy_Tot))
        if next(print_eta_cycler):
            time_left = (target_time - m.time) * (datetime.datetime.now() - start_time_irl).seconds/(m.time - initial_simulation_time)
            if np.isfinite(time_left):
                eta = datetime.datetime.now() + datetime.timedelta(seconds=time_left)
                eta_str = eta.strftime("%d/%m/%Y %H:%M:%S") if eta.date() != datetime.datetime.now().date() else eta.strftime("%H:%M:%S")
                print(m.time,f" | eta = {eta_str}")
        if next(checkpoint_cycler):
            checkpoint = SimulationResultsCheckPoint(RUN_NAME, m.time,m,np.array(stat_time_arr), np.array(stat_com_arr), np.array(stat_rightedge_arr),
                                                     np.array(stat_boundedseg_arr), np.array(stat_total_energy_arr))
            checkpoint.store()
            print(m.time, " -- CHECKPOINT -- ")
            if temp_old_removable_checkpoint is not None:
                temp_old_removable_checkpoint.delete_self_from_disk()
            temp_old_removable_checkpoint = checkpoint

    checkpoint = SimulationResultsCheckPoint(RUN_NAME, m.time,m,np.array(stat_time_arr), np.array(stat_com_arr), np.array(stat_rightedge_arr),
                                             np.array(stat_boundedseg_arr), np.array(stat_total_energy_arr))
    checkpoint.store()
#
# system_state_at_end, stat_time_arr, stat_com_arr, stat_rightedge_arr, stat_boundedseg_arr, stat_total_energy_arr = checkpoint.unload()
#
# # -- Get Indexes Of Where The String Is In Contact With The Bounds
# bound_pts_low = np.where((stat_boundedseg_arr == -0.2))[0]
# bound_pts_high = np.where((stat_boundedseg_arr == 0.2))[0]
#
# # -- Filter Out Indexes Where The Same Bound Is Hit Twice In A Row
# # we are looking for the indexes in which the string hits a bound that is different then the last.
# midranges = []
# midranges_is_low_to_high = []
# boundranges = []
# boundranges_is_low = []
#
# low_bounds_iter = iter(bound_pts_low)
# idx_low = next(low_bounds_iter)
# high_bounds_iter = iter(bound_pts_high)
# idx_high = next(high_bounds_iter)
# cur_boundrange_min = min(idx_low, idx_high)
# try:
#     while True:
#         # switch_just_happend is made of: (is_low_to_high,old_idx,mid_idx,new_idx)
#         switch_just_happend = None
#         if idx_low < idx_high:
#             new_idx_low = next(low_bounds_iter)
#             if new_idx_low > idx_high:
#                 switch_just_happend = True, idx_low, idx_high, new_idx_low
#             idx_low = new_idx_low
#         else:
#             new_idx_high = next(high_bounds_iter)
#             if new_idx_high > idx_low:
#                 switch_just_happend = False, idx_high, idx_low, new_idx_high
#             idx_high = new_idx_high
#
#         if switch_just_happend is not None:
#             is_low_to_high, old_idx, mid_idx, new_idx = switch_just_happend
#             # calculate new midrange
#             new_midrange = (old_idx, mid_idx)
#             # calculate new boundrange
#             new_boundrange = (cur_boundrange_min, old_idx)
#             # maintain cur_boundrange_min
#             cur_boundrange_min = mid_idx
#
#             midranges.append(new_midrange)
#             boundranges.append(new_boundrange)
#
#             # calc
#             if is_low_to_high:
#                 # switch is from low to high
#                 midranges_is_low_to_high.append(True)
#                 boundranges_is_low.append(True)
#             else:
#                 # switch is from high to low
#                 midranges_is_low_to_high.append(False)
#                 boundranges_is_low.append(False)
# except StopIteration as e:
#     pass
#
# # get the value of a statistic array only at when the string is in contact with the bounds
# def get_at_bounds(arr):
#     return np.array([np.average(arr[l:h]) for (l, h) in boundranges])
#
# # get the value of a statistic array only at the midpoint between the two bounds
# def get_at_midranges(arr):
#     return np.array([np.average(arr[l:h]) for (l, h) in midranges])
#
#
# # -- Comparison Histograms Of Extrema Of Different Points On The String
# def num_bins_for_hist(points):
#     return int(len(points) ** 0.5 * 5)
# plt.figure()
# com_v = get_at_bounds(stat_com_arr)
# right_v = get_at_bounds(stat_rightedge_arr)
# bound_V = get_at_bounds(stat_boundedseg_arr)
# minv = min(np.min(com_v), np.min(right_v), np.min(bound_V))
# maxv = max(np.max(com_v), np.max(right_v), np.max(bound_V))
# plt.hist(com_v, bins=np.linspace(minv, maxv, num_bins_for_hist(com_v)), density=True, color="g", alpha=0.5, label="com")
# plt.hist(right_v, bins=np.linspace(minv, maxv, num_bins_for_hist(right_v)), density=True, color="r", alpha=0.5, label="right")
# plt.hist(bound_V, bins=np.linspace(minv, maxv, num_bins_for_hist(bound_V)), density=True, color="b", alpha=0.5, label="left")
# plt.grid(ls="--")
# plt.legend()
# plt.title(f"left, com, and right at extrema\nnum_cells={num_cells}")
#
# # -- Comparison Histograms Of Abs Of Extrema Of Different Points On The String
# plt.figure()
# com_v = np.abs(get_at_bounds(stat_com_arr))
# right_v = np.abs(get_at_bounds(stat_rightedge_arr))
# minv = min(np.min(com_v), np.min(right_v))
# maxv = max(np.max(com_v), np.max(right_v))
# plt.hist(com_v, bins=np.linspace(minv, maxv, num_bins_for_hist(com_v)), density=True, color="g", alpha=0.5, label="com")
# plt.hist(right_v, bins=np.linspace(minv, maxv, num_bins_for_hist(right_v)), density=True, color="r", alpha=0.5, label="right")
# ylim = plt.ylim()
# plt.vlines([np.average(com_v), np.average(right_v)], -10, 10, ["g", "r"], label="averages")
# plt.ylim(*ylim)
# plt.grid(ls="--")
# plt.legend()
# plt.title(f"com vs right distance from center at bound collision\nnum_cells={num_cells}")
#
# # -- Energy Ratio Analysis
# com_vel = np.gradient(stat_com_arr) / np.gradient(stat_time_arr)
# stat_com_velocity_energy_arr = ((0.5 * system_state_at_end.z[0] / system_state_at_end.c[0]) * com_vel ** 2)
# energy_ratio = get_at_midranges(stat_com_velocity_energy_arr) / get_at_midranges(stat_total_energy_arr)
#
# # -- Analysis Of Steady State:
# steady_state_idx = range(205,len(energy_ratio))
# steady_state_energy_ratio_population = energy_ratio[steady_state_idx]
# energy_ratio_mean = np.mean(steady_state_energy_ratio_population)
# energy_ratio_std_of_mean = np.std(steady_state_energy_ratio_population,ddof=1)/np.sqrt(len(steady_state_energy_ratio_population))
# energy_ratio_theory = 1/500
#
# # -- Draw Full Energy Ratio Figure
# plt.figure()
# plt.title(f"kinetic energy of center of mass VS total internal energy\nnum_cells={num_cells}")
# plt.plot(energy_ratio,"o",label="energy state between collisions")
# plt.xlabel("collision index")
# plt.ylabel("energy ratio")
# plt.yscale("log")
# # highlight steady state
# steady_state_min_y = min(steady_state_energy_ratio_population)*0.8
# steady_state_max_y = max(steady_state_energy_ratio_population)*1.2
# temp = plt.xlim()
# plt.gca().add_patch(
#     patches.Rectangle((min(steady_state_idx), steady_state_min_y), plt.xlim()[1]-min(steady_state_idx), steady_state_max_y-steady_state_min_y,
#                       linewidth=1, edgecolor=(0.1,0.9,0.1,0.5),ls="--", facecolor=(0.1,0.9,0.1,0.3),label="steady state"))
#
#
# # -- Draw Full Energy Ratio Figure With X-axis Equals Time
# T = get_at_midranges(stat_time_arr)
# plt.figure()
# plt.title(f"kinetic energy of center of mass VS total internal energy\nnum_cells={num_cells}")
# plt.plot(T,energy_ratio,"o",label="energy state between collisions")
# plt.xlabel("time")
# plt.ylabel("energy ratio")
# plt.yscale("log")
# # highlight steady state
# steady_state_min_y = min(steady_state_energy_ratio_population)*0.8
# steady_state_max_y = max(steady_state_energy_ratio_population)*1.2
# temp = plt.xlim()
# plt.gca().add_patch(
#     patches.Rectangle((T[min(steady_state_idx)], steady_state_min_y), plt.xlim()[1]-min(steady_state_idx), steady_state_max_y-steady_state_min_y,
#                       linewidth=1, edgecolor=(0.1,0.9,0.1,0.5),ls="--", facecolor=(0.1,0.9,0.1,0.3),label="steady state"))
# plt.xlim(*temp)
# # draw steady state analysis results
# x_line = [T[min(steady_state_idx)],T[max(steady_state_idx)]+1]
# plt.plot(x_line,[energy_ratio_mean]*2,"r",label="mean energy ratio")
# plt.plot(x_line,[energy_ratio_mean + energy_ratio_std_of_mean]*2,"--r",label="68% confidence interval")
# plt.plot(x_line,[energy_ratio_mean - energy_ratio_std_of_mean]*2,"--r")
# plt.plot(x_line,[energy_ratio_theory]*2,color=(0.5,0.8,0.5),label="theoretical energy ratio")
#
# plt.legend()
#
# # -- Draw Energy Ratio Figure Zoomed In On Steady State
# plt.figure()
# plt.title(f"kinetic energy of center of mass VS total internal energy At steady state\nnum_cells={num_cells}")
# plt.plot(steady_state_idx,steady_state_energy_ratio_population,"o",label="energy ratio population at steady state")
# x_line = [min(steady_state_idx),max(steady_state_idx) + 1]
# plt.plot(x_line,[energy_ratio_mean]*2,"r",label="mean energy ratio")
# plt.plot(x_line,[energy_ratio_mean + energy_ratio_std_of_mean]*2,"--r",label="68% confidence interval")
# plt.plot(x_line,[energy_ratio_mean - energy_ratio_std_of_mean]*2,"--r")
# plt.plot(x_line,[energy_ratio_theory]*2,color=(0.5,0.8,0.5),label="theoretical energy ratio")
# plt.xlabel("collision index")
# plt.ylabel("energy ratio")
# plt.legend()
# plt.show()