from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys

# allows imports relative to project's base
base_name = "Py_Wave_Simulation"
sys.path.insert(0, __file__[:__file__.index(base_name) + len(base_name)])

from SimulationCheckpoint import SimulationResultsCheckPoint
import numpy as np

class AnalysisParameters:
    # num_cells - the number of cells to be used #L#
    num_cells = None
    # RUN_NAME - the name for the simulation run. this is used for checkpoints.
    RUN_NAME = None
    # target_time - the simulation time to which the simulation will be advanced.
    target_time = None
    # index_range_of_steady_state - the index range at which steady state is reached
    index_range_of_steady_state = None
    # b_plot determines whether or not will plot
    b_plot = None

    def __init__(self,run_parameters):
        self.num_cells = run_parameters.num_cells
        self.RUN_NAME = run_parameters.RUN_NAME
        self.target_time = run_parameters.target_time
        self.index_range_of_steady_state = None

    def unload(self):
        return self.num_cells, self.RUN_NAME, self.target_time, self.index_range_of_steady_state, self.b_plot

def analyseResults(analysis_parameters:AnalysisParameters):
    num_cells, RUN_NAME, target_time, index_range_of_steady_state, b_plot = analysis_parameters.unload()

    if target_time == "last" or target_time == None:
        target_time = SimulationResultsCheckPoint.b_find_less_advanced_cp(RUN_NAME,float("inf"))
    assert SimulationResultsCheckPoint.b_is_cp_exists(RUN_NAME,progress=target_time)
    checkpoint = SimulationResultsCheckPoint.load(RUN_NAME,progress=target_time)
    system_state_at_end, stat_time_arr, stat_com_arr, stat_rightedge_arr, stat_boundedseg_arr, stat_total_energy_arr = checkpoint.unload()

    # -- Get Indexes Of Where The String Is In Contact With The Bounds
    bound_pts_low = np.where((stat_boundedseg_arr == -0.2))[0]
    bound_pts_high = np.where((stat_boundedseg_arr == 0.2))[0]

    # -- Filter Out Indexes Where The Same Bound Is Hit Twice In A Row
    # we are looking for the indexes in which the string hits a bound that is different then the last.
    midranges = []
    midranges_is_low_to_high = []
    boundranges = []
    boundranges_is_low = []

    low_bounds_iter = iter(bound_pts_low)
    idx_low = next(low_bounds_iter)
    high_bounds_iter = iter(bound_pts_high)
    idx_high = next(high_bounds_iter)
    cur_boundrange_min = min(idx_low, idx_high)
    try:
        while True:
            # switch_just_happend is made of: (is_low_to_high,old_idx,mid_idx,new_idx)
            switch_just_happend = None
            if idx_low < idx_high:
                new_idx_low = next(low_bounds_iter)
                if new_idx_low > idx_high:
                    switch_just_happend = True, idx_low, idx_high, new_idx_low
                idx_low = new_idx_low
            else:
                new_idx_high = next(high_bounds_iter)
                if new_idx_high > idx_low:
                    switch_just_happend = False, idx_high, idx_low, new_idx_high
                idx_high = new_idx_high

            if switch_just_happend is not None:
                is_low_to_high, old_idx, mid_idx, new_idx = switch_just_happend
                # calculate new midrange
                new_midrange = (old_idx, mid_idx)
                # calculate new boundrange
                new_boundrange = (cur_boundrange_min, old_idx)
                # maintain cur_boundrange_min
                cur_boundrange_min = mid_idx

                midranges.append(new_midrange)
                boundranges.append(new_boundrange)

                # calc
                if is_low_to_high:
                    # switch is from low to high
                    midranges_is_low_to_high.append(True)
                    boundranges_is_low.append(True)
                else:
                    # switch is from high to low
                    midranges_is_low_to_high.append(False)
                    boundranges_is_low.append(False)
    except StopIteration as e:
        pass

    # get the value of a statistic array only at when the string is in contact with the bounds
    def get_at_bounds(arr):
        return np.array([np.average(arr[l:h]) for (l, h) in boundranges])

    # get the value of a statistic array only at the midpoint between the two bounds
    def get_at_midranges(arr):
        return np.array([np.average(arr[l:h]) for (l, h) in midranges])

    # -- Comparison Histograms Of Extrema Of Different Points On The String
    def num_bins_for_hist(points):
        return int(len(points) ** 0.5 * 5)

    # if b_plot:
    #     plt.figure()
    #     com_v = get_at_bounds(stat_com_arr)
    #     right_v = get_at_bounds(stat_rightedge_arr)
    #     bound_V = get_at_bounds(stat_boundedseg_arr)
    #     minv = min(np.min(com_v), np.min(right_v), np.min(bound_V))
    #     maxv = max(np.max(com_v), np.max(right_v), np.max(bound_V))
    #     plt.hist(com_v, bins=np.linspace(minv, maxv, num_bins_for_hist(com_v)), density=True, color="g", alpha=0.5, label="com")
    #     plt.hist(right_v, bins=np.linspace(minv, maxv, num_bins_for_hist(right_v)), density=True, color="r", alpha=0.5, label="right")
    #     plt.hist(bound_V, bins=np.linspace(minv, maxv, num_bins_for_hist(bound_V)), density=True, color="b", alpha=0.5, label="left")
    #     plt.grid(ls="--")
    #     plt.legend()
    #     plt.title(f"left, com, and right at extrema\nnum_cells={num_cells}")
    #
    # # -- Comparison Histograms Of Abs Of Extrema Of Different Points On The String
    # if b_plot:
    #     plt.figure()
    #     com_v = np.abs(get_at_bounds(stat_com_arr))
    #     right_v = np.abs(get_at_bounds(stat_rightedge_arr))
    #     minv = min(np.min(com_v), np.min(right_v))
    #     maxv = max(np.max(com_v), np.max(right_v))
    #     plt.hist(com_v, bins=np.linspace(minv, maxv, num_bins_for_hist(com_v)), density=True, color="g", alpha=0.5, label="com")
    #     plt.hist(right_v, bins=np.linspace(minv, maxv, num_bins_for_hist(right_v)), density=True, color="r", alpha=0.5, label="right")
    #     ylim = plt.ylim()
    #     plt.vlines([np.average(com_v), np.average(right_v)], -10, 10, ["g", "r"], label="averages")
    #     plt.ylim(*ylim)
    #     plt.grid(ls="--")
    #     plt.legend()
    #     plt.title(f"com vs right distance from center at bound collision\nnum_cells={num_cells}")

    # -- Compactify Raw Stats To Be Plotable
    stat_indexes_compact = range(0,len(stat_total_energy_arr),len(stat_total_energy_arr)//(100*len(midranges)))
    stat_total_energy_arr_compact = stat_total_energy_arr[stat_indexes_compact]
    stat_time_arr_compact = stat_time_arr[stat_indexes_compact]
    indexes_of_midranges = get_at_midranges(range(len(stat_total_energy_arr)))
    midrange_index_arr_compact = np.interp(stat_indexes_compact,indexes_of_midranges,range(len(midranges)))

    # -- Energy Ratio Analysis
    com_vel = np.gradient(stat_com_arr) / np.gradient(stat_time_arr)
    stat_com_velocity_energy_arr = ((0.5 * system_state_at_end.z[0] / system_state_at_end.c[0]) * com_vel ** 2)
    energy_ratio = get_at_midranges(stat_com_velocity_energy_arr) / get_at_midranges(stat_total_energy_arr)

    # -- Analysis Of Steady State:
    if index_range_of_steady_state[1] is None:
        index_range_of_steady_state = (index_range_of_steady_state[0], len(energy_ratio))
    steady_state_idx = range(index_range_of_steady_state[0], index_range_of_steady_state[1])
    steady_state_energy_ratio_population = energy_ratio[steady_state_idx]
    energy_ratio_mean = np.mean(steady_state_energy_ratio_population)
    energy_ratio_std_of_mean = np.std(steady_state_energy_ratio_population, ddof=1) / np.sqrt(len(steady_state_energy_ratio_population))
    energy_ratio_theory = 1 / num_cells

    # -- Draw Full Energy Ratio Figure
    if b_plot:
        plt.figure()
        plt.title(f"kinetic energy of center of mass VS total internal energy\nnum_cells={num_cells}")
        plt.plot(energy_ratio, "o", label="energy state between collisions")
        plt.xlabel("collision index")
        plt.ylabel("energy ratio")
        plt.yscale("log")
        # highlight steady state
        steady_state_min_y = min(steady_state_energy_ratio_population) * 0.8
        steady_state_max_y = max(steady_state_energy_ratio_population) * 1.2
        x_line = [min(steady_state_idx), max(steady_state_idx) + 1]
        plt.gca().add_patch(
            patches.Rectangle((x_line[0], steady_state_min_y), x_line[1]-x_line[0], steady_state_max_y - steady_state_min_y,
                              linewidth=1, edgecolor=(0.1, 0.9, 0.1, 0.5), ls="--", facecolor=(0.1, 0.9, 0.1, 0.3), label="steady state"))
        # highlight simulation death
        death_min_y = min(energy_ratio[max(steady_state_idx):]) * 0.8
        death_max_y = max(energy_ratio[max(steady_state_idx):]) * 1.2
        temp = plt.xlim()
        plt.gca().add_patch(
            patches.Rectangle((max(steady_state_idx), death_min_y), plt.xlim()[1] - max(steady_state_idx), death_max_y - death_min_y,
                              linewidth=1, edgecolor=(0.9, 0.1, 0.1, 0.5), ls="--", facecolor=(0.9, 0.1, 0.1, 0.3), label="simulation death"))
        plt.xlim(*temp)
        # draw steady state analysis results
        plt.plot(x_line, [energy_ratio_mean] * 2, "r", label="mean energy ratio")
        plt.plot(x_line, [energy_ratio_mean + energy_ratio_std_of_mean] * 2, "--r", label="68% confidence interval")
        plt.plot(x_line, [energy_ratio_mean - energy_ratio_std_of_mean] * 2, "--r")
        plt.plot(x_line, [energy_ratio_theory] * 2, color=(0.5, 0.8, 0.5), label="theoretical energy ratio")
        plt.legend()
        # draw total energy on twinx
        plt.twinx()
        plt.plot(midrange_index_arr_compact,stat_total_energy_arr_compact,"-r",label="total system energy")
        plt.yscale("log")
        plt.legend()
        plt.gcf().savefig(f"research_results\\Full Energy Ratio num_cells={num_cells}")

    # -- Draw Full Energy Ratio Figure With X-axis Equals Time
    if b_plot:
        T = get_at_midranges(stat_time_arr)
        plt.figure()
        plt.title(f"kinetic energy of center of mass VS total internal energy\nnum_cells={num_cells}")
        plt.plot(T, energy_ratio, "o", label="energy state between collisions")
        plt.xlabel("time")
        plt.ylabel("energy ratio")
        plt.yscale("log")
        # highlight steady state
        steady_state_min_y = min(steady_state_energy_ratio_population) * 0.8
        steady_state_max_y = max(steady_state_energy_ratio_population) * 1.2
        x_line = [T[min(steady_state_idx)], T[max(steady_state_idx)] + 1]
        plt.gca().add_patch(
            patches.Rectangle((x_line[0], steady_state_min_y), x_line[1]-x_line[0], steady_state_max_y - steady_state_min_y,
                              linewidth=1, edgecolor=(0.1, 0.9, 0.1, 0.5), ls="--", facecolor=(0.1, 0.9, 0.1, 0.3), label="steady state"))
        # highlight simulation death
        death_min_y = min(energy_ratio[max(steady_state_idx):]) * 0.8
        death_max_y = max(energy_ratio[max(steady_state_idx):]) * 1.2
        temp = plt.xlim()
        plt.gca().add_patch(
            patches.Rectangle((x_line[1], death_min_y), plt.xlim()[1] - x_line[1], death_max_y - death_min_y,
                              linewidth=1, edgecolor=(0.9, 0.1, 0.1, 0.5), ls="--", facecolor=(0.9, 0.1, 0.1, 0.3), label="simulation death"))
        plt.xlim(*temp)
        # draw steady state analysis results
        x_line = [T[min(steady_state_idx)], T[max(steady_state_idx)] + 1]
        plt.plot(x_line, [energy_ratio_mean] * 2, "r", label="mean energy ratio")
        plt.plot(x_line, [energy_ratio_mean + energy_ratio_std_of_mean] * 2, "--r", label="68% confidence interval")
        plt.plot(x_line, [energy_ratio_mean - energy_ratio_std_of_mean] * 2, "--r")
        plt.plot(x_line, [energy_ratio_theory] * 2, color=(0.5, 0.8, 0.5), label="theoretical energy ratio")
        plt.legend()
        # draw total energy on twinx
        plt.twinx()
        plt.plot(stat_time_arr_compact, stat_total_energy_arr_compact,"-r",label="total system energy")
        plt.yscale("log")
        plt.legend()
        plt.gcf().savefig(f"research_results\\Full Energy Ratio With X-axis Equals Time num_cells={num_cells}")

    # -- Draw Energy Ratio Figure Zoomed In On Steady State
    if b_plot:
        plt.figure()
        plt.title(f"kinetic energy of center of mass VS total internal energy At steady state\nnum_cells={num_cells}")
        plt.plot(steady_state_idx, steady_state_energy_ratio_population, "o", label="energy ratio population at steady state")
        x_line = [min(steady_state_idx), max(steady_state_idx) + 1]
        plt.plot(x_line, [energy_ratio_mean] * 2, "r", label="mean energy ratio")
        plt.plot(x_line, [energy_ratio_mean + energy_ratio_std_of_mean] * 2, "--r", label="68% confidence interval")
        plt.plot(x_line, [energy_ratio_mean - energy_ratio_std_of_mean] * 2, "--r")
        plt.plot(x_line, [energy_ratio_theory] * 2, color=(0.5, 0.8, 0.5), label="theoretical energy ratio")
        plt.xlabel("collision index")
        plt.ylabel("energy ratio")
        plt.legend()
        plt.gcf().savefig(f"research_results\\Full Energy Ratio Zoomed In On Steady State num_cells={num_cells}")

    return namedtuple("AnalysisResult",
                      "system_state_at_end stat_time_arr stat_com_arr stat_rightedge_arr stat_boundedseg_arr stat_total_energy_arr midranges boundranges stat_indexes_compact stat_total_energy_arr_compact stat_time_arr_compact indexes_of_midranges midrange_index_arr_compact com_vel stat_com_velocity_energy_arr energy_ratio steady_state_idx steady_state_energy_ratio_population energy_ratio_mean energy_ratio_std_of_mean energy_ratio_theory")(
    system_state_at_end ,stat_time_arr ,stat_com_arr ,stat_rightedge_arr ,stat_boundedseg_arr ,stat_total_energy_arr ,midranges ,boundranges ,stat_indexes_compact ,stat_total_energy_arr_compact ,stat_time_arr_compact ,indexes_of_midranges ,midrange_index_arr_compact ,com_vel ,stat_com_velocity_energy_arr ,energy_ratio ,steady_state_idx ,steady_state_energy_ratio_population ,energy_ratio_mean ,energy_ratio_std_of_mean ,energy_ratio_theory)