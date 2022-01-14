import random

import matplotlib.pyplot as plt
import matplotlib.patches

import sys

# allows imports relative to project's base
base_name = "Py_Wave_Simulation"
sys.path.insert(0, __file__[:__file__.index(base_name) + len(base_name)])

from Medium import Medium
from Boundary_Conditions import hard_boundary_conditions_creator, limited_segment_condition_creator, open_boundary_conditions_creator, \
    flow_out_boundary_conditions_creator, BoundaryCondition, LimmitedY_SegmentCondition
from animation_module import run_animation
from unittest import TestCase
import numpy as np
import os,pickle

class SimulationResultsCheckPoint:
    stat_time_arr: np.array
    stat_com_arr: np.array
    stat_rightedge_arr: np.array
    stat_boundedseg_arr: np.array
    stat_total_energy_arr: np.array
    system_state_at_end = None

    cp_dir_name = "CheckpointFiles"

    def __init__(self,stat_time_arr, stat_com_arr, stat_rightedge_arr, stat_boundedseg_arr,stat_total_energy_arr,system_state_at_end):
        self.stat_time_arr = stat_time_arr
        self.stat_com_arr = stat_com_arr
        self.stat_rightedge_arr = stat_rightedge_arr
        self.stat_boundedseg_arr = stat_boundedseg_arr
        self.stat_total_energy_arr = stat_total_energy_arr
        self.system_state_at_end = system_state_at_end

    def store(self,name_base,progress=None):
        file_name = SimulationResultsCheckPoint.get_file_name(name_base,progress)
        if os.path.exists(file_name):
            new_name = self.find_vacent_name(name_base,progress)
            # move old checkpoint file to new place.
            os.rename(file_name, new_name)
        with open(file_name,"wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(name_base,progress=None):
        file_name = SimulationResultsCheckPoint.get_file_name(name_base,progress)
        with open(file_name,"rb") as f:
            a = pickle.load(f)
        return a

    def unload(self):
        return self.stat_time_arr, self.stat_com_arr, self.stat_rightedge_arr, self.stat_boundedseg_arr, self.stat_total_energy_arr, self.system_state_at_end

    @staticmethod
    def b_is_cp_exists(name_base,progress=None):
        file_name = SimulationResultsCheckPoint.get_file_name(name_base,progress)
        return os.path.exists(file_name)

    @staticmethod
    def b_find_less_advanced_cp(name_base,progress):
        assert(progress is not None)
        # todo this implementation is wierd but works, rewrite this with one for loop.
        files = os.listdir(SimulationResultsCheckPoint.cp_dir_name)
        files_with_same_base = [f for f in files if SimulationResultsCheckPoint.extract_base_name_from_cp_file_name(f) == name_base]
        progresses = [SimulationResultsCheckPoint.extract_progress_from_cp_file_name(f) for f in files_with_same_base]
        most_advanced = None
        for i in range(len(progresses)):
            if progresses[i] < progress and ((most_advanced is None) or (progresses[i] > most_advanced)):
                most_advanced = progresses[i]
        return most_advanced

    @staticmethod
    def get_file_name(name_base,progress):
        if progress is not None:
            if progress == float("inf"):
                progress = "inf"
            else:
                progress = int(progress)
            return os.path.join(SimulationResultsCheckPoint.cp_dir_name, f"{name_base}.prg={progress}.checkpoint")
        else:
            return os.path.join(SimulationResultsCheckPoint.cp_dir_name, f"{name_base}.checkpoint")

    @staticmethod
    def extract_base_name_from_cp_file_name(cp_file_name):
        cp_file_name_no_path = os.path.split(cp_file_name)[1]
        base_name = os.path.splitext(os.path.splitext(cp_file_name_no_path)[0])[0]
        return base_name

    @staticmethod
    def extract_progress_from_cp_file_name(cp_file_name):
        cp_file_name_no_path = os.path.split(cp_file_name)[1]
        assert cp_file_name_no_path.count(".") == 2
        prog_text = os.path.splitext(os.path.splitext(cp_file_name_no_path)[0])[1]
        prog = prog_text.split("=")[1]
        return int(prog)

    def find_vacent_name(self,name_base,progress):
        i = 0
        while True:
            i+=1
            name_out = SimulationResultsCheckPoint.get_file_name(f"{name_base}{i}",progress)
            if not os.path.exists(name_out):
                break
        return name_out

target_time = float("17221")# = float("inf")
b_force_start_clean = False
RUN_NAME = f"bounded_wave_statistics"
checkpoint:SimulationResultsCheckPoint
if (SimulationResultsCheckPoint.b_is_cp_exists(RUN_NAME,progress=target_time)):
    checkpoint = SimulationResultsCheckPoint.load(RUN_NAME,progress=target_time)
else:
    start_cp_progress = SimulationResultsCheckPoint.b_find_less_advanced_cp(RUN_NAME, progress=target_time)
    if start_cp_progress is None or b_force_start_clean:
        # make new run
        obstacle_gen = limited_segment_condition_creator(-0.2, 0.2, 0.19)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=1, c=1, boundary_conditions_generators=boundary_conditions + obstacle_gen)

        stat_com_arr = []
        stat_rightedge_arr = []
        stat_boundedseg_arr = []
        stat_total_energy_arr = []

        stat_time_arr = []
    else:
        # load old run and advance
        checkpoint = SimulationResultsCheckPoint.load(RUN_NAME,progress=start_cp_progress)
        stat_time_arr, stat_com_arr, stat_rightedge_arr, stat_boundedseg_arr, stat_total_energy_arr, m = checkpoint.unload()
        stat_time_arr= list(stat_time_arr)
        stat_com_arr = list(stat_com_arr)
        stat_rightedge_arr = list(stat_rightedge_arr)
        stat_boundedseg_arr = list(stat_boundedseg_arr)
        stat_total_energy_arr = list(stat_total_energy_arr)

    obstacle = [b for b in m.boundary_conditions if type(b) is LimmitedY_SegmentCondition][0]
    # todo generate histogram of com_energy/total_energy after boundery collition at steady state, also try and preserve energy better.

    while m.time < target_time:
        m.step()
        stat_time_arr.append(m.time)
        stat_com_arr.append(np.average(m.u))
        stat_rightedge_arr.append(m.u[-1])
        stat_boundedseg_arr.append(m.u[obstacle.lagrangian_slice.start])
        stat_total_energy_arr.append(float(m.energy_Tot))
        if random.randint(0,1000) == 0:
            print(m.time)
        if random.randint(0,400000) == 0:
            checkpoint = SimulationResultsCheckPoint(np.array(stat_time_arr), np.array(stat_com_arr), np.array(stat_rightedge_arr),
                                                     np.array(stat_boundedseg_arr), np.array(stat_total_energy_arr),
                                                     system_state_at_end=m)
            checkpoint.store(RUN_NAME, progress=m.time)

    checkpoint = SimulationResultsCheckPoint(np.array(stat_time_arr), np.array(stat_com_arr), np.array(stat_rightedge_arr),
                                             np.array(stat_boundedseg_arr), np.array(stat_total_energy_arr),
                                             system_state_at_end=m)
    checkpoint.store(RUN_NAME,progress=m.time)

stat_time_arr, stat_com_arr, stat_rightedge_arr, stat_boundedseg_arr, stat_total_energy_arr, system_state_at_end = checkpoint.unload()

bound_pts_low = np.where((stat_boundedseg_arr == -0.2))[0]
bound_pts_high = np.where((stat_boundedseg_arr == 0.2))[0]

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


def gat_at_bounds(arr):
    return np.array([np.average(arr[l:h]) for (l, h) in boundranges])


def gat_at_midranges(arr):
    return np.array([np.average(arr[l:h]) for (l, h) in midranges])


def num_bins_for_hist(points):
    return int(len(points) ** 0.5 * 5)


plt.figure()
com_v = gat_at_bounds(stat_com_arr)
right_v = gat_at_bounds(stat_rightedge_arr)
bound_V = gat_at_bounds(stat_boundedseg_arr)
minv = min(np.min(com_v), np.min(right_v), np.min(bound_V))
maxv = max(np.max(com_v), np.max(right_v), np.max(bound_V))
plt.hist(com_v, bins=np.linspace(minv, maxv, num_bins_for_hist(com_v)), density=True, color="g", alpha=0.5, label="com")
plt.hist(right_v, bins=np.linspace(minv, maxv, num_bins_for_hist(right_v)), density=True, color="r", alpha=0.5, label="right")
plt.hist(bound_V, bins=np.linspace(minv, maxv, num_bins_for_hist(bound_V)), density=True, color="b", alpha=0.5, label="left")
plt.grid(ls="--")
plt.legend()


plt.figure()
com_v = np.abs(gat_at_bounds(stat_com_arr))
right_v = np.abs(gat_at_bounds(stat_rightedge_arr))
minv = min(np.min(com_v), np.min(right_v))
maxv = max(np.max(com_v), np.max(right_v))
plt.hist(com_v, bins=np.linspace(minv, maxv, num_bins_for_hist(com_v)), density=True, color="g", alpha=0.5, label="com")
plt.hist(right_v, bins=np.linspace(minv, maxv, num_bins_for_hist(right_v)), density=True, color="r", alpha=0.5, label="right")
ylim = plt.ylim()
plt.vlines([np.average(com_v), np.average(right_v)], -10, 10, ["g", "r"], label="averages")
plt.ylim(*ylim)
plt.grid(ls="--")
plt.legend()


com_vel = np.gradient(stat_com_arr) / np.gradient(stat_time_arr)
stat_com_velocity_energy_arr = ((0.5 * system_state_at_end.z[0] / system_state_at_end.c[0]) * com_vel ** 2)

plt.figure()
v = gat_at_midranges(stat_com_velocity_energy_arr) / gat_at_midranges(stat_total_energy_arr)
minv = np.min(v)
maxv = np.max(v)
plt.hist(v, bins=np.linspace(minv, maxv, num_bins_for_hist(v)), density=True, color="g", alpha=0.5, label="energy ratio")
plt.grid(ls="--")
plt.legend()

plt.figure()
v = gat_at_midranges(stat_com_velocity_energy_arr) / gat_at_midranges(stat_total_energy_arr)
v = v[200:]
minv = np.min(v)
maxv = np.max(v)
plt.hist(v, bins=np.linspace(minv, maxv, num_bins_for_hist(v)), density=True, color="g", alpha=0.5, label="energy ratio")
plt.grid(ls="--")
plt.legend()

plt.show()
