import sys

# allows imports relative to project's base
base_name = "Py_Wave_Simulation"
sys.path.insert(0, __file__[:__file__.index(base_name) + len(base_name)])

from SimulationCheckpoint import SimulationResultsCheckPoint
from p1RunSimulationForAnalysis import RunParameters, runSimulationForAnalysis
from p2AnalyseSimulationResults import AnalysisParameters, analyseResults

# -- Ready Run Parameters
run_parameters = RunParameters()
# num_cells - the number of cells to be used
run_parameters.num_cells = 400
# RUN_NAME - the name for the simulation run. this is used for checkpoints.
run_parameters.RUN_NAME = f"bounded_wave_statistics.num_cells={run_parameters.num_cells}"
# target_time - the simulation time to which the simulation will be advanced.
run_parameters.target_time = None
# b_force_start_clean - whether or not to check if an old run's checkpoint exists,
# if one exists and "b_force_start_clean" is false then the old checkpoint will be used.
run_parameters.b_force_start_clean = False
# the amount of simulation steps between prints to terminal
run_parameters.frequency_of_print_eta = 20000
# the amount of simulation steps between creation of checkpoints
run_parameters.frequency_of_save_checkpoint = 1000000

if run_parameters.target_time is not None:
    # -- Run Simulation
    runSimulationForAnalysis(run_parameters)

# -- Ready Analysis Parameters
analysis_parameters = AnalysisParameters(run_parameters)
#  index_range_of_steady_state - the range of collision indexes that seems to be in steady state before system collapses
analysis_parameters.index_range_of_steady_state = (210,430)
# b_plot determines whether or not will plot
analysis_parameters.b_plot = __name__ == "__main__"

# -- Plot Simulation Analysis
res = analyseResults(analysis_parameters)

# -- Unload Analysis Results
num_cells = run_parameters.num_cells
system_state_at_end = res.system_state_at_end
stat_time_arr = res.stat_time_arr
stat_com_arr = res.stat_com_arr
stat_rightedge_arr = res.stat_rightedge_arr
stat_boundedseg_arr = res.stat_boundedseg_arr
stat_total_energy_arr = res.stat_total_energy_arr
midranges = res.midranges
boundranges = res.boundranges
stat_indexes_compact = res.stat_indexes_compact
stat_total_energy_arr_compact = res.stat_total_energy_arr_compact
stat_time_arr_compact = res.stat_time_arr_compact
indexes_of_midranges = res.indexes_of_midranges
midrange_index_arr_compact = res.midrange_index_arr_compact
com_vel = res.com_vel
stat_com_velocity_energy_arr = res.stat_com_velocity_energy_arr
energy_ratio = res.energy_ratio
steady_state_idx = res.steady_state_idx
steady_state_energy_ratio_population = res.steady_state_energy_ratio_population
energy_ratio_mean = res.energy_ratio_mean
energy_ratio_std_of_mean = res.energy_ratio_std_of_mean
energy_ratio_theory = res.energy_ratio_theory

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.show()