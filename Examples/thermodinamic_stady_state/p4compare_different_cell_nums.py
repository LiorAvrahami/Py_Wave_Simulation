print("analysing 0/3")
import p3SimulateAndAnalyse400Cells
print("analysing 1/3")
import p3SimulateAndAnalyse450Cells
print("analysing 2/3")
import p3SimulateAndAnalyse500Cells
print("analysing 3/3")

import matplotlib.pyplot as plt

num_cells_arr = [
p3SimulateAndAnalyse400Cells.num_cells,
p3SimulateAndAnalyse450Cells.num_cells,
p3SimulateAndAnalyse500Cells.num_cells
]

steady_state_energy_ratio_theory = [1/a for a in num_cells_arr]

steady_state_energy_ratio = [
p3SimulateAndAnalyse400Cells.energy_ratio_mean,
p3SimulateAndAnalyse450Cells.energy_ratio_mean,
p3SimulateAndAnalyse500Cells.energy_ratio_mean
]

steady_state_energy_ratio_uncertainty = [
p3SimulateAndAnalyse400Cells.energy_ratio_std_of_mean,
p3SimulateAndAnalyse450Cells.energy_ratio_std_of_mean,
p3SimulateAndAnalyse500Cells.energy_ratio_std_of_mean
]

plt.figure()
plt.errorbar(num_cells_arr,steady_state_energy_ratio,steady_state_energy_ratio_uncertainty,fmt="o",capsize=3)
plt.plot(num_cells_arr,steady_state_energy_ratio_theory,"g")

plt.show()