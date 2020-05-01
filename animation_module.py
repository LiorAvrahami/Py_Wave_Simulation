from Medium import Medium
from TicToc import tic,toc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

def run_animation(medium: Medium,fps:float):
    global ylims
    fig, ax = plt.subplots()
    plt.legend()
    plt.grid()
    line: Line2D = medium.plot()[0]
    ylims = float("inf"), -1 * float("inf")
    tic()

    def update_anim(frame):
        global ylims
        toc()
        tic("calc")
        medium.several_steps(30)
        line.set_data(medium.x, medium.y)
        ylims = min(min(line.get_ydata()), ylims[0]), max(max(line.get_ydata()), ylims[1])
        ylims_mid_dif = (ylims[1] + ylims[0]) / 2, (ylims[1] - ylims[0]) / 2
        ax.set_ylim(ylims_mid_dif[0] - ylims_mid_dif[1] * 1.1, ylims_mid_dif[0] + ylims_mid_dif[1] * 1.1)
        toc()
        tic("wait")

    anim = FuncAnimation(fig, update_anim, interval=1000 / fps)
    plt.show()
