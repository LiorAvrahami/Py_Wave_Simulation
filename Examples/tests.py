import matplotlib.pyplot as plt
import matplotlib.patches

from Medium import Medium
from Boundary_Conditions import hard_boundary_conditions_creator, limited_segment_condition_creator, open_boundary_conditions_creator, \
    flow_out_boundary_conditions_creator, BoundaryCondition, LimmitedY_SegmentCondition
from animation_module import run_animation
from unittest import TestCase
import numpy as np


class TestRun_animation(TestCase):
    def test_animate_lowest_standing_wave(self):
        m = Medium(u=lambda x: np.sin(x * (np.pi) * 1))
        run_animation(m, 40)

    def test_animate_steady_state(self):
        m = Medium(u=lambda x: x)
        run_animation(m, 40)

    def test_animate_sticky_collision(self):
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=lambda x: 0.8 * np.sign(x - 0.5), boundary_conditions_generators=boundary_conditions)
        run_animation(m, 20)

    def test_animate_sticky_collision_with_obstacle(self):
        obstacle = limited_segment_condition_creator(-0.5, 0.2, 0.1, 0.11)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=lambda x: 0.8 * np.sign(x - 0.5), boundary_conditions_generators=boundary_conditions + obstacle)
        run_animation(m, 20)

    def test_animate_coupled_oscillators(self):
        obsticle_slider_position = 1 / 4
        obsticle_slider_width = 0.005
        obstacle = limited_segment_condition_creator(-0.01, 0.01, obsticle_slider_position,
                                                     obsticle_slider_position + obsticle_slider_width)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(u=lambda x: -np.cos(x * (np.pi) / (2 * obsticle_slider_position)) * (x < obsticle_slider_position),
                   boundary_conditions_generators=boundary_conditions + obstacle, c=2)
        run_animation(m, 100)

    def test_animate_position_switching(self):
        obsticle_slider_position = 0.5
        obsticle_slider_width = 0.01
        obstacle = limited_segment_condition_creator(0, 1.0, obsticle_slider_position,
                                                     obsticle_slider_position + obsticle_slider_width)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = hard_boundary_conditions_creator()
        m = Medium(u=lambda x: -np.sin(x * (np.pi) / obsticle_slider_position) * (x < obsticle_slider_position),
                   boundary_conditions_generators=boundary_conditions + obstacle)
        run_animation(m, 20)

    def test_animate_traveling_wave_by_steps(self):
        obsticle_slider_positions = np.linspace(0, 1, 10, endpoint=False)
        obsticle_slider_width = 0.01
        obstacles = []
        for i in range(1, len(obsticle_slider_positions)):
            top_bot = i % 2.0
            obstacles += limited_segment_condition_creator(top_bot - 1, top_bot, obsticle_slider_positions[i],
                                                           obsticle_slider_positions[i] + obsticle_slider_width)
        boundary_conditions = hard_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(u=lambda x: -np.sin(x * (np.pi) / obsticle_slider_positions[1]) * (x < obsticle_slider_positions[1]),
                   boundary_conditions_generators=boundary_conditions + obstacles, c=0.1)
        run_animation(m, 20)

    def test_impedence_matching(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: [xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x],
                   c=lambda xarr: [0.3 if start_of_c2 < x <= end_of_c2 else 0.1 for x in xarr], boundary_conditions_generators=boundary_conditions)

        def f_edit_plot(lineu, axu, linev, axv, fig):
            fig: plt.Figure
            ax: plt.Axes = fig.gca()
            p = matplotlib.patches.Rectangle((start_of_c2, 1), end_of_c2 - start_of_c2, -2, alpha=0.3, linestyle="--", color="red")
            ax.add_patch(p)
            return [p]

        run_animation(m, 40, f_edit_plot=f_edit_plot)

    def test_animate_steadty_state2(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        m = Medium(u=lambda xarr: [x*1.5 if  x <= start_of_c2 else 0.75+(x-0.5)*0.5 for x in xarr],
                   z=lambda xarr: [3 if start_of_c2 < x <= end_of_c2 else 1 for x in xarr])

        def f_edit_plot(lineu, axu, linev, axv, fig):
            fig: plt.Figure
            ax: plt.Axes = fig.gca()
            p = matplotlib.patches.Rectangle((start_of_c2, 1), end_of_c2 - start_of_c2, -2, alpha=0.3, linestyle="--", color="red")
            ax.add_patch(p)
            return [p]

        run_animation(m, 40, f_edit_plot=f_edit_plot)
