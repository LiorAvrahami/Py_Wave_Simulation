import matplotlib.pyplot as plt
import matplotlib.patches

from Medium import Medium
from Boundary_Conditions import hard_boundary_conditions_creator, limited_segment_condition_creator, open_boundary_conditions_creator, \
    flow_out_boundary_conditions_creator, BoundaryCondition, LimmitedY_SegmentCondition
from animation_module import run_animation
from unittest import TestCase
import numpy as np


# these animations examples start with the word "test", are are placed inside a TestCase only for it to be possible to run them independently in pycharm.
# these arent the tests, these are just fun example runs of the animation utility.
class ExampleAnimations(TestCase):
    # examples without obstacles and with constant c,z:
    def test_animate_lowest_standing_wave(self):
        m = Medium(u=lambda x: np.sin(x * (np.pi) * 1))
        run_animation(m, 40)

    def test_animate_higher_standing_wave(self):
        m = Medium(u=lambda x: np.sin(x * (np.pi) * 9),c=0.1)
        run_animation(m, 40)

    def test_animate_steady_state(self):
        m = Medium(u=lambda x: x)
        run_animation(m, 40)

    def test_animate_sticky_collision(self):
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=lambda x: 0.8 * np.sign(x - 0.5), boundary_conditions_generators=boundary_conditions)
        run_animation(m, 20)

    # examples without obstacles and with non-constant c,z:

    # steady state when c and z are not constant is no longer straight.
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

    # if z of both parts of an interface is the same, then no reflection should happen ,regardless of c.
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

    # if zt = 3*zi then R = -0.5, T = 0.5. so transmitted/reflected = -1.
    def test_split_in_half_neg(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: np.array([xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x])/0.00125,
                   z=lambda xarr: [0.3 if start_of_c2 < x <= end_of_c2 else 0.1 for x in xarr], boundary_conditions_generators=boundary_conditions)

        m.advance_to_time(0.7)
        self.assertAlmostEqual(np.max(m.u[m.x>0.5])/np.min(m.u[m.x<0.5]),-1, 1)

    # if zt = zi/3 then R = 0.5, T = 1.5. so transmitted/reflected = 3.
    def test_transmit_thrice_reflected(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: np.array([xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x])/0.00125,
                   z=lambda xarr: [0.1 if start_of_c2 < x <= end_of_c2 else 0.3 for x in xarr], boundary_conditions_generators=boundary_conditions)

        m.advance_to_time(0.7)
        self.assertAlmostEqual(np.max(m.u[m.x>0.5])/np.max(m.u[m.x<0.5]),3, 1)

    # if zt = 0 then R = 1, T = 2. so transmitted/reflected = 2.
    def test_transmit_twice_reflected(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: np.array([xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x])/0.00125,
                   z=lambda xarr: [0.01 if start_of_c2 < x <= end_of_c2 else 1 for x in xarr], boundary_conditions_generators=boundary_conditions)

        m.advance_to_time(0.7)
        self.assertAlmostEqual(np.max(m.u[m.x>0.5])/np.max(m.u[m.x<0.5]),2, 1)

    # examples with obstacles and with constant c,z:
    def test_animate_sticky_collision_with_obstacle(self):
        obstacle = limited_segment_condition_creator(-0.5, 0.2, 0.1, 0.11)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=lambda x: 0.8 * np.sign(x - 0.5), boundary_conditions_generators=boundary_conditions + obstacle)
        run_animation(m, 20)

    def test_animate_dynamics_with_loosly_held_end(self):
        obstacle_gen = limited_segment_condition_creator(-0.2, 0.2, 0.19)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=10,c=10, boundary_conditions_generators=boundary_conditions + obstacle_gen)

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

        com_dot = plt.Line2D
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
            nonlocal com_dot
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

        run_animation(m, 20,f_edit_plot=f_edit_plot,on_animation_update=on_animation_update)
        plt.show()


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
