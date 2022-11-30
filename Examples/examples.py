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
        m = Medium(u=lambda x: np.sin(2 * x * (np.pi) * 1))
        run_animation(m, 40, b_draw_v=True)

    # examples without obstacles and with constant c,z:
    def test_animate_string_pluck_wave(self):
        pluck_point = 0.7

        def u_func(x_arr):
            return [x / pluck_point if x < pluck_point else 1 - (x - pluck_point) / (1 - pluck_point) for x in x_arr]

        m = Medium(u=u_func, c=0.1)

        run_animation(m, 40)

    # examples without obstacles and with constant c,z:
    def test_animate_string_pluck_half_wave(self):
        pluck_point = 0.7

        def u_func(x_arr):
            return [x / pluck_point if x < pluck_point else 1 - (x - pluck_point) / (1 - pluck_point) for x in x_arr]

        obstacle = limited_segment_condition_creator(0, 10, 0.41, 0.51)  # obstacle that limits u movement in the indexes 100-200

        m = Medium(u=u_func, c=0.1, boundary_conditions_generators=obstacle)

        def on_animation_update():
            if m.time > 100:
                b = [b for b in m.boundary_conditions if type(b) is LimmitedY_SegmentCondition]
                if len(b) == 0:
                    return
                else:
                    del (m.boundary_conditions[m.boundary_conditions.index(b[0])])

        run_animation(m, 40, on_animation_update=on_animation_update)

    def test_animate_higher_standing_wave(self):
        m = Medium(u=lambda x: np.sin(x * (np.pi) * 9), c=0.1)
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
        m = Medium(u=lambda xarr: [x * 1.5 if x <= start_of_c2 else 0.75 + (x - 0.5) * 0.5 for x in xarr],
                   z=lambda xarr: [3 if start_of_c2 < x <= end_of_c2 else 1 for x in xarr])

        def f_edit_plot(lineu, axu, linev, axv, fig):
            fig: plt.Figure
            ax: plt.Axes = fig.gca()
            p = matplotlib.patches.Rectangle((start_of_c2, 1), end_of_c2 - start_of_c2, -2, alpha=0.3, linestyle="--", color="red")
            ax.add_patch(p)
            text1 = plt.text(0.15, 0.016, "c=0.1, z = 0.1")
            text2 = plt.text(0.65, 0.016, "c=0.3, z = 0.3")
            return [p, text1, text2]

        anim = run_animation(m, 20, animation_length=30, f_edit_plot=f_edit_plot)
        # plt.show()
        anim.save(".\\steadty_state2.gif", writer="ffmpeg")

    # steady state when c and z are not constant is no longer straight.
    def test_animate_steadty_state2_out_of_balance(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        m = Medium(u=lambda xarr: [x * 1.5 if x <= start_of_c2 else 0.75 + (x - 0.5) * 0.5 for x in xarr],
                   z=lambda xarr: [3.5 if start_of_c2 < x <= end_of_c2 else 1 for x in xarr])

        def f_edit_plot(lineu, axu, linev, axv, fig):
            fig: plt.Figure
            ax: plt.Axes = fig.gca()
            p = matplotlib.patches.Rectangle((start_of_c2, 1), end_of_c2 - start_of_c2, -2, alpha=0.3, linestyle="--", color="red")
            ax.add_patch(p)
            text1 = plt.text(0.15, 0.016, "c=1, z = 0.1")
            text2 = plt.text(0.65, 0.016, "c=1, z = 0.3")
            return [p, text1, text2]

        anim = run_animation(m, 20, animation_length=30, f_edit_plot=f_edit_plot)
        # plt.show()
        anim.save(".\\steadty_state2_out_of_balance.gif", writer="ffmpeg")

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
            text1 = plt.text(0.15, 0.016, "c=0.1, z = 1")
            text2 = plt.text(0.65, 0.016, "c=0.3, z = 1")
            return [p, text1, text2]

        anim = run_animation(m, 20, initial_limits_u=(-0.02, 0.02), animation_length=30, f_edit_plot=f_edit_plot)
        # plt.show()
        anim.save(".\\impedence_matching.gif", writer="ffmpeg")

    # if z of both parts of an interface is the same, then no reflection should happen ,regardless of c.
    def test_animate_medium_change2(self):
        start_of_c2 = 0.3
        end_of_c2 = 0.6
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        c_arr_creator = lambda xarr: [0.3 if start_of_c2 < x <= end_of_c2 else 0.1 for x in xarr]
        m = Medium(x=1000, v=lambda x: [xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x],
                   c=c_arr_creator, z=c_arr_creator, boundary_conditions_generators=boundary_conditions)

        def f_edit_plot(lineu, axu, linev, axv, fig):
            fig: plt.Figure
            ax: plt.Axes = fig.gca()

            p = matplotlib.patches.Rectangle((start_of_c2, 1), end_of_c2 - start_of_c2, -2, alpha=0.3, linestyle="--", color="red")
            text1 = plt.text(0.05, 0.016, "c=0.1, z = 0.1")
            text2 = plt.text(0.35, 0.016, "c=0.3, z = 0.3")
            text3 = plt.text(0.65, 0.016, "c=0.1, z = 0.1")
            ax.add_patch(p)
            return [p, text1, text2]

        anim = run_animation(m, 20, initial_limits_u=(-0.02, 0.02), animation_length=30, f_edit_plot=f_edit_plot)
        # plt.show()
        anim.save(".\\animate_medium_change1.gif", writer="ffmpeg")

    # if z of both parts of an interface is the same, then no reflection should happen ,regardless of c.
    def test_animate_medium_change1(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        c_arr_creator = lambda xarr: [0.3 if start_of_c2 < x <= end_of_c2 else 0.1 for x in xarr]
        m = Medium(x=1000, v=lambda x: [xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x],
                   c=c_arr_creator, z=c_arr_creator, boundary_conditions_generators=boundary_conditions)

        def f_edit_plot(lineu, axu, linev, axv, fig):
            fig: plt.Figure
            ax: plt.Axes = fig.gca()

            p = matplotlib.patches.Rectangle((start_of_c2, 1), end_of_c2 - start_of_c2, -2, alpha=0.3, linestyle="--", color="red")
            text1 = plt.text(0.15, 0.016, "c=0.1, z = 0.1")
            text2 = plt.text(0.65, 0.016, "c=0.3, z = 0.3")
            ax.add_patch(p)
            return [p, text1, text2]

        anim = run_animation(m, 20, initial_limits_u=(-0.02, 0.02), animation_length=30, f_edit_plot=f_edit_plot)
        # plt.show()
        anim.save(".\\animate_medium_change1.gif", writer="ffmpeg")

    # if zt = 3*zi then R = -0.5, T = 0.5. so transmitted/reflected = -1.
    def test_split_in_half_neg(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: np.array([xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x]) / 0.00125,
                   z=lambda xarr: [0.3 if start_of_c2 < x <= end_of_c2 else 0.1 for x in xarr], boundary_conditions_generators=boundary_conditions)

        m.advance_to_time(0.7)
        self.assertAlmostEqual(np.max(m.u[m.x > 0.5]) / np.min(m.u[m.x < 0.5]), -1, 1)

    # if zt = zi/3 then R = 0.5, T = 1.5. so transmitted/reflected = 3.
    def test_transmit_thrice_reflected(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: np.array([xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x]) / 0.00125,
                   z=lambda xarr: [0.1 if start_of_c2 < x <= end_of_c2 else 0.3 for x in xarr], boundary_conditions_generators=boundary_conditions)

        m.advance_to_time(0.7)
        self.assertAlmostEqual(np.max(m.u[m.x > 0.5]) / np.max(m.u[m.x < 0.5]), 3, 1)

    # if zt = 0 then R = 1, T = 2. so transmitted/reflected = 2.
    def test_transmit_twice_reflected(self):
        start_of_c2 = 0.5
        end_of_c2 = 1
        boundary_conditions = flow_out_boundary_conditions_creator(side="left") + hard_boundary_conditions_creator(side="right")
        m = Medium(x=1000, v=lambda x: np.array([xi if xi < 0.05 else (0.1 - xi if xi < 0.1 else 0) for xi in x]) / 0.00125,
                   z=lambda xarr: [0.01 if start_of_c2 < x <= end_of_c2 else 1 for x in xarr], boundary_conditions_generators=boundary_conditions)

        m.advance_to_time(0.7)
        self.assertAlmostEqual(np.max(m.u[m.x > 0.5]) / np.max(m.u[m.x < 0.5]), 2, 1)

    # --- examples with obstacles and with constant c,z ---:
    def test_animate_sticky_collision_with_obstacle(self):
        obstacle = limited_segment_condition_creator(-0.5, 0.2, 0.1, 0.11)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=lambda x: 0.8 * np.sign(x - 0.5), boundary_conditions_generators=boundary_conditions + obstacle)
        anim = run_animation(m, 20, initial_limits_u=(-1, 1), animation_length=20)
        plt.show()
        # anim.save(".\\sticky_collision_with_obstacle.gif", writer="ffmpeg")

    def test_animate_dynamics_with_loosly_held_end(self):
        obstacle_gen = limited_segment_condition_creator(-0.2, 0.2, 0.19)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(v=1, c=1, boundary_conditions_generators=boundary_conditions + obstacle_gen)

        anim = run_animation(m, 20, initial_limits_u=(-1, 1), animation_length=20)
        # plt.show()
        anim.save(".\\loosly_held_end.gif", writer="ffmpeg")

    def test_animate_coupled_oscillatorsa(self):
        obsticle_slider_position = 1 / 4
        obsticle_slider_width = 0.005
        obstacle = limited_segment_condition_creator(-0.01, 0.01, obsticle_slider_position,
                                                     obsticle_slider_position + obsticle_slider_width)  # obstacle that limits u movement in the indexes 100-200
        boundary_conditions = open_boundary_conditions_creator()
        m = Medium(u=lambda x: (-np.sin(1 * (x - obsticle_slider_position) * (np.pi) / (2 * obsticle_slider_position))) * (x < obsticle_slider_position),
                   boundary_conditions_generators=boundary_conditions + obstacle, c=2)

        anim = run_animation(m, 20, initial_limits_u=(-1, 1), animation_length=30)
        # plt.show()
        anim.save(".\\coupled_oscillators.gif", writer="ffmpeg")


def test_animate_position_switching(self):
    obsticle_slider_position = 0.5
    obsticle_slider_width = 0.01
    obstacle = limited_segment_condition_creator(0, 1.0, obsticle_slider_position,
                                                 obsticle_slider_position + obsticle_slider_width)  # obstacle that limits u movement in the indexes 100-200
    boundary_conditions = hard_boundary_conditions_creator()
    m = Medium(u=lambda x: -np.sin(x * (np.pi) / obsticle_slider_position) * (x < obsticle_slider_position),
               boundary_conditions_generators=boundary_conditions + obstacle)
    anim = run_animation(m, 20, initial_limits_u=(-1, 1), animation_length=9)
    # plt.show()
    anim.save(".\\position_switching.gif", writer="ffmpeg")


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
    anim = run_animation(m, 20, initial_limits_u=(-1, 1), animation_length=60)
    # plt.show()
    anim.save(".\\traveling_wave_by_steps.gif", writer="ffmpeg")

# def test_animate_coupled_oscillators(self):
#     obstacle = limited_segment_condition_creator(-10, 0.2, 0.2)
#     boundary_conditions = open_boundary_conditions_creator(side = "left") + hard_boundary_conditions_creator(u_right=1,side = "right")
#     m = Medium(u=lambda x:x,boundary_conditions_generators=boundary_conditions + obstacle, c=2)
#     run_animation(m, 100)
#
#
# def test_animate_coupled_oscillators(self):
#     boundary_conditions = open_boundary_conditions_creator(side = "left") + hard_boundary_conditions_creator(u_right=1,side = "right")
#     m = Medium(u=lambda x:x,boundary_conditions_generators=boundary_conditions + obstacle, c=2)
#     run_animation(m, 100)
