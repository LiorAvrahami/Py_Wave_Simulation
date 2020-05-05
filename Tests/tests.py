from Medium import Medium
from Boundery_Conditions import hard_boundary_conditions_creator,limited_segment_condition_creator,open_boundary_conditions_creator, BoundaryCondition,LimmitedY_SegmentCondition
from animation_module import run_animation
from unittest import TestCase
import numpy as np


class TestRun_animation(TestCase):
    def test_animate_lowest_standing_wave(self):
        m = Medium(u=lambda x: np.sin(x * (np.pi) * 1))
        run_animation(m,40,)

    def test_animate_sticky_collision(self):
        boundery_conditions = open_boundary_conditions_creator()
        m = Medium(v =lambda x: 0.8*np.sign(x - 0.5),boundery_conditions_generators=boundery_conditions)
        run_animation(m, 20)

    def test_animate_sticky_collision_with_obstacle(self):
        obstacle = limited_segment_condition_creator(-0.5,0.2,0.1,0.11) # obstacle that limits u movement in the indexes 100-200
        boundery_conditions = open_boundary_conditions_creator()
        m = Medium(v =lambda x: 0.8*np.sign(x - 0.5),boundery_conditions_generators=boundery_conditions + obstacle)
        run_animation(m, 20)

    def test_animate_overcoming_obstacle(self):
        obsticle_slider_position = 1/4
        obsticle_slider_width = 0.005
        obstacle = limited_segment_condition_creator(-0.01,0.01,obsticle_slider_position,obsticle_slider_position+obsticle_slider_width) # obstacle that limits u movement in the indexes 100-200
        boundery_conditions = open_boundary_conditions_creator()
        m = Medium(u=lambda x: -np.cos(x * (np.pi) / (2*obsticle_slider_position))*(x<obsticle_slider_position),boundery_conditions_generators=boundery_conditions + obstacle)
        run_animation(m, 100)

    def test_animate_coupled_oscillators(self):
        obsticle_slider_position = 0.4
        obsticle_slider_width = 0.01
        obstacle = limited_segment_condition_creator(0,0.2,obsticle_slider_position,obsticle_slider_position+obsticle_slider_width) # obstacle that limits u movement in the indexes 100-200
        boundery_conditions = open_boundary_conditions_creator()
        m = Medium(u=lambda x: -np.cos(x * (np.pi) / (2*obsticle_slider_position))*(x<obsticle_slider_position),boundery_conditions_generators=boundery_conditions + obstacle)
        run_animation(m, 20)

    def test_animate_position_switching(self):
        obsticle_slider_position = 0.5
        obsticle_slider_width = 0.01
        obstacle = limited_segment_condition_creator(0,1.0,obsticle_slider_position,obsticle_slider_position+obsticle_slider_width) # obstacle that limits u movement in the indexes 100-200
        boundery_conditions = hard_boundary_conditions_creator()
        m = Medium(u=lambda x: -np.sin(x * (np.pi) / obsticle_slider_position)*(x<obsticle_slider_position),boundery_conditions_generators=boundery_conditions + obstacle)
        run_animation(m, 20)