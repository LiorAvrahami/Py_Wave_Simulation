from Medium import Medium
from animation_module import run_animation
from unittest import TestCase
import numpy as np


class TestRun_animation(TestCase):
    def test_animate_lowest_standing_wave(self):
        m = Medium(y=lambda x: np.sin(x * (np.pi) * 1))
        run_animation(m,40)

    def test_animate_collision(self):
        m = Medium(v =lambda x: 0.05*np.sign(x - 0.5))
        run_animation(m, 20)
