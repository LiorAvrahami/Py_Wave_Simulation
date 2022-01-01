import matplotlib.pyplot as plt
import matplotlib.patches

from Medium import Medium
from Boundary_Conditions import hard_boundary_conditions_creator, limited_segment_condition_creator, open_boundary_conditions_creator, \
    flow_out_boundary_conditions_creator, BoundaryCondition, LimmitedY_SegmentCondition
from animation_module import run_animation
from unittest import TestCase
import numpy as np


class Tests1D(TestCase):
    def test_energy_c_and_z_are_1(self):
        boundary_conditions = open_boundary_conditions_creator()
        c = 1
        z = 1
        v = 1
        m = Medium(v=lambda x: v * np.sign(x - 0.5),c=c,z=z,
                   boundary_conditions_generators=boundary_conditions)
        e0 = m.energy_Tot,m.energy_K,m.energy_U
        m.advance_to_time(0.5/c)
        e1 = m.energy_Tot, m.energy_K, m.energy_U
        self.assertAlmostEqual(e0[0], e1[0],2)
        self.assertAlmostEqual(e0[2], 0, 10)
        self.assertAlmostEqual(e1[1], 0, 2)

    def test_energy_c_and_z_are_not_1(self):
        boundary_conditions = open_boundary_conditions_creator()
        c = 13
        z = 7
        v = 1
        m = Medium(v=lambda x: v * np.sign(x - 0.5),c=c,z=z,
                   boundary_conditions_generators=boundary_conditions)
        e0 = m.energy_Tot,m.energy_K,m.energy_U
        m.advance_to_time(0.5/c)
        e1 = m.energy_Tot, m.energy_K, m.energy_U
        self.assertAlmostEqual(e0[0], e1[0],2)
        self.assertAlmostEqual(e0[2], 0, 10)
        self.assertAlmostEqual(e1[1], 0, 2)

    def test_stability_c_and_z_are_1(self):
        boundary_conditions = open_boundary_conditions_creator()
        c = 1
        z = 1
        m = Medium(x=10,c=c,z=z,
                   boundary_conditions_generators=boundary_conditions)
        m.u = [(-1) ** n for n in range(10)]
        #todo, make conserve energy and stuff, test stabuility
        m.plot(b_draw_v=False)
        m.step()
        m.plot(b_draw_v=False)
        m.advance_to_time(100)
        m.plot(b_draw_v=False)
        plt.show()
