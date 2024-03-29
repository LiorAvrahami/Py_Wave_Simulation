from Boundary_Conditions import hard_boundary_conditions_creator, open_boundary_conditions_creator, BoundaryCondition, \
    BoundaryCreationFunctor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Callable, Sequence, Any, Tuple, Optional
from numbers import Number


# TODO view Furier transform
# TODO implement pause

class Medium:
    c: np.ndarray
    z: np.ndarray
    time: float
    x: np.ndarray
    dx: np.ndarray
    u: np.ndarray
    v: np.ndarray
    boundary_conditions: Sequence[BoundaryCondition]
    default_dt: float

    def __init__(self, x=None, u=None, v=None, c=None, z=None, b_apply_lpf_to_z_and_c=True,
                 boundary_conditions_generators: List[BoundaryCreationFunctor] = None):
        """
        :param x:
        :param u: callable, wave function initial value as a function of x  defaults to 0
        :param v: callable, wave function initial velocity as a function of x defaults to 0
        :param c: callable or number, medium speed of sound as a function of x defaults to 1
        :param z: callable or number, medium impedence as a function of x defaults to 1
        :param boundary_conditions_generators: the boundary conditions defaults to hard boundary conditions
        """
        self.time = 0
        self.x = np.linspace(0, 1, 500) if x is None else (np.linspace(0, 1, x) if type(x) is int else x)
        self.dx = self.x[1] - self.x[0]

        def handle_field_parameter(parameter, default_val):
            temp = np.ones(self.x.shape)
            return temp * default_val if parameter is None else (
                temp * float(parameter) if isinstance(parameter, Number) else np.array(parameter(self.x), float))

        self.u = handle_field_parameter(u, 0)
        self.v = handle_field_parameter(v, 0)
        self.c = handle_field_parameter(c, 1)
        self.z = handle_field_parameter(z, 1)

        if b_apply_lpf_to_z_and_c:
            size_of_filter = 10
            self.z[size_of_filter:-size_of_filter] = \
                np.convolve(self.z, [1 / size_of_filter for i in range(size_of_filter)], "same")[
                size_of_filter:-size_of_filter]
            self.c[size_of_filter:-size_of_filter] = \
                np.convolve(self.c, [1 / size_of_filter for i in range(size_of_filter)], "same")[
                size_of_filter:-size_of_filter]

        self.default_dt = (np.min(self.dx) / np.max(self.c)) * 0.5
        boundary_conditions_generators = hard_boundary_conditions_creator() if boundary_conditions_generators is None else boundary_conditions_generators
        self.boundary_conditions = BoundaryCreationFunctor.create_from_list(boundary_conditions_generators, self)

    @property
    def energy_K(self) -> float:
        return np.sum(0.5 * self.z / self.c * self.v ** 2 * np.gradient(self.x))

    @property
    def energy_U(self) -> float:
        return np.sum(0.5 * self.c * self.z * np.gradient(self.u, self.x) ** 2 * np.gradient(self.x))

    @property
    def energy_Tot(self) -> float:
        return self.energy_K + self.energy_U

    def step(self, dt=None):
        if dt == None:
            dt = self.default_dt

        # # this is optimal for constant c and z, reference: http://hplgit.github.io/num-methods-for-PDEs/doc/pub/wave/pdf/wave-4print-A4-2up.pdf
        # dv = (self.c ** 2) * np.convolve(self.u,(1,-2,1),"same")/(self.dx**2) * dt
        # self.v += dv
        # self.u += self.v * dt

        medium_changed_term = (np.convolve(self.c * self.z, (1, 0, -1), "same") / self.z) * np.convolve(self.u,
                                                                                                        (1, 0, -1),
                                                                                                        "same") / (
                                          4 * self.dx ** 2)
        medium_changed_term[[0, -1]] = 0
        dv = self.c * dt * (self.c * np.convolve(self.u, (1, -2, 1), "same") / (self.dx ** 2) + medium_changed_term)
        self.v += dv
        self.u += self.v * dt

        for boundary_condition in self.boundary_conditions:
            boundary_condition.apply_condition(self.u, self.v)
        self.time += dt

    def several_steps(self, n, dt=None):
        for i in range(n):
            self.step(dt)

    def advance_to_time(self, target_time):
        while self.time < target_time:
            self.step(self.default_dt)
            # todo understand why this stuff destroies stabuility
            # if self.default_dt < (target_time - self.time):
            #     self.step(self.default_dt)
            # else:
            #     self.step(target_time - self.time)

    def plot(self, b_draw_u=True, b_draw_v=True, **kwargs) -> Tuple[
        Optional[Line2D], Optional[Axes], Optional[Line2D], Optional[Axes], Figure]:
        fig, ax1 = plt.subplots()
        l1, l2, ax2 = None, None, None
        plt.grid()
        if b_draw_u:
            l1 = plt.plot(self.x, self.u, "b", label="displacement", **kwargs)[0]
            for cond in self.boundary_conditions:
                cond.draw(ax1)

        if b_draw_v:
            if b_draw_u:
                ax2 = plt.twinx()
                l2 = plt.plot(self.x, self.v, "g", label="velocity", **kwargs)[0]
            else:
                l1 = plt.plot(self.x, self.v, "g", label="velocity", **kwargs)[0]

        # return lines to be updated in animation
        if b_draw_u and b_draw_v:
            plt.legend(loc='upper right', handles=[l1, l2])
            return l1, ax1, l2, ax2, fig
        elif b_draw_u and (not b_draw_v):
            plt.legend(loc='upper right', handles=[l1])
            return l1, ax1, None, None, fig
        elif (not b_draw_u) and b_draw_v:
            plt.legend(loc='upper right', handles=[l1])
            return None, None, l1, ax1, fig
        elif (not b_draw_u) and (not b_draw_v):
            return None, None, None, None, fig
