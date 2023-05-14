import geometry
import numpy as np
from animation_module import run_animation
import matplotlib.pyplot as plt

u = lambda x: np.exp(-((x-0.5)**2/0.001))
v = lambda x: -(u(x)-u(x-0.001))/(0.001)
m = geometry.make_medium(x=10,u=u,v=v)
a = run_animation(m,40)
plt.show()
