import geometry
import numpy as np
from animation_module import run_animation
import matplotlib.pyplot as plt
u = lambda x: np.exp(-((x-0.5)**2/0.0001))
d = 0.0001
v = lambda x: -(u(x + d/2)-u(x-d/2))/d
m = geometry.make_medium(x=2000,u=u,v=v)
a = run_animation(m,40)
plt.show()
