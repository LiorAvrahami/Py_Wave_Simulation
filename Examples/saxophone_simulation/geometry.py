from Medium import Medium
import Boundary_Conditions
def make_medium(x,u,v):
    boundary_conditions = Boundary_Conditions.hard_boundary_conditions_creator(u_left=0,side="left") + Boundary_Conditions.open_boundary_conditions_creator(side="right")
    sax_medium = Medium(x=x,u=u,v=v,z=lambda x:x**2,boundary_conditions_generators=boundary_conditions)
    return sax_medium