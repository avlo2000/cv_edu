import sympy as sm
import sympy.physics.mechanics as me
from sympy import pprint

psi, theta, phi = me.dynamicsymbols('psi, theta, varphi')

T = me.ReferenceFrame('T')
N = me.ReferenceFrame('N')
T.orient_body_fixed(N, (0, 0, psi), 'YXY')
pprint(T.ang_vel_in(N))
