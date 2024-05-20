import sympy as sm
import sympy.physics.mechanics as me
from sympy import pprint
from sympy.solvers.ode.systems import dsolve_system

sm.init_printing()

N = me.ReferenceFrame('N')
m1, m2, G = sm.symbols('m1, m2, G')
x, y, z = me.dynamicsymbols('x, y, z')

p = x * N.x + y * N.y + z * N.z

d_normalized = p.normalize()
R_sq = p.dot(p)


t = sm.symbols('t')
F_g = d_normalized * G * m1 * m2 / R_sq

ma = m1 * p.diff(t, N).diff(t, N)

lhs = list(ma.to_matrix(N))
rhs = list(F_g.to_matrix(N))

eqs = [sm.Eq(l, r) for l, r in zip(lhs, rhs)]
pprint(eqs)
dsolve_system(eqs)
