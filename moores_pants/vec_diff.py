import sympy as sm
import sympy.physics.mechanics as me
from sympy import pprint

sm.init_printing(use_latex='mathjax')

alpha, beta = sm.symbols('alpha, beta')
a, b, c, d, e, f = sm.symbols('a, b, c, d, e, f')

A = me.ReferenceFrame('A')
B = me.ReferenceFrame('B')
C = me.ReferenceFrame('C')

B.orient_axis(A, alpha, A.x)
C.orient_axis(B, beta, B.y)

v = a*A.x + b*A.y + c*B.x + d*B.y + e*C.x + f*C.y

dvdeBx = v.dot(B.x).diff(e)
dvdeBy = v.dot(B.y).diff(e)
dvdeBz = v.dot(B.z).diff(e)
pprint(dvdeBx*B.x + dvdeBy*B.y + dvdeBz*B.z)
