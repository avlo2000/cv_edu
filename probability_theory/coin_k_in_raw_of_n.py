import numpy as np

m = 3
n = 5

p = 0.5

probs = np.zeros(n + 1)

probs[m] = p ** m
for i in range(m, n):
    probs[i + 1] = probs[i] + (1 - probs[i - m]) * (1.0 - p) * (p ** m)

print(probs[n])


##############################################################
markov = np.zeros([m + 1, m + 1])

for i in range(m):
    markov[0, i] = 0.5
    markov[i + 1, i] = 0.5
markov[-1, -1] = 1

start = np.zeros([m + 1])
start[m] = 1

M_p = np.linalg.matrix_power(markov, n)

state = start @ M_p
prob = state[0]
print(prob)
