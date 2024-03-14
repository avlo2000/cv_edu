import copy

import numpy as np
np.random.seed(0)

N = 16
gt = np.random.randint(0, 2, N).astype(bool)


def get_score(g) -> int:
    return np.sum(g == gt).item()


for i in gt:
    print(1 if i else 0, end='')
print()
initial = np.zeros(N).astype(bool)
cnt_zeros = get_score(initial)
cnt_ones = len(initial) - cnt_zeros


def go(left: int, right: int, space: int, guess: np.ndarray, prev_score: int, d: int):
    if left == right:
        return
    to_print = f'[{left}, {right}] -> ' if left + 1 != right else f'[{left}, {right}]'
    score = get_score(guess)
    k = score - prev_score
    to_print = f"score: {score}, " + to_print

    n = right - left
    true = (n + k) // 2
    false = n - true
    to_print = f"'t': {false} | {to_print}"
    to_print = f"'f': {true} | {to_print}"
    print(to_print, end='')
    mid = (right - left + 1) // 2

    guess_l = copy.deepcopy(guess)
    guess_l[mid:] = ~guess_l[mid:]
    go(left + mid, right, space + len(to_print), guess_l, score, d + 1)
    print()
    print(' ' * (space + len(to_print)), end='')

    guess_r = copy.deepcopy(guess)
    guess_r[:mid] = ~guess_r[:mid]
    go(left, right - mid, space + len(to_print), guess_r, score, d + 1)


guess = np.zeros(N).astype(bool)
go(0, N, 0, guess, get_score(guess), 0)
