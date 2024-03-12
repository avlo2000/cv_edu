import copy

import numpy as np
np.random.seed(0)

N = 8
gt = np.random.randint(0, 2, N).astype(bool)


def get_score(g) -> int:
    return np.sum(g == gt).item()


print(gt)
initial = np.zeros(N).astype(bool)
print(get_score(initial))
initial[:N//2] = True
print(get_score(initial))


def go(left: int, right: int, space: int, guess: np.ndarray, prev_score: int, zeros: bool):
    if left == right:
        return
    to_print = f'[{left}, {right}] ---> ' if left + 1 != right else f'[{left}, {right}]'
    score = get_score(guess)
    to_print = f"score: {score}, diff: {score - prev_score} " + to_print
    print(to_print, end='')
    mid = (right - left + 1) // 2

    guess_l = copy.deepcopy(guess)
    guess_l[mid:] = ~guess_l[mid:]
    go(left + mid, right, space + len(to_print), guess_l, score, ~zeros)
    print()
    print(' ' * (space + len(to_print)), end='')

    guess_r = copy.deepcopy(guess)
    guess_r[:mid] = ~guess_r[:mid]
    go(left, right - mid, space + len(to_print), guess_r, score, ~zeros)


guess = np.zeros(N).astype(bool)
go(0, N, 0, guess, get_score(guess), False)
