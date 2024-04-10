import copy

import numpy as np


def generate_gt(n: int):
    return np.random.random_integers(low=0, high=1, size=n).astype(bool)


N = 30
gt = generate_gt(N)
print(gt)


def get_mark(guess: np.ndarray):
    return np.sum(guess == gt)


def print_guess(g):
    for n in g:
        print(int(n), end='')


def main():
    def traverse(lower: int, upper: int, best_guess: np.ndarray) -> np.ndarray:
        # print('-' * (len(guess) - (upper - lower)), end='')
        # print_guess(guess)
        # print()
        # print(f"{lower} {upper}")

        if lower == upper:
            return best_guess
        guess = copy.deepcopy(best_guess)
        guess[lower:upper] = ~guess[lower:upper]
        if get_mark(guess) > get_mark(best_guess):
            best_guess = guess

        mid = (upper - lower + 1) // 2
        better_guess1 = traverse(lower + mid, upper, best_guess)
        better_guess2 = traverse(lower, upper - mid, best_guess)
        if get_mark(better_guess1) > get_mark(better_guess2):
            return better_guess1
        return better_guess2

    def naive():
        best_guess = np.zeros(N).astype(bool)
        for i in range(len(best_guess)):
            try_guess = copy.deepcopy(best_guess)
            try_guess[i] = ~try_guess[i]
            if get_mark(try_guess) >= get_mark(best_guess):
                best_guess = try_guess
        return best_guess

    initial_guess = np.zeros(N).astype(bool)
    res = traverse(0, len(initial_guess), initial_guess)
    print(res)


if __name__ == '__main__':
    main()
