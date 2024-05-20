import numpy as np

from control_theory.quantizer import Quantizer


class QLearning:
    def __init__(self, n_states: np.ndarray, n_actions: np.ndarray):
        self.n_actions = n_actions
        self.n_states = n_states
        self.q_table = np.random.random(np.concatenate([self.n_states, self.n_actions]))
        self.learning_rate = 0.8
        self.discount_fact = 0.95
        self.exploration_prob = 0.6
        self._action = None

    def act(self, current_state: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.exploration_prob:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.unravel_index(self.get_q_val(current_state).argmax(), self.get_q_val(current_state).shape)
            action = np.array(action)
        self._action = action
        return action

    def learn(self, current_state: np.ndarray, next_state: np.ndarray, reward: float):
        self.q_table[tuple(current_state)][tuple(self._action)] += \
            self.learning_rate * \
            (reward + self.discount_fact * np.max(self.get_q_val(next_state)) - self.get_q_val(current_state)[tuple(self._action)])

    def get_q_val(self, state: np.ndarray) -> np.ndarray:
        return self.q_table[tuple(state)]


class QLearningQuant:
    def __init__(self, state_quant: Quantizer, action_quant: Quantizer):
        self._state_quant = state_quant
        self._action_quant = action_quant
        self._n_state = np.array([self._state_quant.num_bins] * self._state_quant.dim)
        self._n_actions = np.array([self._action_quant.num_bins] * self._action_quant.dim)
        self.learning = QLearning(self._n_state, self._n_actions)

    def act(self, current_state: np.ndarray):
        curr_state_q = self._state_quant.quantize(current_state)
        action = self.learning.act(curr_state_q)
        return self._action_quant.dequant(action)

    def learn(self, current_state: np.ndarray, next_state: np.ndarray, reward: float):
        curr_state_q = self._state_quant.quantize(current_state)
        next_state_q = self._state_quant.quantize(next_state)
        self.learning.learn(curr_state_q, next_state_q, reward)


if __name__ == '__main__':
    def main():
        n_actions = np.array([4, 4])
        n_states = np.array([16, 2])
        q_learn = QLearning(n_states, n_actions)
        for i in range(1000):
            state = np.random.randint(0, n_states)
            goal_state = n_states - 1
            while np.all(state != goal_state):
                next_state = (state + 1) % n_states
                reward = 1 if np.any(next_state == goal_state) else 0
                q_learn.act(state)
                q_learn.learn(state, next_state, reward)
                state = next_state
            print(i)
        print(q_learn.q_table)
    main()
