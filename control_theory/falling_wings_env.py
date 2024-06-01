import control
import numpy as np
import tqdm

from matplotlib import pyplot as plt
from control_theory.falling_wings import falling_wing_fn
from control_theory.pid import PIDControl
from control_theory.q_learning import QLearningQuant
from control_theory.quantizer import Quantizer

np.random.seed(42)


class ParamsGenerator:
    def __init__(self):
        self._params_mn_mx = dict()

    def add_param(self, name: str, min_val, max_val):
        self._params_mn_mx[name] = np.array([min_val, max_val])

    def generate(self):
        params = dict()
        for name, (mn, mx) in self._params_mn_mx.items():
            params[name] = np.random.uniform(mn, mx)
        return params


class FallingWingsEnv:
    def __init__(self):
        self.control_latency = 0.2
        self.max_angle = np.deg2rad(15)
        self._vel_pos_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 200.0])
        self._params_gen = ParamsGenerator()
        self._fill_params_generator()
        self.nl_io_sys = control.NonlinearIOSystem(
            falling_wing_fn,
            inputs=('u_alpha', 'u_beta'),
            states=('x velocity', 'y velocity', 'z velocity', 'x position', 'y position', 'z position'),
            params=self._params_gen.generate()
        )
        self.trg_pose = np.random.uniform(
            low=np.array([-30, -30, -100.0]),
            high=np.array([+30, +30, 0.0])
        )
        self._vis_trj = []

    def _fill_params_generator(self):
        self._params_gen.add_param('rho_air', 1.290, 1.295)
        self._params_gen.add_param('C_alpha', 0.1, 0.2)
        self._params_gen.add_param('C_beta', 0.1, 0.2)
        self._params_gen.add_param('C_drag', 0.2, 0.35)
        self._params_gen.add_param('area', 0.02, 0.03)
        self._params_gen.add_param('mass', 1.3, 1.6)
        self._params_gen.add_param('g', -9.81, -9.81)
        self._params_gen.add_param('wind', 0.0 * np.array([-0.1, -0.1, 0.0]), 0.0 * np.array([+0.1, +0.1, 0.0]))

    def reset(self):
        self._vel_pos_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 500.0])
        self.nl_io_sys.params = self._params_gen.generate()
        self.trg_pose = np.random.uniform(
            low=np.array([-30, -30, 0.0]),
            high=np.array([+30, +30, 0.0])
        )
        self._vis_trj = []

    def step(self, action: np.ndarray):
        n = 200
        timeline = np.linspace(0.0, 0.1, 200)
        action = np.clip(action, -self.max_angle, +self.max_angle)
        action = np.repeat(action[:, np.newaxis], n, axis=1)
        _, _, x = control.input_output_response(self.nl_io_sys, timeline, action, self._vel_pos_state, return_x=True)
        self._vel_pos_state = x[:, -1]
        self._vis_trj.append(self.state[:3])
        dist_to_trg = np.linalg.norm(self.state[:3] - self.trg_pose)
        if dist_to_trg <= 2.0:
            return 1.0 / (dist_to_trg + 1.0), True
        if self.state[2] <= self.trg_pose[2]:
            return -1.0, True
        return 0.0, False

    @property
    def state(self):
        return np.concatenate([self._vel_pos_state[3:], self.trg_pose])

    def plot(self, ax: plt.Axes):
        trj = np.stack(self._vis_trj)
        ax.plot(trj[:, 0], trj[:, 1], trj[:, 2])
        ax.scatter(self.trg_pose[0], self.trg_pose[1], self.trg_pose[2], marker='X', c='red')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')


def main():
    env = FallingWingsEnv()
    env.reset()
    episodes = 5000
    pre_episodes = 500
    pid = PIDControl()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    show_episodes_period = 20
    state_quantizer = Quantizer(
        np.array([-40, -40, 0.0, -20, -20]), np.array([+40, +40, 500, +20, +20]), 10
    )
    action_quantizer = Quantizer(
        np.array([-env.max_angle, -env.max_angle]), np.array([+env.max_angle, +env.max_angle]), 20
    )
    q_learning = QLearningQuant(state_quantizer, action_quantizer)
    q_learning.learning.exploration_prob = 0.3

    for i in tqdm.tqdm(range(pre_episodes)):
        episode_end = False
        current_state = env.state
        while not episode_end:
            u = pid.control(env.state) / env.state[2]
            reward, episode_end = env.step(u)
            next_state = env.state
            q_learning.set_action(u)
            q_learning.learn(current_state[:-1], next_state[:-1], reward)
        env.plot(ax)
        env.reset()
        if (i + 1) % show_episodes_period == 0:
            plt.title(f"Episodes: {i + 1}")
            plt.legend()
            plt.savefig(f"results/pre_episode{(i + 1) // show_episodes_period}.png")
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for i in tqdm.tqdm(range(episodes)):
        episode_end = False
        current_state = env.state[:-1]
        while not episode_end:
            u = q_learning.act(current_state=current_state)
            reward, episode_end = env.step(u)
            next_state = env.state[:-1]
            q_learning.learn(current_state, next_state, reward)
        env.plot(ax)
        env.reset()
        if (i + 1) % show_episodes_period == 0:
            plt.title(f"Episodes: {i + 1}")
            plt.legend()
            plt.savefig(f"results/episode{(i + 1) // show_episodes_period}.png")
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


if __name__ == '__main__':
    main()
