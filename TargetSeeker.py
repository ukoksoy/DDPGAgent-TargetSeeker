#!pip install tf_agents

import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.agents import DdpgAgent
from tf_agents.agents.ddpg import actor_network, critic_network
from tf_agents.environments import tf_py_environment, py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import train_utils
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class TargetSeeker(py_environment.PyEnvironment):
    def __init__(self, max_episode_steps=500, min_action=-1.0, max_action=1.0, target_radius=1, min_limit=-20, max_limit=20, fig_size=3, output_interval=1, image_option='', image_path='images'):
        py_environment.PyEnvironment.__init__(self)
        self._min_action = min_action
        self._max_action = max_action
        self._observation_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(shape=(2,),dtype=np.float32, minimum=self._min_action, maximum=self._max_action, name='action')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._episode_number = 0
        self._max_episode_steps = max_episode_steps
        self._target_radius = target_radius
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._fig_size = fig_size
        self._output_interval = output_interval
        self._image_option = image_option
        self._image_path = image_path
        if not isinstance(self._output_interval, int) or self._output_interval < 0:
            raise ValueError("'self._output_interval' should be a positive integer.")
        self._image_options = ['', 'save', 'show', 'saveandshow', 'showandsave']
        if self._image_option not in self._image_options:
            raise ValueError(f"Invalid image_option: '{self._image_option}'. Must be one of {self._image_options}.")
        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._target_point = (np.random.uniform(self._min_limit+self._target_radius, self._max_limit-self._target_radius), 
                             np.random.uniform(self._min_limit+self._target_radius, self._max_limit-self._target_radius))
        self._start_point = (np.random.uniform(self._min_limit+self._target_radius, self._max_limit-self._target_radius), 
                             np.random.uniform(self._min_limit+self._target_radius, self._max_limit-self._target_radius))
        self._state = np.array([self._target_point[0], self._target_point[1], self._start_point[0], self._start_point[1]], dtype=np.float32)
        self._path = [self._start_point]
        self._episode_ended = False
        self._in_target = False
        self._total_rewards = 0
        self._step_counter = 0
        return ts.TimeStep(step_type=ts.StepType.FIRST, reward=np.float32(0.0), discount=np.float32(0.0), observation=self._state)

    def draw_path(self):
        fig, ax = plt.subplots(figsize=(self._fig_size, self._fig_size))
        x_coords = [coord[0] for coord in self._path]
        y_coords = [coord[1] for coord in self._path]
        line_color = '#155B8C'
        ax.plot(x_coords, y_coords, line_color, linewidth=self._fig_size/5)
        color = '#00FF00' if self._in_target else '#FF0000'
        radius = self._fig_size/10
        target_radius = max(self._target_radius, radius)
        circle = Circle(self._target_point, target_radius, fill=False, color=line_color, linewidth=self._fig_size/5)
        ax.add_artist(circle)
        circle = Circle(self._start_point, radius, fill=True, color=line_color, linewidth=0)
        ax.add_artist(circle)
        circle = Circle((self._state[2], self._state[3]), radius, fill=True, color=color, linewidth=0)
        ax.add_artist(circle)
        ax.set_xlim(self._min_limit, self._max_limit)
        ax.set_ylim(self._min_limit, self._max_limit)
        ax.tick_params(axis='both', labelsize=2*self._fig_size)
        step_info = f' Episode:{self._episode_number} | In Target:{self._in_target} | Steps:{self._step_counter} | Rewards:{int(10*self._total_rewards)/10}'
        ax.text(self._min_limit, self._max_limit+2, step_info, ha='left', va='top', fontsize=self._fig_size*1.9)
        if 'show' in self._image_option: plt.show(fig)
        else: print(step_info)
        if 'save' in self._image_option:
            os.makedirs(self._image_path, exist_ok=True)
            plt.savefig(f"{self._image_path}/path_{self._episode_number}.png")
        plt.close(fig)
    
    def random_action(self):
        return np.random.uniform(self._min_action, self._max_action, size=(2,))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._step_counter += 1
        previous_distance = np.linalg.norm(np.array([self._state[2], self._state[3]]) - np.array([self._target_point[0], self._target_point[1]]))
        next_state = np.array([self._target_point[0], self._target_point[1], self._state[2] + action[0], self._state[3] + action[1]], dtype=np.float32)
        self._path.append((next_state[2], next_state[3]))
        last_distance = np.linalg.norm(np.array([next_state[2], next_state[3]]) - np.array([self._target_point[0], self._target_point[1]]))
        angle1 = np.arctan2(next_state[3] - self._state[3], next_state[2] - self._state[2])
        angle2 = np.arctan2(self._target_point[1] - self._state[3], self._target_point[0] - self._state[2])
        angle_reward = np.cos(angle1 - angle2)
        distance_reward = previous_distance - last_distance
        reward = angle_reward + distance_reward - 1
        self._total_rewards += reward
        done = True if last_distance <= self._target_radius or self._step_counter >= self._max_episode_steps else False
        self._in_target = True if last_distance <= self._target_radius else False
        self._state = next_state

        if done:
            self._episode_number += 1
            if self._output_interval > 0:
                if self._episode_number % self._output_interval == 0:
                    self.draw_path()
                self._episode_ended = True
            return ts.TimeStep(step_type=ts.StepType.LAST, reward=np.float32(reward), discount=np.float32(0.0), observation=self._state)
        else:
            return ts.TimeStep(step_type=ts.StepType.MID, reward=np.float32(reward), discount=np.float32(1.0), observation=self._state)


train_py_env = TargetSeeker(image_option='show')
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

num_iterations = 100000
initial_collect_steps = 100
collect_steps_per_iteration = 1
batch_size = 64
replay_buffer_max_length = 10000
train_step = train_utils.create_train_step()

actor_net = actor_network.ActorNetwork(
    input_tensor_spec = train_env.observation_spec(),
    output_tensor_spec = train_env.action_spec(),
    fc_layer_params = (400,300),
    activation_fn = tf.keras.activations.relu,
    name = 'ActorNetwork')

critic_net = critic_network.CriticNetwork(
    input_tensor_spec=(train_env.observation_spec(), train_env.action_spec()),
    observation_fc_layer_params=(400,),
    joint_fc_layer_params=(400,),
    name='CriticNetwork')

agent = DdpgAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    ou_stddev=0.2,
    ou_damping=0.15,
    target_update_tau=0.05,
    target_update_period=5,
    gamma=0.99,
    reward_scale_factor=1,
    train_step_counter=train_step)

agent.initialize()

agent.train = common.function(agent.train, autograph=False, reduce_retracing=True)
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec = agent.collect_data_spec, batch_size = train_env.batch_size, max_length = replay_buffer_max_length)
train_env.reset()
collect_op = dynamic_step_driver.DynamicStepDriver(train_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=initial_collect_steps)
collect_op.run()
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)                                                                                                     

for iteration in tqdm(range(num_iterations)):
    collect_op.run(maximum_iterations=collect_steps_per_iteration)
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

