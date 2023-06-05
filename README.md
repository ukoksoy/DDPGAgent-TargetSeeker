# DDPGAgent Target Seeker

The provided code implements the DDPG reinforcement learning algorithm using the TensorFlow Agents (tf_agents) framework within a user-defined environment called TargetSeeker.


- **Target Seeker Environment**

The TargetSeeker environment is a custom environment designed for training an agent using the DDPG algorithm. It represents a 2D space where the agent's objective is to navigate towards a target point.

The environment has the following properties:

State: The state of the environment is a 4-dimensional vector represented as [target_x, target_y, agent_x, agent_y]. It contains the x and y coordinates of the target point and the agent's current position.

Actions: The agent can take continuous actions in the form of a 2-dimensional vector [action_x, action_y]. These actions represent the movement in the x and y directions.

Observations: The agent receives observations of the current state as input. The observation is a 4-dimensional vector representing the state of the environment.

Rewards: The agent receives rewards based on its actions and the current state. The reward calculation takes into account the agent's distance to the target point and the angle between the agent's current heading and the direction towards the target. The agent is rewarded for reducing the distance to the target and aligning its heading towards the target.

Episode Termination: An episode terminates under two conditions: when the agent reaches the target point or when it exceeds the maximum number of steps allowed in an episode.

Rendering: The environment provides the option to visualize the agent's path during training. It uses matplotlib to plot the agent's path, the target point, and some additional information about the episode.

The TargetSeeker environment provides a simple and customizable 2D space where an agent can learn to navigate towards a target point by optimizing its actions based on the DDPG algorithm.



- **DDPG (Deep Deterministic Policy Gradient)**

DDPG (Deep Deterministic Policy Gradient) is a reinforcement learning algorithm that combines policy gradient and Q-learning methods to train agents in environments with continuous action spaces. It is implemented using the TensorFlow Agents (tf_agents) library.

In DDPG, the agent consists of an actor network and a critic network. The actor network learns the policy, mapping states to continuous actions, while the critic network estimates Q-values to evaluate the chosen actions. The agent interacts with the environment, collecting experiences that are stored in a replay buffer.

During training, the agent performs exploration by taking actions based on an exploration policy. Experiences are sampled from the replay buffer to update the critic network by minimizing the difference between estimated and target Q-values. The actor network is updated by computing gradients of the policy's outputs with respect to the actions and applying them to improve the policy.

The tf_agents library provides functionalities for managing the environment, replay buffer, and training steps. It offers pre-built components such as actor and critic networks, optimizers, and drivers to facilitate DDPG implementation.

By iteratively updating the actor and critic networks, DDPG learns an optimal policy, enabling the agent to navigate and make decisions in environments with continuous action spaces.


- **Output Images**

<img src=https://github.com/ukoksoy/TargetSeeker/blob/main/output_images/episode15.png>
<img src=https://github.com/ukoksoy/TargetSeeker/blob/main/output_images/episode23.png>
<img src=https://github.com/ukoksoy/TargetSeeker/blob/main/output_images/episode28.png>
<img src=https://github.com/ukoksoy/TargetSeeker/blob/main/output_images/episode45.png>
<img src=https://github.com/ukoksoy/TargetSeeker/blob/main/output_images/episode49.png>


