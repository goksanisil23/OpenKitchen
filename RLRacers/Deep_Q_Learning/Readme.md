# Deep-Q Learning

In this implementation of DQN, we train a simple MLP that predicts the Q-value associated to each possible discrete action the agent can apply, given a state.

- State is simply a set of (normalized) range values per each ray emitted from the sensor.
- When following the greedy policy, the discrete action index is mapped to a floating steering and acceleration value for the robot.
    - Action index is obtained via argmax applied on the Q-outputs of the MLP.

Training occurs at the end of each episode, through the samples accumulated in the replay buffer, which consists of:
- current state
- action taken in the current state
- next state that is consequence of current action above
- reward obtained during this action

The "label" we train against is the temporal difference, which represents the estimate of the future expected reward, as we have access to the `next state` sample from the agent.

$$
label = y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)
$$

loss for action A<sub>t</sub> taken at S<sub>t</sub>: 
$$
loss = (y_t - Q(S_t,A_t;\theta))^2
$$

The 2nd term represents the maximum predicted Q-value for the next state, using the MLP.

**Note that we only update the Q value associated to the action taken for that step via the TD term, and the gradients for other actions are zeroed out, since we did not make any observation from the environment regarding the other actions (and their rewards)**

*Note that some implementations uses a separate target network to get this max-Q value from, which gets updated in slower intervals to stabilize the learning process*