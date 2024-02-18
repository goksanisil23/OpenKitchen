# REINFORCE
Simple REINFORCE algorithm. In the raycast environment, we have:
- 5 rays providing distance measurements
- Each ray measurement is normalized using the max range of raycast before fed into the network.
- Robot has 3 actions to pick from:
    - Drive straight with velocity v
    - Steer right with with velocity v/2
    - Steer left with with velocity v/2
- Robot is rewarded +1 for each step until collision.
- At each step, actor samples an action from the probabilities provided by the network, given the sensor measurements.
- After each episode, policy is updated per REINFORCE algorithm:
    - Discounted rewards are calculated such that the early episodes contain the upcoming discounted rewards:
        - reward_step_0 = discount*(sum_of_all_rewards_till_end_of_episode)
    - Discounted rewards are scaled with the *log probability* of the action that was taken that led to that reward.
        - Hence, both the log_probability and the rewards per each step during an episode needs to be saved until the end of the episode when the
        policy gets updated through back-propagation.
        - The loss for the training is taken as the cumulative **negative** log-likelihood of the action.
            - As we want to maximize the log_probability*reward, but gradient descent in torch is used for minimization, we turn the optimization into a minimization by negating it.  