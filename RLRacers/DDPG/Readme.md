# Deep Deterministic Policy Gradient

## Actor
- Actor network represents the policy: [State] -> [Action]
- Actor's parameters are updated to maximize the Critic's value estimates.
    - Given the current state and the action that actor currently associates to the state, the critic produces a value estimate.
        - This value is maximized during training process (or the inverse is minimized).

## Critic
- Critic network represents the estimated Q-value of taking an action in a given state: [State,Action] -> [Value]
- Temporal difference (TD) error is calculated as the difference between predicted and target Q-values.
    - Target-Q is calculated using the reward from the current action plus the estimated Q-value of the next state-action pair, discounted by factor gamma.
        - Next state action pair is obtained from the samples within the replay buffer using the frozen target actor and critic networks.
- Training process minimizes the TD error.

The target actor & critic networks are updated using "soft updates": a weighted sum of their current weights and their corresponding learnt networks, where learnt network's influence is kept low. (To ensure target networks change slowly over time for stability)


* DDPG allows to have continuous action space, via the tanh function at the end of forward pass (scaling between [MIN,MAX])
* Its called "deterministic" since the Actor network directly produces the action instead of probabilities of actions.