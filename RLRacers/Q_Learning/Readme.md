# Q-learning
Simple q-learning algorithm. In the raycast environment, we have:
- 5 rays providing distance measurements
- Each ray is divided into 3 sections based on proximity: Near, mid, far
    - This provides 5^3=243 unique state representations
- Robot has 3 actions to pick from:
    - Drive straight with velocity v
    - Steer right with with velocity v/2
    - Steer left with with velocity v/2
- Robot is rewarded based on:
    - Collisions giving large (-) penalty
    - Progression along the track giving proportional rewards

In this implementation, after each episode, the robot colony shares their Q-tables with each other by taking the average of each q-value associated to all state-action pairs.


https://github.com/goksanisil23/OpenKitchen/assets/42399822/9cf026f0-b661-4b51-8a8c-f1c78624fe5f