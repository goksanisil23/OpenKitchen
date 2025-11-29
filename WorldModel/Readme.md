# Layout
We start with the vanilla version: Vision (V) + Memory (M) + Controller (C)
- Controller is trained separately from V and M

1) Train V:
- randomly rollout some agent on environments to collect enough image data for VAE training
    - leave out 1-2 track for testing
    - also collect actions

2) Use trained V to now train M:
    - Use V to pre-process each image at time t into latent vector z_t
    - RNN can now be trained to predict: P(z_t+1 | a_t, z_t, h_t), where a_t = action, h_t = hidden state of RNN


* Note that V & M has no knowledge about reward signals, they're just trained to compress information and predict the future
* Only C has access to reward, and since it's simple and small number of params, evolutionary algorithms can be used for its
optimization


3) Use only V (i.e. z_t) to train C 
    - Start C as single layer network
    - Check how adding a hidden layer improves C

4) Use both z_t and h_t to train C


VAE only for image distillation vs future estimation too (incorporating current action)


----- Implement this, and then move on to "Car Racing Dreams" part -------