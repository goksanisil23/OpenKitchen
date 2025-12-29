# World Model with VAE & RNN
The setup was inspired by https://worldmodels.github.io/

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


## Steps to run:
1. train_vae.py

This will read traininig images by default from "../FieldNavigators/collect_data/build/measurements_and_actions/", and will test against the images under "../FieldNavigators/collect_data/build/test_dir/"
It will generate vae_outputs folder that contains the model.

2. generate_latent_dataset.py

This will generate a dataset of latent vectors (latent_dataset.npz), using the VAE trained above.

3. train_rnn.py

This will use the latent dataset only to learn to predict the next step's (t+1) latent vector.

4. Run test_rnn.py and test_vae.py to visually verify both networks.

5. run export.py

This will save the rnn, vae encoder and vae decoder as .pt files that are usable from C++ via torch scripting

## Visualization
Right most image is RNN's prediction about what the world state is going to be at t+1, by using the latent vector z_t provided by the VAE, whose decoded version is shown in the middle

<p align="center">
  <img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/WorldModelVaeRnn/vae_rnn.png"/>
</p>

### TODO:
Have not managed to get CMA-ES stable with either the VAE or RNN outputs. These networks give good reconstruction of the BEV images however the latent vectors does not seem to be sufficient for stable driving. Agents do improve during training but cannot complete the track yet. Also need to keep the z_t dimension rather low due to the limitations with the covariance matrix size in CMA-ES.