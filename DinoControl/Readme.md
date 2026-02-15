# Imitation Learning with Dino encodings
- Collect data using FieldNavigators/collect_data
    - Best performance is usually obtained when both random and 3 lanes datasets are used in combination, so that the agent can see how to recover from non-ideal trajectories as well as seeing sufficient samples of nominal driving.
- Run train_dino_control.py, which will be using the CLS (classification) token [B,D] instead of patch tokens [B.N.D] to embed the BEV image and forward to a linear policy head that minimizes the loss against the expert actions.
- Run export_dino_control.py to generate a scripted version of the trained network so that it can be called from C++.
- run main_dino_control.cpp
