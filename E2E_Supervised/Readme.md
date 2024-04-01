# Supervised End-to-End learning with Birdseye Images

We collect observation (birdseye view image) and action (velocity and steering) measurements from an agent that navigates the track using the [Potential Field](/../PotentialField) algorithm. This provides labeled data for our CNN.

In order to have a sufficiently rich training dataset, we collect driving data in 3 main configurations:
- driving closer to left track boundary
- driving closer to right track boundary
- driving in the middle

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/E2E_Supervised/resources/birdseye_IMS_155_0.png" width=20% height=0%>
<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/E2E_Supervised/resources/birdseye_IMS_155_1.png" width=20% height=0%>
<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/E2E_Supervised/resources/birdseye_IMS_155_2.png" width=20% height=0%>

Although the BEV images collected are in RGB, CNN we train uses grayscale images. Architecture of the CNN is simple:
- 5 2d-convolutions, each of which is followed by a Relu
- A flattening layer that takes multi-dim output from the 5th convolutional layer and reduces to single dimension
- 4 fully connected layers, each of which is followed by a Relu.
    - These reduce in size progressively, last layer having the same size as our actions.
- Final tanh layer to normalize the action output.

Activations from the first 3 layers can be seen below:

- 1st layer

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/E2E_Supervised/resources/first_layer_activations.png" width=20% height=0%>

- 2nd layer

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/E2E_Supervised/resources/second_layer_activations.png" width=20% height=0%>

- 3rd layer

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/E2E_Supervised/resources/third_layer_activations.png" width=20% height=0%>

## Trained agent

https://github.com/goksanisil23/OpenKitchen/assets/42399822/b18c4038-f19f-4481-9877-90ba52d36f3d

