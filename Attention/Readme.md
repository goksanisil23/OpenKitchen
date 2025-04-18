# Attention

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/Attention/resources/self_attention.svg" width=52% height=30%>

- Every feature (input) appears in 3 different places:
    - input layer (x3) -> value
    - In weight layer: as common 1st multiplier in a given row  -> query
    - In weight layer: as the 2nd multipler -> key
        - Y3 = (X3⋅X1)*X1 + (X3⋅X2)*X2 + (X3⋅X3)*X3 +(X3⋅X4)*X4 +(X3⋅X5)*X5
                q                             k  v

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/Attention/resources/key_query_value.svg" width=52% height=30%>

Its just like a dictionary but instead:
- Every key matches to a given query to some extent. (determined by dot product)
- Mixture of all values is returned (Y3 above is influenced by all X1 ... X5 as values)

**Self attention**: When keys, queries and values are all from the same set (X1 ... X5)
- To parametrize so that keys, values and queries can specialize, we introduce some learnable parameters K, Q, V and biases
    - Otherwise there's nothing to be learned int Y = W*X since W is only contains X.
<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/Attention/resources/key_query_value_params.svg" width=52% height=30%>

**Multi-head self-attention**: Self attention layers applied in parallel.
- Input is split into many, but lower dimension.
- Each of those is passed into a separate self-attention with different K,Q,V.
- The end result is concatanated, and then passed through a final linear transform (W0)
    - Since the input dimension was lowered according to split, after the concatanation of the outputs, we end up with the same dimensions as a single head self-attention.

In diagram below, 
- our input with 5 sequential elements (X1 ... X5) is split into 4 pieces.
- An element in each input produces a key-value-query
<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/Attention/resources/multi_head_self_attention.svg" width=52% height=30%>



## Vision Transformer Intuition
General architecture can be thought as: Communication + Computation
Assume 128x128 image, chunked into smaller patches of (16x16) --> 64 patches
- Each of these 64 patches are considered to be a separate input to the transformer.
    - 64 parallel processing units each of which processing a patch.
    - Attention in each processing unit looks at its own patch and queries each of the remaining 63 processing units to ask for relevant info.



Resource:
https://towardsdatascience.com/efficient-image-segmentation-using-pytorch-part-4-6c86da083432