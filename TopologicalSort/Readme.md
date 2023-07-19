# Topological Sort
- Can only be used with *directed graphs* (graphs without any closed loops).
- Useful when ordering the steps of a process where certain steps depend on each other:
    - Choosing courses in university, where each course have certain pre-prequisites.
    - In backpropagation, taking derivatives of the nodes starting from the end (loss function).
- A graph can have more than 1 valid topological ordering.

# Implementation
We have 2 implementations: recursive and iterative. The iterative approach is limited since in graphs with multiple root nodes, it cannot sort the root nodes correctly, because as soon as
initial root node is popped, we continue towards it's children, instead of treating the other root nodes as well.

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/TopologicalSort/resources/topo_sort_graph.png" width=50% height=50%>