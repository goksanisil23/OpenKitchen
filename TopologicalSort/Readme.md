# Topological Sort
- Can only be used with *directed graphs* (graphs without any closed loops).
- Useful when ordering the steps of a process where certain steps depend on each other:
    - Choosing courses in university, where each course have certain pre-prequisites.
    - In backpropagation, taking derivatives of the nodes starting from the end (loss function).
- A graph can have more than 1 valid topological ordering.